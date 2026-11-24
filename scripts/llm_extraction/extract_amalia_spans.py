#!/usr/bin/env python3
"""
Extract spans using AMALIA API via langextract.

Uses langextract for prompt construction, text segmentation, and response
parsing. AMALIA's 32k-token context allows a generous max_char_buffer (default
20000), so most segments are processed in a single chunk without splitting.
Four fixed few-shot examples covering 11/12 entity types are included by default.

Usage:
    # Quick test on 5 examples
    python scripts/llm_extraction/extract_amalia_spans.py --strategy few_shot --limit 5

    # Full zero-shot run
    python scripts/llm_extraction/extract_amalia_spans.py --strategy zero_shot

    # Full few-shot run (4 fixed examples, default)
    python scripts/llm_extraction/extract_amalia_spans.py --strategy few_shot

    # Custom chunk size (reduce if hitting OOM or rate limits)
    python scripts/llm_extraction/extract_amalia_spans.py --strategy few_shot --max-char-buffer 10000

    # Evaluate results
    python scripts/llm_extraction/evaluate_spans.py results/generative/amalia/zero_shot.jsonl --relaxed
"""

import argparse
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Load AMALIA_BASE_URL from .env, like every other extraction script.
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.llm_extraction.amalia.extractor import AmaliaSpanExtractor
from src.llm_extraction.amalia.config import AmaliaConfig
from src.llm_extraction.shared.data_utils import load_jsonl, result_to_record, save_jsonl
from src.llm_extraction.shared.retry_policy import merge_all, summarise


# Configure logging before absl (used by langextract) can hijack it
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_handler)

# Also ensure the extractor and alignment loggers are visible
for _name in ['src.llm_extraction.amalia.extractor',
              'src.llm_extraction.span_alignment', 'src.llm_extraction.shared.data_utils']:
    _log = logging.getLogger(_name)
    _log.setLevel(logging.INFO)
    _log.addHandler(_handler)


def main():
    parser = argparse.ArgumentParser(description="Extract spans using AMALIA via langextract")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Extraction strategy",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="AMALIA API base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,  # auto-detected from /v1/models; records what actually served
        help="Model ID (auto-detected if not set)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/citilink-votie/test.jsonl",
        help="Path to test file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to output file (auto-generated if not set)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of fixed few-shot examples (max 5, covers 11 entity types + empty-extraction case)",
    )
    parser.add_argument(
        "--max-char-buffer",
        type=int,
        default=20000,
        help="Max characters per chunk for langextract segmentation (default: 20000 for AMALIA 32k context)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="Max completion tokens per chunk (default: 8192; input uses ~3200 tokens leaving ~29500 headroom in AMALIA's 32k context)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120; increase for long documents)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts on transient errors (default: 3; use 1 to disable retries)",
    )

    args = parser.parse_args()

    # Auto-generate output file name
    if args.output_file is None:
        args.output_file = f"results/generative/amalia/{args.strategy}.jsonl"

    logger.info(f"Starting AMALIA extraction: strategy={args.strategy}")

    # Create config
    config = AmaliaConfig(
        base_url=args.base_url,
        model_id=args.model,
        temperature=args.temperature,
        max_char_buffer=args.max_char_buffer,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    # Create extractor
    extractor = AmaliaSpanExtractor(
        config=config,
        strategy=args.strategy,
        num_examples=args.num_examples,
    )

    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    test_data = load_jsonl(args.test_file)

    if args.limit:
        test_data = test_data[:args.limit]
        logger.info(f"Limited to {args.limit} examples")

    # Resume: keep EVERY prior record, not just the successful ones, and re-run
    # only what errored. Keeping the failed records matters because merge_retry
    # refuses to let a network error overwrite a recorded model failure — see
    # src/llm_extraction/shared/retry_policy.py for why that rule exists.
    prior_results: dict = {}
    ordered_ids = [ex["id"] for ex in test_data]
    if Path(args.output_file).exists():
        existing = load_jsonl(args.output_file)
        prior_results = {r["id"]: r for r in existing}
        skip_ids = {r["id"] for r in existing if r.get("error") is None}
        test_data = [ex for ex in test_data if ex["id"] not in skip_ids]
        prior_counts = summarise(existing)
        logger.info(
            f"Resuming: {len(skip_ids)} already successful, "
            f"{prior_counts['model']} model failures, {prior_counts['infra']} infra failures "
            f"(all {len(existing) - len(skip_ids)} will be retried), {len(test_data)} pending"
        )
        if prior_counts["model"]:
            logger.info(
                f"  {prior_counts['model']} recorded model failures are protected: a retry that "
                f"fails on the network will not overwrite them."
            )
        stale = sum(1 for r in prior_results.values() if not r.get("diagnostics"))
        if stale:
            logger.warning(
                f"{stale}/{len(skip_ids)} resumed records carry no diagnostics "
                f"(pre-instrumentation run). They will NOT be re-run, so span-discard "
                f"figures would cover only part of the set. Write to a fresh "
                f"--output-file if you need a fully instrumented run."
            )

    # Extract spans
    results = []
    for i, example in enumerate(test_data, 1):
        logger.info(f"Processing {i}/{len(test_data)}: {example['id']}")

        result = extractor.extract(
            text=example["text"],
            document_id=example["id"],
        )

        # Convert to dict for JSON serialization (carries the diagnostics block)
        result_dict = result_to_record(result)

        results.append(result_dict)

        # Log progress
        n_entities = len(result.entities)
        status = "OK" if result.error is None else f"ERROR: {result.error}"
        logger.info(f"  -> {n_entities} entities extracted [{status}]")

        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(test_data)} examples")

        # Checkpoint after EVERY document, not every tenth. A single AMALIA
        # document costs 9s when it succeeds and up to 400s when it hits the
        # repetition loop, so a batch of ten can be an hour of work held only in
        # memory. save_jsonl writes to a temp file and renames it, so rewriting
        # the whole file each time cannot corrupt it, and local I/O is free
        # relative to the generation call.
        checkpoint = merge_all(prior_results, results)
        checkpoint_list = [checkpoint[id_] for id_ in ordered_ids if id_ in checkpoint]
        save_jsonl(checkpoint_list, args.output_file)

    # Merge with prior results and restore original order
    if prior_results:
        merged = merge_all(prior_results, results)
        results = [merged[id_] for id_ in ordered_ids if id_ in merged]
        counts = summarise(results)
        logger.info(
            f"After merge: {counts['ok']} ok, {counts['model']} model failures, "
            f"{counts['infra']} infra failures"
        )
        if counts["inconclusive_retries"]:
            logger.warning(
                f"{counts['inconclusive_retries']} documents kept their earlier model-failure "
                f"diagnosis because the retry failed on the network instead. Those retries were "
                f"inconclusive — check the connection and run again."
            )

    # Save results
    logger.info(f"Saving results to {args.output_file}")
    save_jsonl(results, args.output_file)

    # Summary
    successful = sum(1 for r in results if r["error"] is None)
    total_entities = sum(len(r["entities"]) for r in results)
    avg_time = (
        sum(r["processing_time"] for r in results) / len(results) if results else 0
    )

    logger.info("\nExtraction complete!")
    logger.info(f"  Total examples: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {len(results) - successful}")
    logger.info(f"  Total entities extracted: {total_entities}")
    logger.info(f"  Avg processing time: {avg_time:.2f}s")
    logger.info(f"  Output: {args.output_file}")


if __name__ == "__main__":
    main()
