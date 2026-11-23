#!/usr/bin/env python3
"""
Extract spans using GPT-5.5 via langextract.

Usage:
    # Quick test on 5 examples
    python scripts/llm_extraction/extract_gpt_spans.py --strategy few_shot --limit 5

    # Full zero-shot run
    python scripts/llm_extraction/extract_gpt_spans.py --strategy zero_shot

    # Full few-shot run (4 fixed examples)
    python scripts/llm_extraction/extract_gpt_spans.py --strategy few_shot

    # Evaluate results
    python scripts/llm_extraction/evaluate_spans.py results/generative/gpt5.5/few_shot.jsonl --relaxed

Requires:
    IAEDU_API_KEY env var
"""

import argparse
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.llm_extraction.gpt.extractor import GptSpanExtractor
from src.llm_extraction.gpt.config import GptConfig
from src.llm_extraction.shared.data_utils import load_jsonl, result_to_record, save_jsonl
from src.llm_extraction.shared.retry_policy import merge_all, summarise


_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_handler)

for _name in [
    "src.llm_extraction.gpt.extractor",
    "src.llm_extraction.gpt.langextract_provider",
    "src.llm_extraction.span_alignment",
    "src.llm_extraction.shared.data_utils",
]:
    _log = logging.getLogger(_name)
    _log.setLevel(logging.INFO)
    _log.addHandler(_handler)


def main():
    parser = argparse.ArgumentParser(description="Extract spans using GPT-5.5 via langextract")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="few_shot",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="GPT API key (falls back to IAEDU_API_KEY env var)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/citilink-votie/test.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output path (auto-generated if not set)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=5,
                        help="Fixed few-shot demonstrations (5 = the set reported in the paper)")
    parser.add_argument("--max-char-buffer", type=int, default=20000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between requests. The gateway rate-limits with an "
             "HTTP-200 'Rate limit reached (429)' event; pacing avoids it.",
    )
    parser.add_argument("--rate-limit-delay", type=int, default=30,
                        help="Initial backoff after a 429, doubled each retry (cap 600s)")
    parser.add_argument("--rate-limit-retries", type=int, default=6)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Send a temperature to the gateway. Off by default: the endpoint's "
             "handling of extra fields is untested. Probe with --limit 2 first.",
    )

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"results/generative/gpt5.5/{args.strategy}.jsonl"

    logger.info(f"Starting GPT-5.5 extraction: strategy={args.strategy}")

    config = GptConfig(
        api_key=args.api_key,
        max_char_buffer=args.max_char_buffer,
        timeout=args.timeout,
        temperature=args.temperature,
        request_interval=args.request_interval,
        rate_limit_delay=args.rate_limit_delay,
        rate_limit_retries=args.rate_limit_retries,
    )

    extractor = GptSpanExtractor(
        config=config,
        strategy=args.strategy,
        num_examples=args.num_examples,
    )

    logger.info(f"Loading test data from {args.test_file}")
    test_data = load_jsonl(args.test_file)

    if args.limit:
        test_data = test_data[: args.limit]
        logger.info(f"Limited to {args.limit} examples")

    # Resume from an existing output, as the Gemini and AMALIA scripts do.
    # Without this a multi-hour paid run lost everything to a dropped connection:
    # results were only written after the final document.
    #
    # Keep EVERY prior record, not just the successful ones. Dropping the errored
    # ones discards the failure taxonomy the paper reports (R7/A3) and lets a
    # network error overwrite a recorded model failure — see
    # src/llm_extraction/shared/retry_policy.py for the run where that happened.
    # It also silently misreports the run: this file held 149 rows and zero
    # errors after a session in which the gateway 429'd on nearly every request.
    results = []
    prior_results: dict = {}
    ordered_ids = [ex["id"] for ex in test_data]
    processed_ids = set()
    if Path(args.output_file).exists():
        existing = load_jsonl(args.output_file)
        prior_results = {r["id"]: r for r in existing}
        processed_ids = {r["id"] for r in existing if r.get("error") is None}
        prior_counts = summarise(existing)
        logger.info(
            f"Resuming from checkpoint: {len(processed_ids)} successful, "
            f"{prior_counts['model']} model failures, {prior_counts['infra']} infra failures "
            f"(all {len(existing) - len(processed_ids)} will be retried)"
        )
        if prior_counts["model"]:
            logger.info(
                f"  {prior_counts['model']} recorded model failures are protected: a retry that "
                f"fails on the network will not overwrite them."
            )
        stale = sum(1 for r in prior_results.values() if not r.get("diagnostics"))
        if stale:
            logger.warning(
                f"{stale}/{len(prior_results)} resumed records carry no diagnostics "
                f"(pre-instrumentation run). They will NOT be re-run, so cost and "
                f"span-discard figures would cover only part of the set. Write to a "
                f"fresh --output-file if you need a fully instrumented run."
            )

    pending = [ex for ex in test_data if ex["id"] not in processed_ids]
    logger.info(f"Remaining examples to process: {len(pending)}")

    def merged_records() -> list:
        """Prior + new, retry rules applied, back in gold order."""
        merged = merge_all(prior_results, results)
        return [merged[i] for i in ordered_ids if i in merged]

    for i, example in enumerate(pending, 1):
        logger.info(f"Processing {i}/{len(pending)}: {example['id']}")

        result = extractor.extract(
            text=example["text"],
            document_id=example["id"],
        )

        result_dict = result_to_record(result)

        results.append(result_dict)

        status = "OK" if result.error is None else f"ERROR: {result.error}"
        logger.info(f"  -> {len(result.entities)} entities [{status}]")

        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(pending)} examples")

        # Flush after EVERY document. The old interval of 25 was tuned for a run
        # that went at full speed; against the IAEDU gateway's rate limit a
        # document costs 10-15s of wall clock, so 25 of them is five minutes of
        # paid work sitting only in memory. save_jsonl writes to a temp file and
        # renames, so rewriting the whole file each time cannot corrupt it, and
        # the cost is ~2 MB of local I/O against a multi-second network call.
        save_jsonl(merged_records(), args.output_file)

    results = merged_records()
    counts = summarise(results)
    if prior_results:
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

    logger.info(f"Saving results to {args.output_file}")
    save_jsonl(results, args.output_file)

    total_entities = sum(len(r["entities"]) for r in results)
    avg_time = sum(r["processing_time"] for r in results) / len(results) if results else 0

    logger.info(f"\nExtraction complete!")
    logger.info(f"  Total: {len(results)}, Successful: {counts['ok']}, "
                f"Failed: {counts['model']} model + {counts['infra']} infra")
    logger.info(f"  Total entities: {total_entities}, Avg time: {avg_time:.2f}s")
    logger.info(f"  Output: {args.output_file}")


if __name__ == "__main__":
    main()
