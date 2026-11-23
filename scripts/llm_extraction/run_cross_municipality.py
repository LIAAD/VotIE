#!/usr/bin/env python3
"""
Run LOMO (Leave-One-Municipality-Out) cross-validation for a generative provider.

For each municipality (M01-M06):
- Tests on every unique segment of the held-out municipality (train + dev + test
  splits pooled and deduplicated, since the training split is oversampled).
- Uses demonstrations selected per fold from the five *training* municipalities
  only. The curated set in fixed_examples.py is unusable here: four of its five
  examples are gold documents from Porto and Covilha, which are themselves
  evaluated in their own folds.

Usage:
    python scripts/llm_extraction/run_cross_municipality.py --provider gemini
    python scripts/llm_extraction/run_cross_municipality.py --provider gpt --strategy zero_shot
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import langextract as lx

from src.llm_extraction.shared.data_utils import load_jsonl, result_to_record
from src.llm_extraction.shared.retry_policy import classify_failure, merge_retry, summarise
from src.llm_extraction.shared.evaluation import evaluate_span_extraction
from src.llm_extraction.schemas import SpanEntity, SpanExtractionResult
from src.llm_extraction.lomo_examples import (
    Selection,
    assert_no_leakage,
    select_lomo_examples,
)


# Per-provider request timeouts, mirroring the standard-benchmark scripts so a
# LOMO fold is not run under different settings than the Table 1 row it is
# compared against. AMALIA's 600 is deliberate: the published run used the
# library default of 120 and 49 of its 58 zero-shot failures were "Request timed
# out" at exactly 120.1 s — a measurement of our client, not of the model.
PROVIDER_TIMEOUTS = {"gemini": 180, "gpt": 120, "amalia": 600}

# Adapters served alongside the base model that must never be selected here.
# The vLLM endpoint lists LoRA adapters next to the base weights and the
# auto-detect takes the first entry, so a server-side reorder is all it would
# take to silently evaluate a CitiLink-tuned model on CitiLink data.
CONTAMINATED_MODEL_MARKERS = ("citilink", "claim", "persuasion")


def build_extractor(provider: str, model: str, temperature: float, max_char_buffer: int,
                    strategy: str, num_examples: int, timeout: int | None = None,
                    base_url: str | None = None, max_output_tokens: int | None = None):
    """Construct the extractor for one provider.

    Imported lazily so a run with one provider does not require the others'
    SDKs or API keys to be present.
    """
    timeout = timeout if timeout is not None else PROVIDER_TIMEOUTS[provider]

    if provider == "gemini":
        from src.llm_extraction.gemini.extractor import GeminiSpanExtractor
        from src.llm_extraction.gemini.config import GeminiConfig
        config = GeminiConfig(model_id=model, temperature=temperature,
                              max_char_buffer=max_char_buffer, timeout=timeout)
        return GeminiSpanExtractor(config=config, strategy=strategy,
                                   num_examples=num_examples)
    if provider == "gpt":
        from src.llm_extraction.gpt.extractor import GptSpanExtractor
        from src.llm_extraction.gpt.config import GptConfig
        # model_id was previously dropped on the floor here, so --model silently
        # did nothing and the run recorded whatever the config default was.
        kwargs = {"max_char_buffer": max_char_buffer, "timeout": timeout}
        if model:
            kwargs["model_id"] = model
        config = GptConfig(**kwargs)
        return GptSpanExtractor(config=config, strategy=strategy,
                                num_examples=num_examples)
    if provider == "amalia":
        from src.llm_extraction.amalia.extractor import AmaliaSpanExtractor
        from src.llm_extraction.amalia.config import AmaliaConfig
        kwargs = {"temperature": temperature, "max_char_buffer": max_char_buffer,
                  "timeout": timeout}
        if model:
            kwargs["model_id"] = model
        if base_url:
            kwargs["base_url"] = base_url
        if max_output_tokens:
            kwargs["max_output_tokens"] = max_output_tokens
        config = AmaliaConfig(**kwargs)
        extractor = AmaliaSpanExtractor(config=config, strategy=strategy,
                                        num_examples=num_examples)
        assert_clean_model(getattr(config, "model_id", None) or getattr(extractor, "model_id", None))
        return extractor
    raise ValueError(f"Unknown provider: {provider}")


def assert_clean_model(model_id: str | None) -> None:
    """Stop before spending GPU-days on a fine-tuned adapter.

    The AMALIA endpoint serves `amalia-base` next to LoRA adapters including
    `amalia_citilink_theme` and `amalia_citilink_summ`, which are tuned on the
    very corpus this paper evaluates. Auto-detection picks whichever the server
    lists first, so the only thing standing between the published "AMALIA 8B
    zero-shot" number and a contaminated one is list ordering.
    """
    if not model_id:
        logger.warning("  [WARN] could not determine the AMALIA model id; "
                       "verify it is the base model before trusting these results")
        return
    lowered = model_id.lower()
    if any(marker in lowered for marker in CONTAMINATED_MODEL_MARKERS):
        raise SystemExit(
            f"[FAIL] refusing to run: model id {model_id!r} looks like a fine-tuned adapter.\n"
            "  The AMALIA endpoint serves CitiLink-tuned LoRA adapters beside the base\n"
            "  weights; evaluating one of them on CitiLink data is contamination.\n"
            "  Pass the base model explicitly:  --model amalia-base"
        )
    logger.info(f"  [OK] model id: {model_id}")



# Map LOMO codes -> real municipality ID prefixes used in citilink-votie/{train,dev,test}.jsonl.
# Ordering follows scripts/dataset_generation/create_lomo_splits_v6.py.
MUNICIPALITY_MAP = {
    'M01': 'Alandroal',
    'M02': 'Campomaior',
    'M03': 'Covilha',
    'M04': 'Fundao',
    'M05': 'Guimaraes',
    'M06': 'Porto',
}


# Configure logging before absl (used by langextract) can hijack it
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_handler)

for _name in [
    'src.llm_extraction.gemini.extractor',
    'src.llm_extraction.gpt.extractor',
    'src.llm_extraction.amalia.extractor',
    'src.llm_extraction.span_alignment',
    'src.llm_extraction.shared.data_utils',
    'src.llm_extraction.lomo_examples',
]:
    _log = logging.getLogger(_name)
    _log.setLevel(logging.INFO)
    _log.addHandler(_handler)


def filter_by_municipality(data: List[dict], municipality_code: str) -> List[dict]:
    """Filter data for specific municipality (accepts M01..M06 codes or real names).

    Prefers the explicit ``municipality`` field and falls back to the id prefix.
    """
    prefix = MUNICIPALITY_MAP.get(municipality_code, municipality_code)
    return [
        ex for ex in data
        if ex.get('municipality') == municipality_code
        or (ex.get('municipality') is None and ex.get('id', '').startswith(prefix + '_'))
    ]


def deduplicate(data: List[dict]) -> List[dict]:
    """Drop repeated segment ids, keeping first occurrence.

    The training split is oversampled — 2,463 rows over 1,803 unique ids, with
    multi-vote and minority-type segments duplicated 5x or 9x (Appendix B.5).
    A LOMO fold pools train+dev+test of the held-out municipality, so without
    this the duplicates would be scored repeatedly (over-weighting exactly the
    hard cases) and re-sent to the API, inflating both the metric and the bill.
    Deduplicating reproduces the documented fold sizes: 504/398/720/235/556/472.
    """
    seen = set()
    unique = []
    for example in data:
        if example['id'] in seen:
            continue
        seen.add(example['id'])
        unique.append(example)
    return unique


ENCODER_LOMO_SPLITS = Path(__file__).parent.parent.parent / "data" / "citilink-votie-lomo-splits"


def assert_matches_encoder_fold(municipality: str, test_data: List[dict]) -> None:
    """Refuse to spend API budget unless this fold is the encoders' fold.

    Table 2 puts generative and encoder rows in the same column, which is only
    meaningful if both were scored on the same documents against the same gold.
    That was silently false for a full submission cycle: an earlier version of
    this script built folds with its own segmentation and id scheme, so the
    published Gemini M06 row was scored against 366 gold entities where the
    encoders saw 1,786 — same 472 documents, a fifth of the annotations, and
    three entity types missing entirely. The document count matched, which is
    why the audit passed; nothing compared the gold.

    Checking ids and gold-span totals against the encoder split is the check that
    would have caught it, and it costs milliseconds before a paid run.
    """
    reference = ENCODER_LOMO_SPLITS / municipality / "test.jsonl"
    if not reference.exists():
        logger.warning(
            f"  [WARN] no encoder split at {reference}; cannot verify fold "
            f"commensurability. Table 2 rows may not be comparable."
        )
        return

    encoder = [json.loads(line) for line in reference.open(encoding="utf-8")]
    enc_ids = {ex["id"] for ex in encoder}
    our_ids = {ex["id"] for ex in test_data}
    enc_gold = sum(len(ex.get("spans") or []) for ex in encoder)
    our_gold = sum(len(ex.get("spans") or []) for ex in test_data)

    problems = []
    if our_ids != enc_ids:
        problems.append(
            f"document ids differ: {len(our_ids - enc_ids)} only here, "
            f"{len(enc_ids - our_ids)} only in the encoder split"
        )
    if our_gold != enc_gold:
        problems.append(f"gold span count differs: {our_gold} here vs {enc_gold} in the encoder split")

    if problems:
        raise SystemExit(
            f"[FAIL] {municipality} fold does not match the encoder LOMO split:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + f"\n  reference: {reference}\n"
            "  Results from this fold would not be comparable to the encoder rows of "
            "Table 2. Fix the fold construction before spending API budget."
        )

    logger.info(
        f"  [OK] fold matches the encoder LOMO split: {len(our_ids)} documents, "
        f"{our_gold} gold spans"
    )


def build_lomo_examples(
    held_out: str, num_examples: int = 5
) -> Tuple[List[lx.data.ExampleData], List[dict]]:
    """Select few-shot demonstrations that exclude the held-out municipality.

    The curated set in `fixed_examples.py` must NOT be used here: four of its
    five examples are gold documents from Porto (M06) and Covilha (M03), and a
    LOMO fold evaluates every split of the held-out municipality — so for those
    folds the prompt would carry test documents together with their answers.
    """
    examples, provenance = select_lomo_examples(held_out, num_examples=num_examples)
    return examples, [p._asdict() for p in provenance]


def run_single_experiment(
    test_municipality: str,
    train_municipalities: List[str],
    train_file: str,
    dev_file: str,
    test_file: str,
    output_dir: Path,
    extractor,
    model_id: str,
    strategy: str,
    provider: str,
    retry_infra_only: bool = False,
) -> Dict:
    """
    Run single LOMO experiment: test on the held-out municipality.

    Strategy:
    - Few-shot examples are selected per fold from the five training
      municipalities only (see build_lomo_examples).
    - Test set: ALL examples from held-out municipality (train + dev + test splits).
    """
    logger.info("=" * 80)
    logger.info(f"LOMO Experiment: Test on {test_municipality} ({MUNICIPALITY_MAP.get(test_municipality, test_municipality)})")
    logger.info("=" * 80)
    logger.info(
        "Train municipalities: "
        + ', '.join(f"{c}={MUNICIPALITY_MAP.get(c, c)}" for c in train_municipalities)
    )

    # Select demonstrations for this fold. langextract requires >=1 example
    # unconditionally, so zero-shot is really one scaffold — and that scaffold
    # has to respect the fold boundary too.
    n_examples = 5 if strategy == "few_shot" else 1
    lx_examples, example_provenance = build_lomo_examples(test_municipality, n_examples)

    # Hard stop rather than a silent bad run: this is the check whose absence
    # let leaked prompts reach the previously published LOMO results.
    assert_no_leakage(test_municipality, [Selection(**p) for p in example_provenance])

    extractor.lx_examples = lx_examples
    logger.info(f"Selected {len(lx_examples)} demonstrations, none from {test_municipality}:")
    for p in example_provenance:
        logger.info(
            f"    {p['phenomenon']:<24} {p['tier']:<8} {p['municipality']}  {p['id']}"
        )

    logger.info(f"Loading ALL examples from {test_municipality} (train + dev + test splits)...")
    all_train_data = load_jsonl(train_file)
    all_dev_data = load_jsonl(dev_file)
    all_test_data = load_jsonl(test_file)

    train_examples = filter_by_municipality(all_train_data, test_municipality)
    dev_examples = filter_by_municipality(all_dev_data, test_municipality)
    test_examples = filter_by_municipality(all_test_data, test_municipality)
    test_data = deduplicate(train_examples + dev_examples + test_examples)

    logger.info(f"  Train split: {len(train_examples)} rows")
    logger.info(f"  Dev split:   {len(dev_examples)} rows")
    logger.info(f"  Test split:  {len(test_examples)} rows")
    logger.info(
        f"  TOTAL:       {len(test_data)} unique segments "
        f"({len(train_examples) + len(dev_examples) + len(test_examples) - len(test_data)} "
        f"oversampling duplicates dropped)"
    )

    if len(test_data) == 0:
        logger.error(f"No examples found for {test_municipality}")
        return None

    assert_matches_encoder_fold(test_municipality, test_data)

    predictions_file = output_dir / f'{provider}_{strategy}_{test_municipality}.jsonl'
    existing_predictions: Dict[str, dict] = {}

    # Prior records for documents that failed. Kept, never deleted: the failure
    # rate is a reported result, and a retry that dies on the
    # network must not be allowed to overwrite a recorded model failure — see
    # src/llm_extraction/shared/retry_policy.py.
    prior_failures: Dict[str, dict] = {}

    if predictions_file.exists():
        logger.info(f"\nFound existing predictions at {predictions_file}")
        logger.info("Loading existing predictions to resume from checkpoint...")
        try:
            all_prior = []
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    pred = json.loads(line.strip())
                    all_prior.append(pred)
                    if pred.get('error') is None:
                        existing_predictions[pred['id']] = pred
                    else:
                        prior_failures[pred['id']] = pred
            counts = summarise(all_prior)
            logger.info(
                f"Loaded {counts['ok']} successful predictions; "
                f"{counts['model']} model failures and {counts['infra']} infra failures "
                f"will be retried"
            )
            if counts['model']:
                logger.info(
                    f"  {counts['model']} recorded model failures are protected: a retry that "
                    f"fails on the network will not overwrite them."
                )
        except Exception as e:
            logger.warning(f"Error loading existing predictions: {e}")
            logger.warning("Starting fresh...")
            existing_predictions = {}
            prior_failures = {}

        # Rewrite keeping BOTH successes and failures. This used to drop the
        # errored entries "so retries write cleanly", which silently deleted the
        # failure taxonomy: after one resume, M02 held 122 rows and zero errors
        # where the run had actually recorded 61 failures.
        keep = list(existing_predictions.values()) + list(prior_failures.values())
        if keep:
            tmp = predictions_file.with_name(predictions_file.name + '.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                for pred in keep:
                    f.write(json.dumps(pred, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, predictions_file)
        else:
            predictions_file.unlink()

    logger.info(f"\nRunning predictions on {len(test_data)} examples...")
    # One record per document id, so a retried failure replaces its earlier
    # record instead of appending a second line with the same id.
    records: Dict[str, dict] = dict(existing_predictions)
    records.update(prior_failures)
    order = [ex['id'] for ex in test_data]
    errors = 0
    skipped = 0

    def flush_predictions() -> None:
        """Rewrite the file atomically, in gold order. Cheap next to an API call."""
        tmp = predictions_file.with_name(predictions_file.name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            for doc_id in order:
                if doc_id in records:
                    f.write(json.dumps(records[doc_id], ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, predictions_file)

    for i, example in enumerate(test_data, 1):
        if example['id'] in existing_predictions:
            skipped += 1
            continue
        # A recorded model failure is a result; re-asking would give this fold an
        # extra attempt the folds that never resumed did not get.
        prior = prior_failures.get(example['id'])
        if retry_infra_only and prior and classify_failure(prior.get('error')) == 'model':
            skipped += 1
            continue

        logger.info(f"Processing {i}/{len(test_data)}: {example['id']}")

        result = extractor.extract(
            text=example['text'],
            document_id=example['id'],
        )

        pred_dict = result_to_record(
            result,
            test_municipality=test_municipality,
            train_municipalities=train_municipalities,
        )

        # merge_retry refuses to let a network error overwrite a recorded model
        # failure, and maintains `attempts` — the failure rate is only
        # interpretable as "failed on all N attempts".
        records[example['id']] = merge_retry(prior_failures.get(example['id']), pred_dict)

        if records[example['id']].get('error'):
            errors += 1

        flush_predictions()

    predictions = [records[doc_id] for doc_id in order if doc_id in records]
    counts = summarise(predictions)
    logger.info(
        f"Completed predictions. Total: {len(test_data)}, "
        f"New: {len(test_data) - skipped}, Skipped: {skipped}, "
        f"Errors: {counts['model']} model + {counts['infra']} infra"
    )
    if counts['inconclusive_retries']:
        logger.warning(
            f"{counts['inconclusive_retries']} documents kept their earlier model-failure "
            f"diagnosis because the retry failed on the network instead. Those retries were "
            f"inconclusive — check the connection and run again."
        )
    logger.info(f"All predictions saved to {predictions_file}")

    logger.info("\nEvaluating predictions...")
    pred_results = []
    for pred in predictions:
        entities = [
            SpanEntity(
                text=e['text'],
                type=e['type'],
                start=e['start'],
                end=e['end'],
            )
            for e in pred.get('entities', [])
        ]
        pred_results.append(SpanExtractionResult(
            id=pred['id'],
            text=pred['text'],
            entities=entities,
            model=pred['model'],
            strategy=pred['strategy'],
            processing_time=pred.get('processing_time', 0.0),
            error=pred.get('error'),
        ))

    evaluation_results = evaluate_span_extraction(
        predictions=pred_results,
        ground_truth=test_data,
        compute_fidelity=True,
        relaxed_matching=True,
    )

    evaluation_results['metadata'] = {
        'test_municipality': test_municipality,
        'train_municipalities': train_municipalities,
        'provider': provider,
        'model_id': model_id,
        'strategy': strategy,
        'num_test_examples': len(test_data),
        'num_errors': errors,
        # Provenance of every demonstration, so the fold's prompt is auditable
        # and the absence of held-out material is checkable after the fact.
        'few_shot_examples': example_provenance,
        'predictions_file': str(predictions_file),
        'timestamp': datetime.now().isoformat(),
    }

    eval_file = output_dir / f'{provider}_{strategy}_{test_municipality}_evaluation.json'
    logger.info(f"Saving evaluation to {eval_file}")

    def convert_numpy_types(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    evaluation_results_serializable = convert_numpy_types(evaluation_results)
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results_serializable, f, indent=2, ensure_ascii=False)

    entity_metrics = evaluation_results['entity_level']
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RESULTS: {test_municipality}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Precision: {entity_metrics['precision']:.4f}")
    logger.info(f"Recall:    {entity_metrics['recall']:.4f}")
    logger.info(f"F1:        {entity_metrics['f1']:.4f}")
    logger.info(f"{'=' * 80}\n")

    return evaluation_results


def _counts_from_per_type(per_type: Dict[str, dict]) -> Tuple[int, int, int, int]:
    """``(gold, TP, FN, FP)`` reconstructed from per-type precision/recall/support.

    The evaluator stores only precision/recall/f1 under ``entity_level`` — never
    raw counts — so this table used to raise ``KeyError: 'true_positives'`` after
    the last fold, *after* every evaluation file had already been written. The
    counts are recoverable exactly: ``TP = recall x support``, ``FP = TP/precision
    - TP``, ``FN = support - TP``. Rounded once at the end so the columns sum.
    """
    gold = tp = fp = 0.0
    for name, m in per_type.items():
        if name in ("micro avg", "macro avg", "weighted avg"):
            continue
        support = m.get("support", 0)
        if support <= 0:
            continue
        precision = m.get("precision", 0.0)
        recall = m.get("recall", 0.0)
        hit = recall * support
        gold += support
        tp += hit
        if precision > 0:
            fp += hit / precision - hit
    return round(gold), round(tp), round(gold - tp), round(fp)


def generate_summary_table(
    results: Dict[str, Dict],
    output_dir: Path,
    model_id: str,
    strategy: str,
    num_examples: int,
    provider: str,
):
    """Generate comparative table of LOMO results across all municipalities."""
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SUMMARY TABLE")
    logger.info("=" * 80)

    municipalities = sorted(results.keys())

    lines = []
    lines.append(f"# LOMO Cross-Validation Results — {provider} ({model_id})\n\n")
    lines.append("## Leave-One-Municipality-Out Generalization\n\n")
    lines.append(
        "Each row shows results when testing on the held-out municipality. "
        "Few-shot demonstrations are selected per fold from the five training "
        "municipalities only; none originates in the held-out municipality. "
        "Provenance for every demonstration is recorded in the per-fold evaluation JSON.\n\n"
    )

    lines.append("### Entity-Level Span Extraction Results\n\n")
    lines.append("| Test Municipality | Examples | Precision | Recall | F1 | Gold | TP | FN | FP |\n")
    lines.append("|-------------------|----------|-----------|--------|-----|------|----|----|----|\n")

    for muni in municipalities:
        if results[muni] is None:
            continue
        entity_metrics = results[muni]['entity_level']
        metadata = results[muni]['metadata']
        gold, tp, fn, fp = _counts_from_per_type(results[muni].get('per_type_metrics') or {})
        lines.append(
            f"| {muni} | {metadata['num_test_examples']} | "
            f"{entity_metrics['precision']:.3f} | "
            f"{entity_metrics['recall']:.3f} | "
            f"{entity_metrics['f1']:.3f} | "
            f"{gold} | {tp} | {fn} | {fp} |\n"
        )

    successful_results = [r for r in results.values() if r]
    avg_p = avg_r = avg_f1 = 0.0
    if successful_results:
        avg_p = sum(r['entity_level']['precision'] for r in successful_results) / len(successful_results)
        avg_r = sum(r['entity_level']['recall'] for r in successful_results) / len(successful_results)
        avg_f1 = sum(r['entity_level']['f1'] for r in successful_results) / len(successful_results)
        lines.append(
            f"| **Average** | - | **{avg_p:.3f}** | **{avg_r:.3f}** | **{avg_f1:.3f}** | - | - | - |\n"
        )
    else:
        logger.warning("No successful results to calculate averages")
        lines.append("| **Average** | - | N/A | N/A | N/A | - | - | - |\n")

    lines.append("\n### Per-Entity-Type F1 Scores\n\n")
    lines.append("| Entity Type | " + " | ".join(municipalities) + " | Avg |\n")
    lines.append("|-------------|" + "|".join(["-------"] * len(municipalities)) + "|-----|\n")

    # per_type_metrics sits at the TOP level of the evaluation JSON, not under
    # entity_level. Reading the wrong path left entity_types empty, so this table
    # rendered as a header with no rows and no error — the quiet counterpart of
    # the KeyError above.
    entity_types = sorted({
        name
        for muni in municipalities
        if results[muni]
        for name in (results[muni].get('per_type_metrics') or {})
        if name not in ("micro avg", "macro avg", "weighted avg")
    })

    for etype in entity_types:
        cells, present = [], []
        for muni in municipalities:
            per_type = (results[muni] or {}).get('per_type_metrics') or {}
            metrics = per_type.get(etype)
            # A type absent from a fold has no support there; scoring it 0.0
            # would understate the average, so it is left out of the mean and
            # shown as "--" rather than as a failure.
            if metrics and metrics.get('support', 0) > 0:
                f1 = metrics.get('f1', metrics.get('f1_score', 0.0))
                cells.append(f"{f1:.3f}")
                present.append(f1)
            else:
                cells.append("--")
        avg = f"**{sum(present) / len(present):.3f}**" if present else "--"
        lines.append(f"| {etype} | " + " | ".join(cells) + f" | {avg} |\n")

    # Same class of bug: the key is 'alignment_fidelity', and its fields are
    # total_spans / aligned_spans / alignment_fidelity. Under the old names this
    # table was always empty, which would have been read as "nothing to report"
    # rather than "never looked". Document-level failures are shown alongside,
    # since a discarded span and a document that produced none are different
    # losses.
    lines.append("\n### Alignment Fidelity\n\n")
    lines.append("| Municipality | Spans Extracted | Aligned | Alignment Rate | Failed Docs |\n")
    lines.append("|--------------|-----------------|---------|----------------|-------------|\n")
    for muni in municipalities:
        fidelity = (results[muni] or {}).get('alignment_fidelity')
        if not fidelity:
            continue
        total = fidelity.get('total_spans', 0)
        aligned = fidelity.get('aligned_spans', 0)
        rate = fidelity.get('alignment_fidelity', 0.0)
        failed_docs = fidelity.get('documents_with_errors', 0)
        lines.append(f"| {muni} | {total} | {aligned} | {rate:.3f} | {failed_docs} |\n")

    lines.append("\n### Experimental Setup\n\n")
    lines.append(f"- **Model**: {model_id}\n")
    lines.append(f"- **Pipeline**: langextract ({provider})\n")
    lines.append(f"- **Strategy**: {strategy} (1 example per training municipality, 5 total)\n")
    lines.append("- **Example Selection**: Per-fold, one demonstration per phenomenon, "
                 "drawn only from the five training municipalities (see src/llm_extraction/lomo_examples.py)\n")
    lines.append("- **Evaluation**: Character-level span matching (exact match)\n")
    lines.append("- **Data Source**: `data/citilink-votie/{train,dev,test}.jsonl`\n\n")

    summary_file = output_dir / f'{provider}_lomo_summary.md'
    logger.info(f"Saving summary to {summary_file}")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))

    summary_json = {
        'model_id': model_id,
        'strategy': strategy,
        'num_examples': num_examples,
        'average_metrics': {
            'precision': avg_p,
            'recall': avg_r,
            'f1': avg_f1,
        },
        'per_municipality': {
            muni: {
                'precision': results[muni]['entity_level']['precision'],
                'recall': results[muni]['entity_level']['recall'],
                'f1': results[muni]['entity_level']['f1'],
                'examples': results[muni]['metadata']['num_test_examples'],
            }
            for muni in municipalities if results[muni]
        },
    }

    summary_json_file = output_dir / f'{provider}_lomo_summary.json'
    with open(summary_json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved JSON summary to {summary_json_file}")

    logger.info("\n" + "=" * 80)
    logger.info("LOMO CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    for line in lines[4:]:
        if line.strip():
            logger.info(line.strip())
    logger.info("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete LOMO cross-validation for one generative provider'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['gemini', 'gpt', 'amalia'],
        default='gemini',
        help='Generative provider to evaluate',
    )
    parser.add_argument(
        '--municipalities',
        nargs='+',
        default=['M01', 'M02', 'M03', 'M04', 'M05', 'M06'],
        help='Municipalities to test (default: all M01-M06)',
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['zero_shot', 'few_shot'],
        default='few_shot',
        help='Extraction strategy',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model ID (default: the provider default — gemini-2.5-pro, gpt-5.5, '
             'or AMALIA auto-detected from /v1/models)',
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=5,
        help='Demonstrations per fold, one per phenomenon, selected from the five '
             'training municipalities (see src/llm_extraction/lomo_examples.py)',
    )
    parser.add_argument(
        '--max-char-buffer',
        type=int,
        default=20000,
        help='Max characters per chunk for langextract segmentation',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature',
    )
    parser.add_argument(
        '--train-file',
        default='data/citilink-votie/train.jsonl',
        help='Training data file (for evaluation)',
    )
    parser.add_argument(
        '--dev-file',
        default='data/citilink-votie/dev.jsonl',
        help='Dev data file (for evaluation)',
    )
    parser.add_argument(
        '--test-file',
        default='data/citilink-votie/test.jsonl',
        help='Test data file (for evaluation)',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: results/generative/lomo/<provider>)',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip municipalities with existing evaluation files',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Request timeout in seconds. Default is per provider '
             f'({PROVIDER_TIMEOUTS}), matching the standard-benchmark scripts so a '
             'LOMO fold runs under the same settings as the Table 1 row it is '
             'compared against.',
    )
    parser.add_argument(
        '--base-url',
        default=None,
        help='Override the AMALIA endpoint (default: AMALIA_BASE_URL, else http://localhost:8000)',
    )
    parser.add_argument(
        '--max-output-tokens',
        type=int,
        default=None,
        help='Max output tokens per chunk (AMALIA; default 8192, a circuit breaker '
             'on runaway generation rather than a budget)',
    )

    parser.add_argument(
        '--retry-infra-only',
        action='store_true',
        help='On resume, retry only infrastructure failures (connection, 5xx, 429); '
             'keep recorded model failures as they are',
    )

    args = parser.parse_args()

    PROVIDER_DEFAULT_MODEL = {'gemini': 'gemini-2.5-pro', 'gpt': 'gpt-5.5',
                              'amalia': 'amalia-base'}
    if args.model is None:
        args.model = PROVIDER_DEFAULT_MODEL[args.provider]
    if args.output_dir is None:
        args.output_dir = f'results/generative/lomo/{args.provider}'

    logger.info("=" * 80)
    logger.info(f"LOMO CROSS-VALIDATION — {args.provider.upper()}")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model or '(auto-detected from the server)'}")
    logger.info(f"Strategy: {args.strategy} ({args.num_examples} per-fold demonstrations)")
    logger.info(f"Municipalities: {', '.join(args.municipalities)}")
    logger.info(f"Train file: {args.train_file}")
    logger.info(f"Dev file: {args.dev_file}")
    logger.info(f"Test file: {args.test_file}")
    logger.info("Evaluation strategy: ALL splits (train+dev+test) from held-out municipality, deduplicated")
    logger.info("Few-shot: selected per fold from the five training municipalities only")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("=" * 80 + "\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One extractor for all folds; its lx_examples are replaced per fold, since
    # the demonstrations are municipality-dependent.
    extractor = build_extractor(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_char_buffer=args.max_char_buffer,
        strategy=args.strategy,
        num_examples=args.num_examples,
        timeout=args.timeout,
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
    )
    resolved_model = getattr(extractor, 'model_id', args.model)
    if resolved_model != args.model:
        logger.info(f"Server resolved model id: {resolved_model}")
    # Re-check after the server has had its say: auto-detection happens inside the
    # extractor, so the id that actually gets used may differ from the one requested.
    assert_clean_model(resolved_model)

    all_municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']

    results = {}

    for test_muni in args.municipalities:
        eval_file = output_dir / f'{args.provider}_{args.strategy}_{test_muni}_evaluation.json'
        if args.skip_existing and eval_file.exists():
            logger.info(f"Skipping {test_muni} (evaluation file already exists)")
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    results[test_muni] = json.load(f)
                continue
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted JSON file for {test_muni}: {e}")
                logger.warning(f"Deleting corrupted file and re-running: {eval_file}")
                eval_file.unlink()

        train_munis = [m for m in all_municipalities if m != test_muni]

        try:
            result = run_single_experiment(
                test_municipality=test_muni,
                train_municipalities=train_munis,
                train_file=args.train_file,
                dev_file=args.dev_file,
                test_file=args.test_file,
                output_dir=output_dir,
                extractor=extractor,
                model_id=resolved_model,
                strategy=args.strategy,
                provider=args.provider,
                retry_infra_only=args.retry_infra_only,
            )
            results[test_muni] = result
        except Exception as e:
            logger.error(f"Failed to run experiment for {test_muni}: {e}")
            results[test_muni] = None

    generate_summary_table(results, output_dir, resolved_model, args.strategy,
                           args.num_examples, args.provider)

    logger.info("\n" + "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {output_dir / (args.provider + '_lomo_summary.md')}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
