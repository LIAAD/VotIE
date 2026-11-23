#!/usr/bin/env python3
"""Paired bootstrap tests between encoder and generative baselines (in-domain).

``scripts/statistical_tests.py`` compares the three encoder-CRF models against
each other. It cannot compare an encoder against Gemini/AMALIA/GPT because the
two paradigms store predictions in different formats: encoders write subword
offsets plus BIO labels, generative runs write character spans. This script
brings both onto the same per-document TP/FP/FN representation and runs the same
Berg-Kirkpatrick (2012) paired bootstrap over it.

Why the comparison is well-posed
--------------------------------
``src/llm_extraction/shared/evaluation.py:spans_to_bio_labels`` converts both a
generative model's spans *and* the gold spans to BIO using
``src.data.span_to_bio.tokenize_text_with_offsets`` — the same tokenizer that
produced the gold BIO the encoders are scored against. So both paradigms are
scored as entity sets over the same document, and per-document counts are
directly poolable.

Two properties are enforced rather than assumed:

  * **Shared type axis.** Counts are projected onto the 11-type schema
    (``src.llm_extraction.schemas.ENTITY_TYPES``; ``Count-Against`` excluded, as
    everywhere else) so that column *j* means the same type on both sides.
    ``statistical_tests.py`` builds its axis from observed types, which is safe
    between encoders but not across paradigms.
  * **Reproduction of the reported numbers.** ``--check`` re-derives each side's
    macro F1 from the counts and compares it against the cached evaluation JSON,
    so a mismatch surfaces as an error instead of a plausible-looking p-value.

The generative run is a single sample, so pooling across the encoder's three
seeds holds the generative counts fixed and averages the three per-seed deltas
per bootstrap iteration — the same statistic ``paired_bootstrap_test_pooled``
computes for encoder pairs.

Usage
-----
    python scripts/statistical_tests_generative.py
    python scripts/statistical_tests_generative.py --n-bootstrap 10000
    python scripts/statistical_tests_generative.py --encoders deberta_crf \
        --generative gemini:few_shot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.statistical_tests import (  # noqa: E402
    SEEDS,
    bootstrap_ci,
    in_domain_pred,
    load_counts,
    macro_f1_from_counts,
    paired_bootstrap_test,
    paired_bootstrap_test_pooled,
    per_doc_counts,
)
from src.llm_extraction.schemas import ENTITY_TYPES, SpanEntity  # noqa: E402
from src.llm_extraction.shared.evaluation import (  # noqa: E402
    overlap_length,
    relaxed_span_match,
    spans_to_bio_labels,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("stat_gen")

RESULTS_DIR = ROOT / "results"
GENERATIVE_ROOTS = [RESULTS_DIR / "generative"]
GOLD_TEST = ROOT / "data" / "citilink-votie" / "test.jsonl"
OUT_PATH = RESULTS_DIR / "statistical_tests_generative.json"


def resolve_generative(provider: str, strategy: str) -> Optional[Path]:
    """Predictions file for this run, required to sit next to its evaluation JSON
    so that an unfinished run (predictions but no evaluation yet) is never used.
    """
    for root in GENERATIVE_ROOTS:
        preds = root / provider / f"{strategy}.jsonl"
        if preds.exists() and preds.with_name(f"{strategy}_evaluation.json").exists():
            return preds
    return None

# Canonical axis. Sorted so the ordering is stable and independent of the order
# the schema happens to declare.
CANONICAL_TYPES: List[str] = sorted(ENTITY_TYPES)

DEFAULT_ENCODERS = ["deberta_crf", "xlmr_crf", "bert_crf"]
DEFAULT_GENERATIVE = [
    ("gemini", "few_shot"),
    ("gemini", "zero_shot"),
    ("gpt5.5", "few_shot"),
    ("gpt5.5", "zero_shot"),
    ("amalia", "few_shot"),
    ("amalia", "zero_shot"),
]


# --------------------------------------------------------------------- #
# Type-axis projection                                                  #
# --------------------------------------------------------------------- #

def project(counts: np.ndarray, types: List[str]) -> np.ndarray:
    """Re-index a (n_docs, n_types, 3) array onto CANONICAL_TYPES.

    Types outside the schema are dropped — they cannot be scored against a gold
    standard that never contains them, and both paradigms already filter them
    upstream (the generative aligner rejects out-of-schema labels; the encoders
    cannot emit them). Anything dropped is logged rather than silently absorbed.
    """
    out = np.zeros((counts.shape[0], len(CANONICAL_TYPES), 3), dtype=np.int64)
    idx = {t: i for i, t in enumerate(CANONICAL_TYPES)}
    dropped: Dict[str, int] = {}
    for j, t in enumerate(types):
        k = idx.get(t)
        if k is None:
            total = int(counts[:, j, :].sum())
            if total:
                dropped[t] = total
            continue
        out[:, k, :] += counts[:, j, :]
    if dropped:
        logger.warning("Dropped out-of-schema types: %s", dropped)
    return out


def reindex(run: dict) -> dict:
    """Return *run* with both count arrays on the canonical type axis."""
    return {
        "ids": run["ids"],
        "exact": project(run["exact"], run["exact_types"]),
        "relaxed": project(run["relaxed"], run["relaxed_types"]),
        "types": CANONICAL_TYPES,
    }


# --------------------------------------------------------------------- #
# Generative predictions -> per-document counts                         #
# --------------------------------------------------------------------- #

def load_gold(path: Path) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def _strict_entities(labels: List[str]) -> List[Tuple[str, int, int]]:
    """Chunk a BIO sequence with seqeval's strict IOB2 scheme.

    ``statistical_tests.py`` uses the loose ``get_entities`` chunker, which is
    what the encoder evaluator effectively uses too: ``entity_metrics.py`` passes
    ``scheme=IOB2`` without ``mode='strict'``, and seqeval ignores ``scheme``
    unless strict mode is on. The generative evaluator does pass
    ``mode='strict'``, and the difference is not cosmetic here — generative
    models emit *nested* spans (a \\textsc{Voter-Abstention} inside a
    \\textsc{Counting-Majority}), and flattening those into one BIO array leaves
    an orphan ``I-`` tag where the outer span resumes. The loose chunker scores
    that orphan as a second entity, the strict one discards it.

    So each paradigm is chunked the way its own reported number is chunked. The
    alternative — one chunker for both — would produce p-values for a statistic
    that appears in no table.
    """
    entities: List[Tuple[str, int, int]] = []
    start: Optional[int] = None
    tag: Optional[str] = None
    for i, label in enumerate(labels + ["O"]):
        prefix, _, name = label.partition("-")
        if prefix == "I" and tag == name and start is not None:
            continue
        if start is not None:
            entities.append((tag, start, i - 1))
            start, tag = None, None
        if prefix == "B":
            start, tag = i, name
        # A bare I- that does not continue an open chunk is invalid under IOB2
        # and is dropped, which is exactly what seqeval's strict mode does.
    return entities


def _exact_counts_strict(
    pred_labels: List[str], gold_labels: List[str]
) -> Dict[str, List[int]]:
    """Per-type [tp, fp, fn] for one document under strict IOB2 chunking."""
    counts: Dict[str, List[int]] = {}

    def bucket(t: str) -> List[int]:
        return counts.setdefault(t, [0, 0, 0])

    gold_entities = _strict_entities(gold_labels)
    gold_remaining = {e: 0 for e in gold_entities}
    for e in gold_entities:
        gold_remaining[e] += 1

    for ent in _strict_entities(pred_labels):
        if gold_remaining.get(ent, 0) > 0:
            gold_remaining[ent] -= 1
            bucket(ent[0])[0] += 1
        else:
            bucket(ent[0])[1] += 1
    for ent, left in gold_remaining.items():
        if left > 0:
            bucket(ent[0])[2] += left
    return counts


def _relaxed_counts_char(
    pred_entities: List["SpanEntity"], gold_entities: List["SpanEntity"]
) -> Dict[str, List[int]]:
    """Per-type [tp, fp, fn] using character-level overlap.

    Mirrors ``src/llm_extraction/shared/evaluation.py:compute_relaxed_metrics``,
    including its rule that a predicted span takes the gold span it overlaps
    *most* rather than the first one it happens to touch.
    """
    counts: Dict[str, List[int]] = {}

    def bucket(t: str) -> List[int]:
        return counts.setdefault(t, [0, 0, 0])

    matched: set = set()
    for pred in pred_entities:
        best, best_overlap = None, 0
        for i, goldent in enumerate(gold_entities):
            if i in matched or not relaxed_span_match(pred, goldent):
                continue
            overlap = overlap_length(pred, goldent)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        if best is not None:
            matched.add(best)
            bucket(pred.type)[0] += 1
        else:
            bucket(pred.type)[1] += 1
    for i, goldent in enumerate(gold_entities):
        if i not in matched:
            bucket(goldent.type)[2] += 1
    return counts


def _to_array(
    per_doc: List[Dict[str, List[int]]]
) -> Tuple[np.ndarray, List[str]]:
    """Stack per-document count dicts into a (n_docs, n_canonical_types, 3) array."""
    idx = {t: i for i, t in enumerate(CANONICAL_TYPES)}
    arr = np.zeros((len(per_doc), len(CANONICAL_TYPES), 3), dtype=np.int64)
    dropped: Dict[str, int] = {}
    for i, tc in enumerate(per_doc):
        for t, (tp, fp, fn) in tc.items():
            j = idx.get(t)
            if j is None:
                dropped[t] = dropped.get(t, 0) + tp + fp + fn
                continue
            arr[i, j] = (tp, fp, fn)
    if dropped:
        logger.warning("Dropped out-of-schema types: %s", dropped)
    return arr, CANONICAL_TYPES


def load_generative_counts(
    pred_path: Path, gold: Dict[str, dict]
) -> Optional[dict]:
    """Per-document counts for one generative run, on the canonical axis.

    Scored exactly as ``evaluate_span_extraction`` scores it: exact match by
    strict IOB2 chunking over the shared tokenizer, relaxed match by character
    overlap on the aligned spans.

    Documents the model failed on are kept with zero predictions, which is how
    the reported figures score them: a failure contributes no true positives and
    every gold span in it becomes a false negative. Dropping them here would
    silently inflate the generative side of the comparison.
    """
    if not pred_path.exists():
        return None

    with open(pred_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    ids: List[str] = []
    exact_per_doc: List[Dict[str, List[int]]] = []
    relaxed_per_doc: List[Dict[str, List[int]]] = []
    missing_gold = 0
    for row in rows:
        gt = gold.get(row["id"])
        if gt is None:
            missing_gold += 1
            continue

        pred_entities = [
            SpanEntity(
                text=e.get("text", ""),
                type=e.get("type", e.get("label", "UNKNOWN")),
                start=e.get("start"),
                end=e.get("end"),
            )
            for e in (row.get("entities") or [])
            # Unaligned spans carry start=None and are discarded before scoring
            # (this is the span-discard rate reported in Appendix C).
            if e.get("start") is not None and e.get("end") is not None
        ]
        gold_entities = [
            SpanEntity(
                text=s.get("text", ""),
                type=s.get("label", s.get("type", "UNKNOWN")),
                start=s["start"],
                end=s["end"],
            )
            for s in gt.get("spans", [])
        ]

        _, pred_labels = spans_to_bio_labels(gt["text"], pred_entities)
        _, gold_labels = spans_to_bio_labels(gt["text"], gold_entities)

        ids.append(row["id"])
        exact_per_doc.append(_exact_counts_strict(pred_labels, gold_labels))
        relaxed_per_doc.append(_relaxed_counts_char(pred_entities, gold_entities))

    if missing_gold:
        logger.warning("%s: %d prediction(s) had no gold entry",
                       pred_path.name, missing_gold)

    exact, _ = _to_array(exact_per_doc)
    relaxed, _ = _to_array(relaxed_per_doc)
    return {
        "ids": ids,
        "exact": exact,
        "relaxed": relaxed,
        "types": CANONICAL_TYPES,
    }


# --------------------------------------------------------------------- #
# Alignment                                                             #
# --------------------------------------------------------------------- #

def align(ra: dict, rb: dict) -> Tuple[dict, dict]:
    """Restrict two runs to their common doc ids, in a single matched order."""
    if ra["ids"] == rb["ids"]:
        return ra, rb
    common = set(ra["ids"]) & set(rb["ids"])
    order_ids = [i for i in ra["ids"] if i in common]
    if len(order_ids) < max(len(ra["ids"]), len(rb["ids"])):
        logger.warning(
            "Pair shares %d of %d/%d documents; testing on the intersection.",
            len(order_ids), len(ra["ids"]), len(rb["ids"]),
        )
    def take(r):
        pos = {i: k for k, i in enumerate(r["ids"])}
        o = [pos[i] for i in order_ids]
        return {**r, "ids": order_ids, "exact": r["exact"][o], "relaxed": r["relaxed"][o]}
    return take(ra), take(rb)


# --------------------------------------------------------------------- #
# Reproduction check                                                    #
# --------------------------------------------------------------------- #

def _reported_f1(report: dict, block_keys: Tuple[str, ...]) -> Optional[float]:
    """Pull macro F1 out of an evaluation JSON, tolerating both layouts.

    ``evaluate_span_extraction`` emits ``entity_level.f1``; running its output
    through ``evaluate_spans.transform_to_deberta_format`` renames that to
    ``entity_level_metrics.entity_f1``. The files on disk are the transformed
    kind, but both are accepted so a re-run of either path still validates.
    """
    for key in block_keys:
        block = report.get(key)
        if not isinstance(block, dict):
            continue
        for field in ("entity_f1", "f1", "f1_score"):
            value = block.get(field)
            if value is not None:
                return float(value)
    return None


def check_generative(
    provider: str, strategy: str, run: dict, tol: float, pred_path: Path
) -> List[str]:
    """Compare re-derived macro F1 against the cached evaluation JSON.

    A metric that cannot be located is reported as a problem, not skipped: the
    whole point of the check is that the counts feeding the bootstrap are the
    counts behind the table, and a silently absent reference proves nothing.
    """
    report_path = pred_path.with_name(f"{strategy}_evaluation.json")
    if not report_path.exists():
        return [f"{provider}/{strategy}: no evaluation JSON at {report_path}"]
    report = json.loads(report_path.read_text())
    problems = []
    pairs = [
        ("exact", ("entity_level_metrics", "entity_level")),
        ("relaxed", ("relaxed_boundary_metrics",)),
    ]
    for key, block_keys in pairs:
        reported = _reported_f1(report, block_keys)
        if reported is None:
            problems.append(
                f"{provider}/{strategy} {key}: no macro F1 found under "
                f"{block_keys} in {report_path.name}"
            )
            continue
        got = macro_f1_from_counts(run[key].sum(axis=0)) * 100
        ref = reported * 100 if reported <= 1.0 else reported
        if abs(got - ref) > tol:
            problems.append(
                f"{provider}/{strategy} {key}: re-derived {got:.2f} vs reported {ref:.2f}"
            )
        else:
            logger.info("  check %s/%s %-8s re-derived %.2f vs reported %.2f  OK",
                        provider, strategy, key, got, ref)
    return problems


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def parse_generative(spec: str) -> Tuple[str, str]:
    provider, _, strategy = spec.partition(":")
    if not strategy:
        raise argparse.ArgumentTypeError(
            f"--generative expects provider:strategy, got {spec!r}")
    return provider, strategy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    ap.add_argument("--generative", nargs="+", type=parse_generative,
                    default=DEFAULT_GENERATIVE)
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="Max allowed gap, in F1 points, between re-derived and "
                         "reported macro F1 before the run is rejected")
    ap.add_argument("--no-check", action="store_true",
                    help="Skip the reproduction check (not recommended)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gold = load_gold(GOLD_TEST)
    logger.info("Gold: %d documents from %s", len(gold), GOLD_TEST)

    # Encoders, one run per seed.
    enc: Dict[str, Dict[str, dict]] = {}
    for model in args.encoders:
        enc[model] = {}
        for seed_id, _ in SEEDS:
            raw = load_counts(in_domain_pred(model, seed_id))
            if raw is None:
                logger.warning("%s seed %s: predictions missing", model, seed_id)
                continue
            enc[model][seed_id] = reindex(raw)
        if enc[model]:
            n = len(next(iter(enc[model].values()))["ids"])
            logger.info("%s: %d seed(s), %d documents", model, len(enc[model]), n)

    # Generative runs, one sample each.
    gen: Dict[str, dict] = {}
    sources: Dict[str, str] = {}
    problems: List[str] = []
    for provider, strategy in args.generative:
        path = resolve_generative(provider, strategy)
        if path is None:
            logger.warning("%s/%s: no predictions under %s",
                           provider, strategy,
                           " or ".join(str(r.relative_to(ROOT)) for r in GENERATIVE_ROOTS))
            continue
        run = load_generative_counts(path, gold)
        if run is None:
            continue
        name = f"{provider}_{strategy}"
        gen[name] = run
        sources[name] = str(path.relative_to(ROOT))
        logger.info("%s: %d documents  <- %s", name, len(run["ids"]), sources[name])
        if not args.no_check:
            problems += check_generative(provider, strategy, run, args.tolerance, path)

    if problems:
        for p in problems:
            logger.error(p)
        raise SystemExit(
            "Re-derived macro F1 does not reproduce the reported values. The "
            "counts feeding the bootstrap are not the counts behind the table, "
            "so the p-values would describe a different statistic. Fix before use."
        )

    results: Dict[str, dict] = {}
    for model, seeds in enc.items():
        if not seeds:
            continue
        for name, grun in gen.items():
            per_seed = {}
            a_e, b_e, a_r, b_r = [], [], [], []
            for seed_id, erun in seeds.items():
                ea, ga = align(erun, grun)
                per_seed[seed_id] = {
                    "exact": paired_bootstrap_test(
                        ea["exact"], ga["exact"], args.n_bootstrap, rng),
                    "relaxed": paired_bootstrap_test(
                        ea["relaxed"], ga["relaxed"], args.n_bootstrap, rng),
                }
                a_e.append(ea["exact"]); b_e.append(ga["exact"])
                a_r.append(ea["relaxed"]); b_r.append(ga["relaxed"])
            results[f"{model}_vs_{name}"] = {
                "per_seed": per_seed,
                "pooled_exact": (
                    paired_bootstrap_test_pooled(a_e, b_e, args.n_bootstrap, rng)
                    if len(a_e) >= 2 else None),
                "pooled_relaxed": (
                    paired_bootstrap_test_pooled(a_r, b_r, args.n_bootstrap, rng)
                    if len(a_r) >= 2 else None),
            }

    # Observed macro F1 per system, for the record.
    observed = {}
    for model, seeds in enc.items():
        for seed_id, r in seeds.items():
            observed[f"{model}_s{seed_id}"] = {
                "exact": macro_f1_from_counts(r["exact"].sum(axis=0)) * 100,
                "relaxed": macro_f1_from_counts(r["relaxed"].sum(axis=0)) * 100,
            }
    for name, r in gen.items():
        observed[name] = {
            "exact": macro_f1_from_counts(r["exact"].sum(axis=0)) * 100,
            "relaxed": macro_f1_from_counts(r["relaxed"].sum(axis=0)) * 100,
        }

    payload = {
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "alpha": args.alpha,
            "rng_seed": args.seed,
            "encoders": args.encoders,
            "generative": [f"{p}:{s}" for p, s in args.generative],
            "generative_sources": sources,
            "type_axis": CANONICAL_TYPES,
            "gold": str(GOLD_TEST.relative_to(ROOT)),
            "note": (
                "Encoder rows pool the three seeds by averaging per-seed deltas "
                "within each bootstrap iteration; the generative run is a single "
                "sample held fixed across them."
            ),
        },
        "observed_macro_f1": observed,
        "pairwise": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    # ── Report ───────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("Encoder vs generative — paired bootstrap, in-domain test set")
    print("=" * 78)
    print(f"{'comparison':<34} {'metric':<8} {'Δ F1':>8} {'95% CI':>18} {'p':>8}")
    print("-" * 78)
    for key, res in results.items():
        for metric, block in (("exact", res["pooled_exact"]),
                              ("relaxed", res["pooled_relaxed"])):
            if block is None:
                continue
            d = block["observed_delta_macro_f1"] * 100
            lo = block["ci_low"] * 100
            hi = block["ci_high"] * 100
            p = block["p_two_sided"]
            p_str = "<0.0005" if p == 0 else f"{p:.4f}"
            print(f"{key:<34} {metric:<8} {d:>+8.2f} "
                  f"{f'[{lo:+.2f}, {hi:+.2f}]':>18} {p_str:>8}")
    print("-" * 78)
    print(f"Δ is encoder − generative, macro F1 points. "
          f"{args.n_bootstrap} iterations, pooled over "
          f"{len(SEEDS)} encoder seeds.")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
