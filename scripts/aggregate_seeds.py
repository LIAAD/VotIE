"""Aggregate macro/entity F1 across seeds for the in-domain standard benchmark
and for the LOMO experiments. Writes a JSON summary and prints a human-readable
table; optionally emits the Table 1 / Table 2 LaTeX so no cell is typed by hand.

File layout under results/:
  In-domain:  standard/<model>/seed_s<SEED>_evaluation.json
  LOMO:       lomo/<model>/<M>_seed_s<SEED>_evaluation.json

THE MACRO DENOMINATOR IS A CHOICE, AND IT MOVES EVERY LOMO CELL.
The eval JSONs come from seqeval's classification_report, which emits a row only
for types present in gold or prediction — so the number of rows varies by fold
(8 to 12 across M01-M06, 11 on the standard test split). Averaging over
"whatever rows exist" would silently change the denominator per cell. Two defensible
policies are implemented, and both are always computed so the difference is visible:

  support   average over types with support > 0 in that fold (8-12 types).
            This is the policy used in Tables 1 and 2. Cells within a
            column share a denominator, so per-municipality model rankings are
            sound, but the Mean column averages across different denominators.
  fixed11   average over the 11 schema types excluding Count-Against, scoring
            absent types 0.0. Every cell shares one denominator, so the Mean and
            the Table 1 vs Table 2 comparison are commensurable — at the cost of
            charging a model 0.0 for types its held-out municipality never uses.

On the standard test split the two coincide (exactly 11 types have support), so
they differ only for LOMO. Count-Against is excluded from both: the corpus holds
exactly ONE Count-Against span — "1 voto contra" in
Porto_cm_004_2021-12-06_17 — annotated twice at identical offsets. It falls in
M06 and never in the standard test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "seed_aggregate_summary.json"

# The 11 scored types. Mirrors src/llm_extraction/schemas.py:ENTITY_TYPES so the
# generative rows of Tables 1 and 2 share this denominator with the encoders.
ENTITY_TYPES = [
    "COUNT-BLANK", "COUNT-FAVOR", "COUNTING-MAJORITY", "COUNTING-UNANIMITY",
    "SUBJECT", "VOTER-ABSENT", "VOTER-ABSTENTION", "VOTER-AGAINST",
    "VOTER-FAVOR", "VOTING", "VOTING-METHOD",
]

MUNICIPALITIES = ["M01", "M02", "M03", "M04", "M05", "M06"]

# Models reported with 3 seeds; everything else is seed 42 only, matching how
# non-CRF rows are reported in Table 1.
MULTI_SEED_MODELS = ["bert_crf", "xlmr_crf", "deberta_crf"]
SINGLE_SEED_MODELS = ["bert_linear", "deberta_linear", "xlmr_linear",
                      "bilstm_fasttext", "crf"]
MODELS = MULTI_SEED_MODELS + SINGLE_SEED_MODELS

# Paper-facing names. `bilstm_fasttext` is the config's model.name (hence the
# results directory); the paper calls it BiLSTM-CRF.
DISPLAY_NAMES = {
    "bert_crf": "BERTimbau-CRF",
    "bert_linear": "BERTimbau-Lin.",
    "deberta_crf": "DeBERTa-CRF",
    "deberta_linear": "DeBERTa-Lin.",
    "xlmr_crf": "XLM-R-CRF",
    "xlmr_linear": "XLM-R-Lin.",
    "bilstm_fasttext": "BiLSTM-CRF",
    "crf": "CRF",
}

SEEDS = ["42", "13", "123"]

EXACT_BLOCK = "entity_level_metrics"
RELAXED_BLOCK = "relaxed_boundary_metrics"


def _rel(path: Path) -> str:
    """Repo-relative when possible; absolute paths (e.g. a scratch dir) pass through."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def seeds_for(model: str) -> list[str]:
    return SEEDS if model in MULTI_SEED_MODELS else ["42"]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def macro_f1(report: dict, block: str, denominator: str) -> tuple[float | None, int | None]:
    """Macro F1 over per-type scores, plus the denominator actually used.

    Returning the count alongside the value is what stops a caption from drifting
    away from the table: the caller can assert every cell used the same one.
    """
    per_type = (report.get(block) or {}).get("per_type_metrics") or {}
    if not per_type:
        return None, None

    if denominator == "fixed11":
        f1s = [per_type.get(t, {}).get("f1_score", 0.0) for t in ENTITY_TYPES]
    elif denominator == "support":
        # ENTITY_TYPES is the 11-type schema minus Count-Against, so filtering
        # through it applies the same exclusion fixed11 applies. Without this the
        # policies disagree on M06 only — see the module docstring.
        f1s = [v["f1_score"] for t, v in per_type.items()
               if t in ENTITY_TYPES and v.get("support", 0) > 0]
    else:
        raise ValueError(f"unknown denominator {denominator!r}")

    if not f1s:
        return None, 0
    return sum(f1s) / len(f1s), len(f1s)


def micro_f1(report: dict) -> float | None:
    return (report.get(EXACT_BLOCK) or {}).get("entity_f1")


def exact_pr(report: dict) -> tuple[float | None, float | None]:
    em = report.get(EXACT_BLOCK) or {}
    return em.get("entity_precision"), em.get("entity_recall")


def relaxed_pr(report: dict) -> tuple[float | None, float | None]:
    rm = report.get(RELAXED_BLOCK) or {}
    return rm.get("precision"), rm.get("recall")


def aggregate(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "range": None, "values": []}
    n = len(values)
    return {
        "n": n,
        "mean": mean(values),
        # Sample std (ddof=1). The seeds are a sample of the possible random
        # initialisations, not the population, so Bessel's correction is the
        # unbiased estimate of run-to-run variance. This matches the convention
        # error_classification.py uses for Table 3.
        "std": stdev(values) if n > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "values": values,
    }


def in_domain_paths(model: str, standard_dir: Path) -> dict[str, Path]:
    return {s: standard_dir / model / f"seed_s{s}_evaluation.json" for s in SEEDS}


def lomo_paths(model: str, mun: str, lomo_dir: Path) -> dict[str, Path]:
    return {s: lomo_dir / model / f"{mun}_seed_s{s}_evaluation.json" for s in SEEDS}


# --- Generative LOMO ------------------------------------------------------- #
# The generative runs live in their own tree with their own file naming, and
# their evaluation JSON has a different shape from the encoder one. Rather than
# duplicate the denominator policy for them, normalise into the encoder shape and
# let macro_f1 do exactly the same thing to both. That is what keeps the Gemini
# row in Table 2 commensurable with the encoder rows.
GENERATIVE_LOMO_DIR = ROOT / "results" / "generative" / "lomo"
GENERATIVE_MODELS = {"gemini": "Gemini 2.5 Pro", "gpt": "GPT-5.5", "amalia": "AMALIA 8B"}
GENERATIVE_STRATEGY = "few_shot"


def generative_lomo_path(provider: str, mun: str, root: Path) -> Path:
    return root / provider / f"{provider}_{GENERATIVE_STRATEGY}_{mun}_evaluation.json"


def normalise_generative(report: dict) -> dict:
    """Re-key a generative evaluation JSON into the encoder layout.

    Differences handled: per-type scores sit at the top level rather than inside
    the exact-match block, the block itself is ``entity_level`` not
    ``entity_level_metrics``, and the per-type F1 field is ``f1`` not
    ``f1_score``. Nothing is recomputed — only moved.
    """
    def per_type(block: dict) -> dict:
        return {
            name: {**m, "f1_score": m.get("f1_score", m.get("f1"))}
            for name, m in (block or {}).items()
            if name not in ("micro avg", "macro avg", "weighted avg")
        }

    entity = report.get("entity_level") or {}
    relaxed = report.get("relaxed_boundary_metrics") or {}
    return {
        EXACT_BLOCK: {
            "entity_precision": entity.get("precision"),
            "entity_recall": entity.get("recall"),
            "entity_f1": entity.get("f1"),
            "per_type_metrics": per_type(report.get("per_type_metrics")),
        },
        RELAXED_BLOCK: {
            "precision": relaxed.get("precision"),
            "recall": relaxed.get("recall"),
            "per_type_metrics": per_type(relaxed.get("per_type_metrics")),
        },
    }


def _cell(report: dict, denominator: str) -> dict:
    exact, n_exact = macro_f1(report, EXACT_BLOCK, denominator)
    relaxed, n_relaxed = macro_f1(report, RELAXED_BLOCK, denominator)
    eP, eR = exact_pr(report)
    rP, rR = relaxed_pr(report)
    return {
        "present": True,
        "macro_f1": exact,
        "relaxed_f1": relaxed,
        "n_types_exact": n_exact,
        "n_types_relaxed": n_relaxed,
        "micro_f1": micro_f1(report),
        "exact_precision": eP,
        "exact_recall": eR,
        "relaxed_precision": rP,
        "relaxed_recall": rR,
    }


def summarise_indomain(standard_dir: Path, denominator: str) -> dict:
    out = {}
    for model in MODELS:
        paths = in_domain_paths(model, standard_dir)
        per_seed, keys = {}, {
            "macro_f1": [], "relaxed_f1": [], "micro_f1": [],
            "exact_precision": [], "exact_recall": [],
            "relaxed_precision": [], "relaxed_recall": [],
        }
        denominators = set()
        for seed in seeds_for(model):
            path = paths[seed]
            report = load(path)
            if report is None:
                per_seed[seed] = {"path": _rel(path), "present": False}
                continue
            cell = _cell(report, denominator)
            cell["path"] = _rel(path)
            per_seed[seed] = cell
            for key in keys:
                if cell.get(key) is not None:
                    keys[key].append(cell[key])
            if cell["n_types_exact"]:
                denominators.add(cell["n_types_exact"])
        out[model] = {
            "display_name": DISPLAY_NAMES.get(model, model),
            "per_seed": per_seed,
            "seed_count": sum(1 for v in per_seed.values() if v.get("present")),
            "denominators_used": sorted(denominators),
            **{key: aggregate(vals) for key, vals in keys.items()},
        }
    return out


def summarise_lomo(lomo_dir: Path, denominator: str) -> dict:
    out = {}
    for model in MODELS:
        per_mun, exact_means, relaxed_means = {}, [], []
        denominators = set()
        for mun in MUNICIPALITIES:
            paths = lomo_paths(model, mun, lomo_dir)
            per_seed, exacts, relaxeds = {}, [], []
            for seed in seeds_for(model):
                report = load(paths[seed])
                if report is None:
                    per_seed[seed] = {"path": _rel(paths[seed]), "present": False}
                    continue
                cell = _cell(report, denominator)
                cell["path"] = _rel(paths[seed])
                per_seed[seed] = cell
                if cell["macro_f1"] is not None:
                    exacts.append(cell["macro_f1"])
                if cell["relaxed_f1"] is not None:
                    relaxeds.append(cell["relaxed_f1"])
                if cell["n_types_exact"]:
                    denominators.add(cell["n_types_exact"])
            per_mun[mun] = {
                "per_seed": per_seed,
                "seed_count": sum(1 for v in per_seed.values() if v.get("present")),
                "macro_f1": aggregate(exacts),
                "relaxed_f1": aggregate(relaxeds),
            }
            if exacts:
                exact_means.append(mean(exacts))
            if relaxeds:
                relaxed_means.append(mean(relaxeds))
        out[model] = {
            "display_name": DISPLAY_NAMES.get(model, model),
            "per_municipality": per_mun,
            "denominators_used": sorted(denominators),
            "summary": {
                "macro_f1": aggregate(exact_means),
                "relaxed_f1": aggregate(relaxed_means),
            },
        }

    # Generative rows. Single-sample by construction — these providers are not
    # reproducible even at temperature 0, so
    # a seed count of 1 here means "one run", not "one seed".
    for provider, display in GENERATIVE_MODELS.items():
        per_mun, exact_means, relaxed_means = {}, [], []
        denominators = set()
        found = False
        for mun in MUNICIPALITIES:
            path = generative_lomo_path(provider, mun, GENERATIVE_LOMO_DIR)
            raw = load(path)
            if raw is None:
                per_mun[mun] = {"per_seed": {"42": {"path": _rel(path), "present": False}},
                                "seed_count": 0,
                                "macro_f1": aggregate([]), "relaxed_f1": aggregate([])}
                continue
            found = True
            cell = _cell(normalise_generative(raw), denominator)
            cell["path"] = _rel(path)
            per_mun[mun] = {
                "per_seed": {"42": cell},
                "seed_count": 1,
                "macro_f1": aggregate([cell["macro_f1"]]),
                "relaxed_f1": aggregate([cell["relaxed_f1"]]),
            }
            if cell["macro_f1"] is not None:
                exact_means.append(cell["macro_f1"])
            if cell["relaxed_f1"] is not None:
                relaxed_means.append(cell["relaxed_f1"])
            if cell["n_types_exact"]:
                denominators.add(cell["n_types_exact"])
        if not found:
            continue
        out[provider] = {
            "display_name": display,
            "per_municipality": per_mun,
            "denominators_used": sorted(denominators),
            "summary": {
                "macro_f1": aggregate(exact_means),
                "relaxed_f1": aggregate(relaxed_means),
            },
        }
    return out


def fmt_pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def print_indomain(summary: dict, denominator: str) -> None:
    print(f"\n=== In-domain (standard benchmark) — denominator: {denominator} ===\n")
    print(f"  {'model':<16} {'seeds':>5} {'den':>4} "
          f"{'eP':>6} {'eR':>6} {'eF1':>6} {'±':>5}  {'rP':>6} {'rR':>6} {'rF1':>6} {'±':>5}  src")
    for model, d in summary.items():
        if d["seed_count"] == 0:
            print(f"  {d['display_name']:<16} {'--':>5}  (no results found)")
            continue
        den = ",".join(str(n) for n in d["denominators_used"]) or "?"
        sources = {v.get("source") for v in d["per_seed"].values() if v.get("present")}
        print(f"  {d['display_name']:<16} {d['seed_count']:>5} {den:>4} "
              f"{fmt_pct(d['exact_precision']['mean']):>6} {fmt_pct(d['exact_recall']['mean']):>6} "
              f"{fmt_pct(d['macro_f1']['mean']):>6} {fmt_pct(d['macro_f1']['std']):>5}  "
              f"{fmt_pct(d['relaxed_precision']['mean']):>6} {fmt_pct(d['relaxed_recall']['mean']):>6} "
              f"{fmt_pct(d['relaxed_f1']['mean']):>6} {fmt_pct(d['relaxed_f1']['std']):>5}  "
              f"{'/'.join(sorted(s for s in sources if s))}")


def incomplete_folds(entry: dict, key: str) -> list[str]:
    """Municipalities with no value for this metric.

    A row missing a fold still gets a Mean, and that Mean is over fewer
    municipalities than the rows above it — a number that looks directly
    comparable and is not. Callers mark it rather than printing it bare.
    """
    return [m for m in MUNICIPALITIES
            if entry["per_municipality"][m][key]["mean"] is None]


def print_lomo(summary: dict, denominator: str) -> None:
    warnings: list[str] = []
    for label, key in [("Exact Match", "macro_f1"), ("Relaxed Match", "relaxed_f1")]:
        print(f"\n=== LOMO — {label} F1 — denominator: {denominator} ===\n")
        print(f"{'model':<16} " + " ".join(f"{m:>6}" for m in MUNICIPALITIES) +
              f" {'Mean':>7} {'±std':>6}  seeds/cell")
        for model, d in summary.items():
            cells = [fmt_pct(d["per_municipality"][m][key]["mean"]) for m in MUNICIPALITIES]
            counts = "".join(str(d["per_municipality"][m]["seed_count"]) for m in MUNICIPALITIES)
            s = d["summary"][key]
            missing = incomplete_folds(d, key)
            flag = "  <-- PARTIAL" if missing else ""
            if missing:
                warnings.append(f"{d['display_name']} ({label}): mean over "
                                f"{len(MUNICIPALITIES) - len(missing)}/{len(MUNICIPALITIES)} folds, "
                                f"missing {', '.join(missing)}")
            print(f"{d['display_name']:<16} " + " ".join(f"{c:>6}" for c in cells) +
                  f" {fmt_pct(s['mean']):>7} {fmt_pct(s['std']):>6}  {counts}{flag}")

    if warnings:
        print("\n  WARNING: rows marked PARTIAL average over fewer municipalities than the")
        print("  others, so their Mean is NOT comparable with the complete rows:")
        for w in warnings:
            print(f"    - {w}")
        print("  Finish the missing folds, or drop the Mean for those rows in the table.")

    dens = sorted({n for d in summary.values() for n in d["denominators_used"]})
    if len(dens) > 1:
        # Spell the per-fold denominators out rather than naming folds in prose.
        by_muni = {}
        for d in summary.values():
            for m in MUNICIPALITIES:
                for seed in d["per_municipality"][m]["per_seed"].values():
                    n = seed.get("n_types_exact")
                    if n:
                        by_muni.setdefault(m, set()).add(n)
        per_fold = ", ".join(
            f"{m}:{'/'.join(str(x) for x in sorted(by_muni[m]))}"
            for m in MUNICIPALITIES if m in by_muni
        )
        print(f"\n  NOTE: cells used differing denominators {dens} ({per_fold})."
              f"\n  Under 'support' this is expected — a fold is scored only on the types its"
              f"\n  municipality actually uses — but it means the Mean column averages across"
              f"\n  denominators. Use --denominator fixed11 to remove it.")
    elif dens:
        print(f"\n  Every cell used a denominator of {dens[0]} types.")


def emit_latex(lomo: dict, denominator: str, path: Path) -> None:
    """Write the Table 2 body so no cell is transcribed by hand.

    The header comments carry the two facts a caption has to state: the per-cell
    seed count and the per-fold denominator, kept next to the numbers they describe.
    """
    present = [m for m, d in lomo.items()
               if not all(d["per_municipality"][mu]["seed_count"] == 0 for mu in MUNICIPALITIES)]

    lines = [
        f"% Generated by scripts/aggregate_seeds.py --denominator {denominator}",
        "% Do not edit by hand; re-run the script instead.",
        "%",
        "% Per-cell seed counts (M01..M06) — the caption must match these:",
    ]
    partial = []
    for model in present:
        d = lomo[model]
        counts = "".join(str(d["per_municipality"][mu]["seed_count"]) for mu in MUNICIPALITIES)
        missing = incomplete_folds(d, "macro_f1")
        note = f"   <-- PARTIAL: no {', '.join(missing)}" if missing else ""
        if missing:
            partial.append((d["display_name"], missing))
        lines.append(f"%   {d['display_name']:<16} {counts}{note}")

    # A row missing folds still produces a Mean, over fewer municipalities than
    # the rows above it. Left unmarked that is a table that quietly compares a
    # 4-fold average against a 6-fold one.
    if partial:
        lines.append("%")
        lines.append("% !! INCOMPLETE ROWS — the Mean below is NOT comparable across these rows:")
        for name, missing in partial:
            lines.append(f"%      {name}: mean over {len(MUNICIPALITIES) - len(missing)}"
                         f"/{len(MUNICIPALITIES)} folds (missing {', '.join(missing)})")
        lines.append("%    Finish those folds before using this table, or drop their Mean cell.")

    dens = sorted({n for m in present for n in lomo[m]["denominators_used"]})
    lines.append("%")
    if len(dens) > 1:
        per_fold = []
        for mu in MUNICIPALITIES:
            ns = {lomo[m]["per_municipality"][mu]["per_seed"][s].get("n_types_exact")
                  for m in present for s in SEEDS
                  if lomo[m]["per_municipality"][mu]["per_seed"].get(s, {}).get("present")}
            ns = sorted(n for n in ns if n)
            per_fold.append(f"{mu}={','.join(str(n) for n in ns)}")
        lines.append(f"% Denominator VARIES by fold: {'  '.join(per_fold)}")
        lines.append("% The Mean column therefore averages across denominators — state this.")
    else:
        lines.append(f"% Every cell used a denominator of {dens[0] if dens else '?'} types.")

    for label, key in [("Exact Match", "macro_f1"), ("Relaxed Match", "relaxed_f1")]:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{8}}{{c}}{{\textbf{{{label}}}}} \\")
        lines.append(r"\midrule")
        # Best value per column, so the bolding is derived rather than eyeballed.
        columns = MUNICIPALITIES + ["__mean__"]
        best = {}
        for col in columns:
            vals = [lomo[m]["summary"][key]["mean"] if col == "__mean__"
                    else lomo[m]["per_municipality"][col][key]["mean"] for m in present]
            vals = [v for v in vals if v is not None]
            best[col] = max(vals) if vals else None

        def render(value: float | None, col: str) -> str:
            if value is None:
                return "--"
            text = f"{100 * value:.1f}"
            return rf"\textbf{{{text}}}" if best[col] is not None and value == best[col] else text

        for model in present:
            d = lomo[model]
            cells = [render(d["per_municipality"][mu][key]["mean"], mu) for mu in MUNICIPALITIES]
            overall = render(d["summary"][key]["mean"], "__mean__")
            lines.append(f"{d['display_name']:<16} & {' & '.join(cells)} & {overall} \\\\")

    path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {_rel(path)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--denominator", choices=["fixed11", "support"], default="support",
                    help="Macro-average denominator policy (default: support, the policy "
                         "used in the paper). Both are always written to the "
                         "JSON; this selects what is printed and emitted as LaTeX.")
    ap.add_argument("--lomo-dir", type=Path, default=RESULTS / "lomo",
                    help="LOMO results root (default: results/lomo)")
    ap.add_argument("--standard-dir", type=Path, default=RESULTS / "standard",
                    help="In-domain results root (default: results/standard)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--latex", type=Path, default=None,
                    help="Also write the Table 2 body to this path")
    args = ap.parse_args()

    for d in (args.lomo_dir, args.standard_dir):
        if not d.exists():
            raise SystemExit(f"Results root not found: {d}")
    standard_dir = args.standard_dir

    # Compute under both policies so the JSON records the difference, and the
    # choice made for the paper can be checked against the alternative later.
    summary = {"denominator_reported": args.denominator, "by_denominator": {}}
    for denominator in ("fixed11", "support"):
        summary["by_denominator"][denominator] = {
            "in_domain": summarise_indomain(standard_dir, denominator),
            "lomo": summarise_lomo(args.lomo_dir, denominator),
        }
    summary["notes"] = {
        "entity_types": ENTITY_TYPES,
        "excluded": "COUNT-AGAINST (ONE gold span corpus-wide — '1 voto contra' in "
                    "Porto_cm_004_2021-12-06_17, annotated twice at identical offsets; "
                    "it falls in M06 and never in the standard test split)",
        "denominators": {
            "fixed11": "average over the 11 types above, absent types scored 0.0",
            "support": "average over types with support > 0 in that fold (8-11 for LOMO)",
        },
        "sources": {"lomo": _rel(args.lomo_dir), "in_domain": _rel(standard_dir)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {_rel(args.out)}")

    chosen = summary["by_denominator"][args.denominator]
    print_indomain(chosen["in_domain"], args.denominator)
    print_lomo(chosen["lomo"], args.denominator)

    other = "support" if args.denominator == "fixed11" else "fixed11"
    print(f"\n  (The same run under --denominator {other} is in {args.out.name} "
          f"under by_denominator.{other}.)")

    if args.latex:
        emit_latex(chosen["lomo"], args.denominator, args.latex)


if __name__ == "__main__":
    main()
