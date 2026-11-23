#!/usr/bin/env python3
"""One place that says what the generative standard benchmark actually contains.

WHY THIS EXISTS. There are three provider directories, two strategies each, and
several partial runs that look complete because the row count is right. A file
with 529 rows can be a finished run, a run where every row is a DNS error, or a
run half of whose failures are the institutional gateway rate-limiting us. The
row count does not distinguish them, and the difference decides whether a number
belongs in Table 1.

It also draws the one distinction that keeps getting lost: a document can fail
because the MODEL failed (runaway generation, malformed JSON) or because the
NETWORK failed (DNS, connection reset, rate limit). Only the first is a property
of the model. Reporting the second in the paper's failure-rate table publishes our infrastructure as a finding about Gemini or AMALIA.

Usage:
    python scripts/llm_extraction/consolidate_standard_benchmark.py
    python scripts/llm_extraction/consolidate_standard_benchmark.py --score
    python scripts/llm_extraction/consolidate_standard_benchmark.py --dir results/generative
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GOLD = ROOT / "data" / "citilink-votie" / "test.jsonl"
PROVIDERS = ["gemini", "gpt5.5", "amalia"]
STRATEGIES = ["zero_shot", "few_shot"]

DISPLAY = {"gemini": "Gemini 2.5 Pro", "gpt5.5": "GPT-5.5", "amalia": "AMALIA 8B"}

# $/1M tokens. AMALIA is self-hosted, so its
# cost is GPU-hours, not dollars.
PRICING = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
}

# The infra-vs-model split is defined once, in retry_policy, because the two
# places that apply it must agree: the extractors use it to decide what a retry
# may overwrite, and this script uses it to decide whether a config is complete.
# They were separate copies until one of them drifted — this script's bare "429"
# marker read the character offset in
#   "Unterminated string ... (char 20429)"
# as a rate-limit, and filed one of AMALIA's repetition-loop failures under our
# network, which is exactly the misreport retry_policy exists to prevent.
from src.llm_extraction.shared.retry_policy import classify_failure as classify  # noqa: E402


def summarise(path: Path, gold_ids: set[str]) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    ids = [r["id"] for r in rows]
    unique = set(ids)

    buckets = {"ok": [], "infra": [], "model": []}
    for r in rows:
        buckets[classify(r.get("error"))].append(r)

    diag = [r.get("diagnostics") or {} for r in rows]
    usage = [d.get("usage") or {} for d in diag]
    tin = sum(u.get("input_tokens", 0) for u in usage)
    tout = sum(u.get("billable_output_tokens", u.get("output_tokens", 0)) for u in usage)

    models = sorted({r.get("model") for r in rows if r.get("model")})
    price = PRICING.get(models[0]) if len(models) == 1 else None
    cost = (tin / 1e6) * price["input"] + (tout / 1e6) * price["output"] if price else None

    ok = buckets["ok"]
    g = lambda field: sum((r.get("diagnostics") or {}).get(field, 0) for r in ok)
    raw, before, aligned = g("n_raw_generated"), g("n_before_alignment"), g("n_aligned")
    dropped_align, dropped_type = g("n_dropped_alignment"), g("n_dropped_unknown_type")

    times = [r.get("processing_time") or 0.0 for r in rows]

    # "Complete" is deliberately strict. Full gold coverage is not enough: an
    # infra failure is a document we simply have no answer for, and a run whose
    # coverage depends on those is not finished, only stopped.
    missing = gold_ids - unique
    blockers = []
    if missing:
        blockers.append(f"{len(missing)} gold documents absent")
    if len(ids) != len(unique):
        blockers.append(f"{len(ids) - len(unique)} duplicate ids")
    if buckets["infra"]:
        blockers.append(f"{len(buckets['infra'])} infrastructure failures (retry these)")
    if len(models) > 1:
        blockers.append(f"mixed model ids: {models}")
    if tin == 0:
        blockers.append("no token usage captured — cost cannot be reported")

    return {
        "path": str(path.relative_to(ROOT)),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "rows": len(rows),
        "unique_ids": len(unique),
        "gold_covered": len(gold_ids & unique),
        "gold_missing": sorted(missing)[:5],
        "model_ids": models,
        "succeeded": len(ok),
        "failed_infra": len(buckets["infra"]),
        "failed_model": len(buckets["model"]),
        "model_failure_rate": len(buckets["model"]) / len(rows) if rows else None,
        "entities_retained": sum(len(r.get("entities") or []) for r in rows),
        "spans_raw_generated": raw,
        "spans_dropped_unknown_type": dropped_type,
        "spans_dropped_alignment": dropped_align,
        "span_discard_rate": dropped_align / before if before else None,
        "spans_aligned": aligned,
        "tokens_input": tin,
        "tokens_billable_output": tout,
        "cost_usd": round(cost, 2) if cost is not None else None,
        "latency_mean_s": round(statistics.mean(times), 1) if times else None,
        "latency_median_s": round(statistics.median(times), 1) if times else None,
        "wall_clock_h": round(sum(times) / 3600, 2),
        "complete": not blockers,
        "blockers": blockers,
    }


def score(path: Path, out_path: Path) -> dict | None:
    """Delegate to the canonical scorer rather than reimplementing it."""
    cmd = [sys.executable, str(ROOT / "scripts/llm_extraction/evaluate_spans.py"), str(path),
           "--ground-truth", str(GOLD), "--relaxed", "--output-file", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [WARN] scoring failed for {path.name}: {proc.stderr.strip().splitlines()[-1:] }")
        return None
    return json.loads(out_path.read_text(encoding="utf-8"))


def pct(x, digits=1):
    return "--" if x is None else f"{100 * x:.{digits}f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "generative"))
    ap.add_argument("--score", action="store_true",
                    help="also run evaluate_spans.py on every COMPLETE config")
    ap.add_argument("--out", default=None, help="JSON path (default <dir>/STANDARD_BENCHMARK.json)")
    args = ap.parse_args()

    root = Path(args.dir)
    gold_ids = {json.loads(l)["id"] for l in GOLD.open(encoding="utf-8")}

    configs = {}
    for provider in PROVIDERS:
        for strategy in STRATEGIES:
            path = root / provider / f"{strategy}.jsonl"
            key = f"{provider}/{strategy}"
            if not path.exists():
                configs[key] = {"path": str(path.relative_to(ROOT)), "complete": False,
                                "blockers": ["file does not exist — run not started"]}
                continue
            configs[key] = summarise(path, gold_ids)

    if args.score:
        for key, c in configs.items():
            if c.get("complete"):
                p = ROOT / c["path"]
                ev = score(p, p.with_name(p.stem + "_evaluation.json"))
                if ev:
                    c["scores"] = {
                        "exact_f1": ev["entity_level_metrics"]["entity_f1"],
                        "exact_precision": ev["entity_level_metrics"]["entity_precision"],
                        "exact_recall": ev["entity_level_metrics"]["entity_recall"],
                        "relaxed_f1": ev["relaxed_boundary_metrics"]["entity_f1"],
                        "evaluation_file": str(p.with_name(p.stem + "_evaluation.json").relative_to(ROOT)),
                    }

    n_done = sum(1 for c in configs.values() if c.get("complete"))
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(root.relative_to(ROOT)),
        "gold": str(GOLD.relative_to(ROOT)),
        "gold_documents": len(gold_ids),
        "configs_complete": n_done,
        "configs_total": len(configs),
        "ready_for_table1": n_done == len(configs),
        "failure_taxonomy": {
            "infra": "network/gateway failure (DNS, connection, rate limit, timeout). "
                     "OURS, not the model's. Retry; never report in the paper's failure rate.",
            "model": "the model returned something unusable (runaway generation, malformed "
                     "JSON). A genuine property of the model.",
        },
        "configs": configs,
    }

    out = Path(args.out) if args.out else root / "STANDARD_BENCHMARK.json"
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)

    # ---- human-readable twin ------------------------------------------------
    md = [f"# Generative standard benchmark — consolidated status",
          "",
          f"Generated {payload['generated']} by `scripts/llm_extraction/consolidate_standard_benchmark.py`.",
          f"Source: `{payload['source_dir']}`  ·  gold: `{payload['gold']}` ({len(gold_ids)} documents).",
          "",
          f"**{n_done} of {len(configs)} configurations complete.**"
          + ("  Table 1 can be built." if n_done == len(configs)
             else "  Table 1 is NOT ready — see blockers below."),
          "",
          "A run is complete only when it covers every gold document, has no duplicate ids,",
          "has zero *infrastructure* failures, uses a single model id, and captured token usage.",
          "Model failures do not block completion — they are a result, not an error.",
          "",
          "| Config | Docs | OK | Infra fail | Model fail | Entities | Discard | Cost | Status |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key, c in configs.items():
        if "rows" not in c:
            md.append(f"| `{key}` | — | — | — | — | — | — | — | **not started** |")
            continue
        cost = "self-hosted" if key.startswith("amalia") else (
            f"${c['cost_usd']:.2f}" if c["cost_usd"] is not None else "—")
        md.append(
            f"| `{key}` | {c['rows']} | {c['succeeded']} | {c['failed_infra']} | "
            f"{c['failed_model']} | {c['entities_retained']} | {pct(c['span_discard_rate'])} | "
            f"{cost} | {'**COMPLETE**' if c['complete'] else 'incomplete'} |")

    incomplete = {k: c for k, c in configs.items() if not c.get("complete")}
    if incomplete:
        md += ["", "## Blockers", ""]
        for key, c in incomplete.items():
            md.append(f"**`{key}`**")
            for b in c["blockers"]:
                md.append(f"- {b}")
            md.append("")

    done = {k: c for k, c in configs.items() if c.get("complete")}
    if done:
        md += ["## Complete runs — paper-facing numbers", "",
               "| Config | Model failure rate | Span discard | Bad-type spans | Latency s/doc | Wall clock |",
               "|---|---:|---:|---:|---:|---:|"]
        for key, c in done.items():
            bad_type = (c["spans_dropped_unknown_type"] / c["spans_raw_generated"]
                        if c["spans_raw_generated"] else None)
            md.append(f"| `{key}` | {pct(c['model_failure_rate'], 2)} | {pct(c['span_discard_rate'])} | "
                      f"{pct(bad_type)} | {c['latency_mean_s']} | {c['wall_clock_h']} h |")
        if any("scores" in c for c in done.values()):
            md += ["", "| Config | Exact P | Exact R | Exact F1 | Relaxed F1 |", "|---|---:|---:|---:|---:|"]
            for key, c in done.items():
                s = c.get("scores")
                if s:
                    md.append(f"| `{key}` | {pct(s['exact_precision'])} | {pct(s['exact_recall'])} | "
                              f"{pct(s['exact_f1'])} | {pct(s['relaxed_f1'])} |")

    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nWrote {out.relative_to(ROOT)} and {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
