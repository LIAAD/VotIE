#!/usr/bin/env python3
"""
Automated decomposition of VotIE boundary errors.

Answers the question §6 of the paper leaves open: what is the 32-point
exact-match gap between encoders and LLMs actually made of?

For every INC_BOUNDARY error this reports

  * direction  -- truncated (pred inside gold), over-extended (gold inside
                  pred), or shifted (overlap but neither contains the other);
  * edge       -- which side moved, and by how many characters and words;
  * delta text -- the exact string added or dropped at each edge, which is what
                  makes the manual cause taxonomy checkable rather than
                  impressionistic;
  * triviality -- whether gold and prediction are the same string once leading
                  and trailing punctuation is removed, and whether the gold
                  boundary falls *inside* an orthographic word (the Portuguese
                  contraction case, which no subword model can reproduce).

It also reports, per model, how much of the exact-match error mass survives a
relaxed (type-agreement + any-overlap) criterion, the error rate on multi-vote
versus single-vote segments normalised by gold-span count, and how often one
predicted span swallows two gold spans -- an artifact of greedy one-to-one
matching that inflates MIS and BND together.

Usage:
    python scripts/boundary_error_analysis.py
    python scripts/boundary_error_analysis.py \
        --predictions 'Gemini=results/generative/gemini/few_shot.jsonl'
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from error_classification import (  # noqa: E402
    DEFAULT_MODELS,
    Entity,
    ErrorClassifier,
    resolve_spec,
)
from src.llm_extraction.shared.data_utils import load_jsonl  # noqa: E402


# Leading/trailing runs of non-word characters. \w is Unicode-aware for str
# patterns, so accented Portuguese letters and digits are preserved.
_EDGE_PUNCT = re.compile(r'^[\W_]+|[\W_]+$', re.UNICODE)
_WORD_CHAR = re.compile(r'\w', re.UNICODE)


def strip_edge_punct(s: str) -> str:
    return _EDGE_PUNCT.sub('', s.strip())


def is_intraword(text: str, pos: int) -> bool:
    """True if `pos` splits an orthographic word.

    This is the mechanism behind gold spans like "o BE" inside the contraction
    "do BE": the annotation boundary sits between two word characters, so a
    subword tokenizer can never place a span edge there.
    """
    if pos <= 0 or pos >= len(text):
        return False
    return bool(_WORD_CHAR.match(text[pos - 1]) and _WORD_CHAR.match(text[pos]))


def n_words(s: str) -> int:
    return len(s.split())


# ---------------------------------------------------------------------------
# Cause families.
#
# The families are the ones that recurred in a manually inspected sample of 64
# errors, restated here as a test on the *delta string* -- the text a prediction
# adds or drops at the edge -- so that every boundary error can be assigned one.
# The sample supplies the categories; these rules supply the counts.
#
# Precedence is the order in `classify_cause`, from the most specific surface
# test to the most general syntactic one. `other` is reported rather than
# absorbed, so the coverage of the scheme is visible.
# ---------------------------------------------------------------------------

# Closed-class determiners and preposition+article contractions. `d`, `pel`,
# `n` are the stranded prefixes left when a gold span starts inside a
# contraction ("d|os eleitos"), which is why they appear as whole deltas.
_DET = frozenset("""
o a os as um uma uns umas do da dos das no na nos nas ao aos à às
pelo pela pelos pelas d pel n de dum duma este esta estes estas esse essa
presente referido referida referidos referidas respetivo respetiva
respetivos respetivas seu sua seus suas
""".split())
_PREP = frozenset("""
de do da dos das para com em no na nos nas por pelo pela sobre a ao à
entre sem após durante mediante perante junto face
""".split())
_SUBORD = frozenset("""
que qual quais cujo cuja cujos cujas onde tendo sendo referente referentes
relativo relativa relativos relativas destinado destinada destinados
destinadas conforme constante constantes apresentado apresentada elaborado
elaborada requerido solicitado submeter denominado situado sito
""".split())
_COORD = frozenset('e ou bem assim'.split())

# A decision verb or a counting-and-stance clause pulled into the span.
_STANCE = re.compile(
    r'\b(unanimidade|maioria|absten|voto[s]?\s+contra|contra\s+d[oa]s?|'
    r'aprovad|aprovou|deliberou|deliberad|ratific|votou|votaram|favor)',
    re.IGNORECASE)
# Agenda-item material: a numbered marker, or the nominalised heading verb.
_HEADING = re.compile(
    r'^\s*\d+[\.\)º]|\b(aprovação|ratificação|proposta|reconhecimento|'
    r'informação|assunto|ponto)\b', re.IGNORECASE)

CAUSE_FAMILIES = (
    'punctuation only',
    'gold edge inside a word',
    'article or contraction',
    'agenda-heading material',
    'stance or outcome clause',
    'coordination',
    'relative or participial continuation',
    'prepositional continuation',
    'other',
)

# Column-width-friendly forms for the appendix table.
CAUSE_LABEL = {
    'punctuation only': 'Punctuation only',
    'gold edge inside a word': 'Gold edge inside a word',
    'article or contraction': 'Article or contraction',
    'agenda-heading material': 'Agenda heading',
    'stance or outcome clause': 'Stance or outcome clause',
    'coordination': 'Coordination',
    'relative or participial continuation': 'Relative / participial',
    'prepositional continuation': 'Prepositional',
    'other': '\\textit{other}',
}
SHORT_NAME = {
    'BERTimbau-CRF': 'BERTimbau',
    'DeBERTa-CRF': 'DeBERTa',
    'XLM-R-CRF': 'XLM-R',
    'Gemini-Few-Shot': 'Gemini',
    'GPT-5.5-Few-Shot': 'GPT-5.5',
    'AMALIA-Few-Shot': 'AMALIA',
}


def _tokens(s: str) -> List[str]:
    """Word tokens of a delta string, lowercased, punctuation dropped."""
    return [t.lower() for t in re.findall(r'[^\W\d_]+', s, re.UNICODE)]


def _is_infinitive(tok: str) -> bool:
    return tok.endswith(('ar', 'er', 'ir')) and len(tok) > 3


def classify_cause(d: Dict) -> str:
    """Assign one cause family to a decomposed boundary error.

    Precedence runs from the most specific surface test to the most general
    syntactic one. Note that `article or contraction` requires the delta to *be*
    a closed-class item, not merely to start with one: Portuguese articles and
    prepositions overlap heavily (`a`, `de`, `da`, `ao`), so a head-token test
    would book every long prepositional phrase as an article.
    """
    if d['punct_only']:
        return 'punctuation only'
    if ((d['left_delta_chars'] and d['gold_start_intraword'])
            or (d['right_delta_chars'] and d['gold_end_intraword'])):
        return 'gold edge inside a word'

    deltas = [x for x in (d['delta_text_left'], d['delta_text_right']) if x.strip()]
    if _HEADING.search(d['delta_text_left'] or ''):
        return 'agenda-heading material'
    if _STANCE.search(' '.join(deltas)):
        return 'stance or outcome clause'

    token_lists = [_tokens(x) for x in deltas]
    if deltas and all(len(t) == 1 and t[0] in _DET for t in token_lists):
        return 'article or contraction'

    if any(t and t[0] in _COORD for t in token_lists):
        return 'coordination'
    # "a submeter à Assembleia", "para elaboração": a preposition followed by an
    # infinitive opens a clause, not a prepositional modifier.
    if any(t and t[0] in _SUBORD or (len(t) > 1 and t[0] in _PREP and _is_infinitive(t[1]))
           for t in token_lists):
        return 'relative or participial continuation'
    if any(t and t[0] in _PREP for t in token_lists):
        return 'prepositional continuation'
    return 'other'


def describe_boundary(error, text: str) -> Dict:
    """Decompose a single INC_BOUNDARY error case."""
    g, p = error.gold_entity, error.pred_entity
    left = p.start - g.start      # < 0 -> prediction starts earlier (adds text)
    right = p.end - g.end         # > 0 -> prediction ends later (adds text)

    if p.start >= g.start and p.end <= g.end:
        direction = 'truncated'
    elif p.start <= g.start and p.end >= g.end:
        direction = 'over-extended'
    else:
        direction = 'shifted'

    if left != 0 and right != 0:
        edge = 'both'
    elif left != 0:
        edge = 'left-only'
    else:
        edge = 'right-only'

    delta_left = text[min(p.start, g.start):max(p.start, g.start)]
    delta_right = text[min(p.end, g.end):max(p.end, g.end)]

    out = {
        'doc_id': error.doc_id,
        'entity_type': error.entity_type,
        'direction': direction,
        'edge': edge,
        'left_delta_chars': left,
        'right_delta_chars': right,
        'displacement_chars': abs(left) + abs(right),
        'displacement_words': n_words(delta_left) + n_words(delta_right),
        'delta_text_left': delta_left,
        'delta_text_right': delta_right,
        'punct_only': strip_edge_punct(g.text) == strip_edge_punct(p.text),
        'gold_start_intraword': is_intraword(text, g.start),
        'gold_end_intraword': is_intraword(text, g.end),
        'gold_text': g.text,
        'pred_text': p.text,
        'gold_start': g.start,
        'gold_end': g.end,
        'pred_start': p.start,
        'pred_end': p.end,
        'gold_len_words': n_words(g.text),
    }
    out['cause_family'] = classify_cause(out)
    return out


def match_counts(gold: List[Entity], pred: List[Entity], mode: str):
    """Count unmatched gold / unmatched predictions under one criterion.

    exact   -- identical type and identical character boundaries
    relaxed -- identical type and any character overlap

    Greedy by descending overlap, which keeps the result independent of span
    order (unlike the first-overlap loop in scripts/evaluate.py:338-349).
    """
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if g.type != p.type:
                continue
            if mode == 'exact':
                if g.start == p.start and g.end == p.end:
                    pairs.append((1, gi, pi))
            else:
                ov = min(g.end, p.end) - max(g.start, p.start)
                if ov > 0:
                    pairs.append((ov, gi, pi))

    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_g, used_p = set(), set()
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)

    return len(gold) - len(used_g), len(pred) - len(used_p)


def count_merges(gold: List[Entity], pred: List[Entity]) -> int:
    """Predictions that cover at least half of two or more gold spans.

    One such prediction makes the greedy classifier emit a boundary error for
    the gold span it matches plus a spurious-miss for the one it does not, so a
    single underlying merge shows up as two errors.
    """
    merges = 0
    for p in pred:
        covered = 0
        for g in gold:
            ov = min(g.end, p.end) - max(g.start, p.start)
            if ov > 0 and ov >= 0.5 * (g.end - g.start):
                covered += 1
        if covered >= 2:
            merges += 1
    return merges


def summarise(rows: List[Dict], errors, gold, pred, texts, multi_vote) -> Dict:
    """Aggregate one model's decomposition into reportable numbers."""
    n = len(rows)
    out: Dict = {'n_boundary_errors': n}
    if n == 0:
        return out

    def share(pred_fn):
        k = sum(1 for r in rows if pred_fn(r))
        return {'n': k, 'pct': round(100 * k / n, 1)}

    out['direction'] = {
        d: share(lambda r, d=d: r['direction'] == d)
        for d in ('truncated', 'over-extended', 'shifted')
    }
    out['edge'] = {
        e: share(lambda r, e=e: r['edge'] == e)
        for e in ('left-only', 'right-only', 'both')
    }
    out['entity_types'] = dict(Counter(r['entity_type'] for r in rows).most_common())
    out['subject_share'] = share(lambda r: r['entity_type'] == 'SUBJECT')

    out['displacement'] = {
        'median_chars': median(r['displacement_chars'] for r in rows),
        'median_words': median(r['displacement_words'] for r in rows),
        'le_3_chars': share(lambda r: r['displacement_chars'] <= 3),
    }
    out['punct_only'] = share(lambda r: r['punct_only'])
    out['gold_boundary_intraword'] = share(
        lambda r: r['gold_start_intraword'] or r['gold_end_intraword']
    )

    # Cause families over every boundary error, and over the SUBJECT subset,
    # which is where three quarters of the error mass sits.
    fam_all = Counter(r['cause_family'] for r in rows)
    subj_rows = [r for r in rows if r['entity_type'] == 'SUBJECT']
    fam_subj = Counter(r['cause_family'] for r in subj_rows)
    out['cause_families'] = {
        f: {'n': fam_all[f], 'pct': round(100 * fam_all[f] / n, 1)}
        for f in CAUSE_FAMILIES
    }
    out['cause_families_subject'] = {
        'n_subject_boundary_errors': len(subj_rows),
        **{f: {'n': fam_subj[f],
               'pct': round(100 * fam_subj[f] / len(subj_rows), 1) if subj_rows else 0.0}
           for f in CAUSE_FAMILIES},
    }

    # The most frequent strings the model adds at each edge: the raw material
    # for the manual cause codebook.
    for side in ('left', 'right'):
        added = Counter()
        dropped = Counter()
        for r in rows:
            delta = r[f'delta_text_{side}'].strip()
            if not delta:
                continue
            signed = r[f'{side}_delta_chars']
            gained = signed < 0 if side == 'left' else signed > 0
            (added if gained else dropped)[delta] += 1
        out[f'{side}_edge_added'] = added.most_common(15)
        out[f'{side}_edge_dropped'] = dropped.most_common(15)

    # How much of the exact-match error mass a relaxed criterion forgives.
    ex_fn = ex_fp = rx_fn = rx_fp = merges = 0
    for doc_id in sorted(set(gold) | set(pred)):
        g, p = gold.get(doc_id, []), pred.get(doc_id, [])
        a, b = match_counts(g, p, 'exact')
        c, d = match_counts(g, p, 'relaxed')
        ex_fn += a
        ex_fp += b
        rx_fn += c
        rx_fp += d
        merges += count_merges(g, p)

    exact_total, relaxed_total = ex_fn + ex_fp, rx_fn + rx_fp
    out['match_criterion'] = {
        'exact': {'missed': ex_fn, 'spurious': ex_fp, 'total': exact_total},
        'relaxed': {'missed': rx_fn, 'spurious': rx_fp, 'total': relaxed_total},
        'forgiven_by_relaxed': exact_total - relaxed_total,
        'forgiven_pct': round(100 * (exact_total - relaxed_total) / exact_total, 1)
        if exact_total else 0.0,
    }
    out['merged_predictions'] = merges

    # A "missing" span that comes with a spurious span of the same type in the
    # same segment is not a recall failure but a substitution: the model found
    # the role and filled it from a different part of the text. Table 3 cannot
    # show this, because it books the two halves in different columns.
    subst = {}
    for etype in ('SUBJECT', None):
        mis = [e for e in errors if e.category == 'MIS'
               and (etype is None or e.entity_type == etype)]
        spu = [e for e in errors if e.category == 'SPU'
               and (etype is None or e.entity_type == etype)]
        by_doc = defaultdict(list)
        for e in spu:
            by_doc[(e.doc_id, e.entity_type)].append(e.pred_entity.start)
        paired = [e for e in mis if by_doc.get((e.doc_id, e.entity_type))]
        earlier = sum(
            1 for e in paired
            if e.gold_entity.start < min(by_doc[(e.doc_id, e.entity_type)])
        )
        # Co-occurrence alone cannot distinguish substitution from coincidence:
        # every model predicts a Subject in ~98% of segments, so a missed gold
        # Subject almost always has *some* predicted Subject beside it. The
        # claim that a miss is not a recall failure needs the direct statistic --
        # did the model predict this role in this segment at all?
        no_pred = sum(
            1 for e in mis
            if not any(x.type == e.entity_type for x in pred.get(e.doc_id, []))
        )
        subst[etype or 'ALL'] = {
            'missing': len(mis),
            'spurious': len(spu),
            'missing_with_same_type_substitute': len(paired),
            'pct_of_missing': round(100 * len(paired) / len(mis), 1) if mis else 0.0,
            'gold_earlier_than_substitute': earlier,
            'detection_failure': no_pred,
            'detection_failure_pct': round(100 * no_pred / len(mis), 1) if mis else 0.0,
        }
    out['substitution'] = subst

    # Decompose the spurious mass. A spurious span is a *substitution* when the
    # same role in the same segment is also recorded missing (the model named the
    # right thing from the wrong place), a *duplicate* when the role was filled
    # correctly and the model emitted a second span for it, and otherwise an
    # invention. Only the last is a model asserting something not in the text.
    roles_in_gold = {(d, g.type) for d in gold for g in gold[d]}
    missed_roles = {(e.doc_id, e.entity_type) for e in errors if e.category == 'MIS'}
    spu_all = [e for e in errors if e.category == 'SPU']
    dup = [e for e in spu_all
           if (e.doc_id, e.entity_type) in roles_in_gold
           and (e.doc_id, e.entity_type) not in missed_roles]
    sub = [e for e in spu_all if (e.doc_id, e.entity_type) in missed_roles]
    n = len(spu_all)
    out['spurious_breakdown'] = {
        'spurious': n,
        'substitution': len(sub),
        'duplicate_of_filled_role': len(dup),
        'invention': n - len(sub) - len(dup),
        'pct_not_invention': round(100 * (len(sub) + len(dup)) / n, 1) if n else 0.0,
        'duplicate_by_type': dict(Counter(e.entity_type for e in dup).most_common(5)),
    }

    # Multi-vote segments are 4.7% of the test set but carry more gold spans, so
    # the comparison only means anything per gold span.
    buckets = {'multi_vote': [0, 0], 'single_vote': [0, 0]}
    for doc_id, spans in gold.items():
        key = 'multi_vote' if doc_id in multi_vote else 'single_vote'
        buckets[key][1] += len(spans)
    for e in errors:
        key = 'multi_vote' if e.doc_id in multi_vote else 'single_vote'
        buckets[key][0] += 1
    out['multi_vote'] = {
        k: {
            'errors': v[0],
            'gold_spans': v[1],
            'errors_per_gold_span': round(v[0] / v[1], 3) if v[1] else None,
        }
        for k, v in buckets.items()
    }
    return out


def render_markdown(results: Dict) -> str:
    lines = ["# Boundary error decomposition\n",
             "Generated by `scripts/boundary_error_analysis.py`.\n"]

    lines.append("\n## Direction and scale\n")
    lines.append("| Model | BND | Subject | Truncated | Over-ext. | Median disp. "
                 "| <=3 chars | Punct-only | Gold edge mid-word |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        s = r['summary']
        if not s.get('n_boundary_errors'):
            continue
        lines.append(
            f"| {name} | {s['n_boundary_errors']} | {s['subject_share']['pct']}% "
            f"| {s['direction']['truncated']['pct']}% "
            f"| {s['direction']['over-extended']['pct']}% "
            f"| {s['displacement']['median_chars']:.0f} ch "
            f"| {s['displacement']['le_3_chars']['pct']}% "
            f"| {s['punct_only']['pct']}% "
            f"| {s['gold_boundary_intraword']['pct']}% |"
        )

    models = [n for n, r in results.items() if r['summary'].get('n_boundary_errors')]
    for title, key in (("all boundary errors", 'cause_families'),
                       ("SUBJECT boundary errors", 'cause_families_subject')):
        lines.append(f"\n## Cause families, {title}\n")
        lines.append(
            "> Families come from the 64-error manual sample; the counts are the\n"
            "> mechanical rules in `classify_cause` applied to every boundary error.\n"
        )
        lines.append("| Cause family | " + " | ".join(models) + " |")
        lines.append("|---" * (len(models) + 1) + "|")
        header = "| n | " + " | ".join(
            str(results[n]['summary'][key].get('n_subject_boundary_errors',
                                               results[n]['summary']['n_boundary_errors']))
            for n in models) + " |"
        lines.append(header)
        for fam in CAUSE_FAMILIES:
            cells = []
            for n in models:
                c = results[n]['summary'][key][fam]
                cells.append(f"{c['n']} ({c['pct']}%)")
            lines.append(f"| {fam} | " + " | ".join(cells) + " |")

    lines.append("\n## Exact vs relaxed criterion\n")
    lines.append(
        "> These counts are unmatched gold + unmatched predictions under a strict\n"
        "> type-and-boundary criterion, i.e. the FN+FP accounting behind macro F1.\n"
        "> They are deliberately *not* the Table 3 totals, which count error cases\n"
        "> under a greedy 0.5-overlap matcher and merge a near-miss into a single\n"
        "> boundary error instead of one FN plus one FP. Do not compare the two\n"
        "> columns across tables.\n"
    )
    lines.append("| Model | Exact errors | Relaxed errors | Forgiven | Merged preds |")
    lines.append("|---|---|---|---|---|")
    for name, r in results.items():
        m = r['summary'].get('match_criterion')
        if not m:
            continue
        lines.append(
            f"| {name} | {m['exact']['total']} | {m['relaxed']['total']} "
            f"| {m['forgiven_by_relaxed']} ({m['forgiven_pct']}%) "
            f"| {r['summary']['merged_predictions']} |"
        )

    lines.append("\n## Missing spans that are really substitutions\n")
    lines.append("| Model | Type | Missing | Role not predicted at all | "
                 "With same-type substitute | Gold earlier |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in results.items():
        for etype, d in r['summary'].get('substitution', {}).items():
            lines.append(
                f"| {name} | {etype} | {d['missing']} "
                f"| {d['detection_failure']} ({d['detection_failure_pct']}%) "
                f"| {d['missing_with_same_type_substitute']} ({d['pct_of_missing']}%) "
                f"| {d['gold_earlier_than_substitute']} |"
            )

    lines.append("\n## What the spurious spans are\n")
    lines.append("| Model | SPU | Substitution | Duplicate of a filled role | Invention |")
    lines.append("|---|---|---|---|---|")
    for name, r in results.items():
        d = r['summary'].get('spurious_breakdown')
        if not d:
            continue
        lines.append(
            f"| {name} | {d['spurious']} | {d['substitution']} "
            f"| {d['duplicate_of_filled_role']} | {d['invention']} |"
        )

    lines.append("\n## Errors per gold span, multi-vote vs single-vote\n")
    lines.append("| Model | Multi-vote | Single-vote | Ratio |")
    lines.append("|---|---|---|---|")
    for name, r in results.items():
        mv = r['summary'].get('multi_vote')
        if not mv:
            continue
        a = mv['multi_vote']['errors_per_gold_span']
        b = mv['single_vote']['errors_per_gold_span']
        ratio = f"{a / b:.2f}x" if a and b else "n/a"
        lines.append(f"| {name} | {a} | {b} | {ratio} |")

    lines.append("\n## Most frequent edge deltas\n")
    for name, r in results.items():
        s = r['summary']
        if not s.get('n_boundary_errors'):
            continue
        lines.append(f"\n### {name}\n")
        for side in ('left', 'right'):
            for kind in ('added', 'dropped'):
                items = s.get(f'{side}_edge_{kind}', [])[:8]
                if not items:
                    continue
                lines.append(f"**{side} edge, {kind}**\n")
                for txt, cnt in items:
                    snippet = txt if len(txt) <= 70 else txt[:67] + '...'
                    lines.append(f"- `{snippet}` x{cnt}")
                lines.append("")
    return "\n".join(lines) + "\n"


PRETTY = {
    'Gemini-Few-Shot': 'Gemini (5s)',
    'GPT-5.5-Few-Shot': 'GPT (5s)',
    'AMALIA-Few-Shot': 'AMALIA (5s)',
}


def emit_latex(results: Dict, out: str):
    """Emit the appendix boundary-decomposition table as a complete float."""
    lines = [
        "% Generated by scripts/boundary_error_analysis.py -- do not edit by hand.",
        r"\begin{table}[ht!]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Boundary error decomposition on the standard benchmark test "
        r"set (seed 42 for the encoders). \textit{Med.} is the median total edge "
        r"displacement in characters, \textit{Punct.} the share whose gold and "
        r"predicted strings coincide once edge punctuation is stripped, and "
        r"\textit{Mid-word} the share whose gold boundary falls inside an "
        r"orthographic word.}",
        r"\label{tab:boundary_decomposition}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{BND} & \textbf{Over-ext.} & \textbf{Med.} & "
        r"\textbf{Punct.} & \textbf{Mid-word} \\",
        r"\midrule",
    ]
    for name, r in results.items():
        s = r['summary']
        if not s.get('n_boundary_errors'):
            continue
        # Half-up rounding: Python's round() is banker's, which would render
        # 2.5% as 2 here and 3 in a hand-check.
        pct = lambda x: int(x + 0.5)  # noqa: E731
        lines.append(
            f"{PRETTY.get(name, name):<14}& {s['n_boundary_errors']:>4} "
            f"& {pct(s['direction']['over-extended']['pct'])}\\% "
            f"& {s['displacement']['median_chars']:.0f} "
            f"& {pct(s['punct_only']['pct'])}\\% "
            f"& {pct(s['gold_boundary_intraword']['pct'])}\\%" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding='utf-8')
    print(f"✓ {out}")


def _tex_escape(s: str) -> str:
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                 ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'),
                 ('}', r'\}'), ('~', r'\textasciitilde{}'),
                 ('^', r'\textasciicircum{}')]:
        s = s.replace(a, b)
    return s


def emit_delta_latex(results: Dict, out: str, models=None, top: int = 8,
                     maxlen: int = 46):
    """Table of the most frequent strings each model adds at a span edge.

    This is the objective counterpart to a hand-built cause taxonomy: the rows
    are literal substrings with frequencies over every boundary error, so the
    linguistic pattern is visible without anyone having to label anything.
    """
    models = models or ['XLM-R-CRF', 'Gemini-Few-Shot']
    lines = [
        "% Generated by scripts/boundary_error_analysis.py -- do not edit by hand.",
        r"\begin{table}[ht!]",
        r"\centering",
        r"\footnotesize",
        r"\caption{The most frequent strings each model adds at a span edge, "
        r"counted over every boundary error on the test set. No manual labelling "
        r"is involved: these are literal substrings and their frequencies. The "
        r"encoder's additions are punctuation and the single letter of a split "
        r"contraction; the generative model's are articles, honorifics and "
        r"stance clauses that the annotation assigns to other spans.}",
        r"\label{tab:edge_deltas}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Added at edge} & \textbf{$n$} \\",
        r"\midrule",
    ]
    for mi, name in enumerate(models):
        r = results.get(name)
        if not r:
            continue
        merged = Counter()
        for side in ('left', 'right'):
            for txt, n in r['summary'].get(f'{side}_edge_added', []):
                merged[txt] += n
        if mi:
            lines.append(r"\addlinespace[2pt]")
        rows = merged.most_common(top)
        for j, (txt, n) in enumerate(rows):
            label = PRETTY.get(name, name) if j == 0 else ''
            shown = txt if len(txt) <= maxlen else txt[:maxlen - 1] + '…'
            lines.append(f"{label} & \\texttt{{{_tex_escape(shown)}}} & {n}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding='utf-8')
    print(f"✓ {out}")


def emit_cause_latex(results: Dict, out: str, models=None, subject_only: bool = True):
    """Cause-family shares, one column per model (Appendix table).

    Reported over SUBJECT boundary errors by default: that is where three
    quarters of the error mass sits, and the paradigm contrast is sharpest there.
    """
    names = [n for n in (models or results)
             if results.get(n, {}).get('summary', {}).get('n_boundary_errors')]
    key = 'cause_families_subject' if subject_only else 'cause_families'
    scope = (r'\textsc{Subject} boundary errors' if subject_only
             else 'all boundary errors')
    caption = (
        rf'\caption{{Cause families over {scope} (\%), seed 42. Families were '
        r'obtained from the manual sample (Appendix~\ref{app:manual_sample}) and '
        r'restated as tests on the string a prediction adds or drops at an edge, '
        r'so every error is assigned one; \textit{other} is the residual, '
        r'reported rather than absorbed.}'
    )
    lines = [
        '% Generated by scripts/boundary_error_analysis.py -- do not edit by hand.',
        r'\begin{table}[ht!]',
        r'\centering',
        r'\footnotesize',
        caption,
        r'\label{tab:cause_families}',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}l' + 'r' * len(names) + r'@{}}',
        r'\toprule',
        r'\textbf{Cause family} & '
        + ' & '.join(rf'\textbf{{{_tex_escape(SHORT_NAME.get(n, n))}}}' for n in names)
        + r' \\',
        r'\midrule',
        r'\textit{n} & '
        + ' & '.join(
            str(results[n]['summary'][key].get(
                'n_subject_boundary_errors',
                results[n]['summary']['n_boundary_errors']))
            for n in names) + r' \\',
        r'\midrule',
    ]
    for fam in CAUSE_FAMILIES:
        cells = [f"{results[n]['summary'][key][fam]['pct']:.0f}" for n in names]
        lines.append(f'{CAUSE_LABEL[fam]} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text('\n'.join(lines), encoding='utf-8')
    print(f"✓ {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--predictions', action='append', metavar='NAME=PATH')
    parser.add_argument('--gold', default='data/citilink-votie/test.jsonl')
    parser.add_argument('--out-json',
                        default='results/error_analysis/boundary_decomposition.json')
    parser.add_argument('--out-md',
                        default='results/error_analysis/boundary_decomposition.md')
    parser.add_argument('--out-latex',
                        default='ACL_CitiLink___Vote_Identification__José_Evans_/latex/'
                                'boundary_decomposition_table.tex',
                        help='Appendix table; set to "" to skip')
    args = parser.parse_args()

    specs = ([resolve_spec(s) for s in args.predictions]
             if args.predictions else
             [resolve_spec(f"{n}={p}") for n, p in DEFAULT_MODELS])

    classifier = ErrorClassifier()
    gold = classifier.load_gold_standard_spans(args.gold)
    gold_data = load_jsonl(args.gold)
    texts = {ex['id']: ex['text'] for ex in gold_data}
    multi_vote = {ex['id'] for ex in gold_data if ex.get('is_multi_vote')}

    results = {}
    for name, files in specs:
        # The reference seed only: mixing seeds would double-count deltas.
        path = files[0]
        if not Path(path).exists():
            print(f"! skipping {name}, missing {path}")
            continue
        predictions = classifier.load_predictions(path)
        errors = classifier.classify_errors(gold, predictions, texts)
        rows = [describe_boundary(e, texts.get(e.doc_id, ''))
                for e in errors if e.category == 'INC_BOUNDARY']
        results[name] = {
            'prediction_file': path,
            'summary': summarise(rows, errors, gold, predictions, texts, multi_vote),
            'boundary_errors': rows,
        }
        s = results[name]['summary']
        print(f"{name:<20} BND={s['n_boundary_errors']:<5} "
              f"over-ext={s['direction']['over-extended']['pct']:>5}%  "
              f"median disp={s['displacement']['median_chars']:>3.0f}ch  "
              f"punct-only={s['punct_only']['pct']:>5}%  "
              f"forgiven by relaxed={s['match_criterion']['forgiven_pct']:>5}%")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    Path(args.out_md).write_text(render_markdown(results), encoding='utf-8')
    print(f"\n✓ {args.out_json}\n✓ {args.out_md}")
    if args.out_latex:
        emit_latex(results, args.out_latex)
        emit_delta_latex(results,
                         str(Path(args.out_latex).with_name('edge_delta_table.tex')))
        # Cause families are a diagnostic: a rule over delta strings, not an
        # annotation. They stay in the JSON and the markdown report; use
        # emit_cause_latex explicitly if a table is ever wanted.


if __name__ == '__main__':
    main()
