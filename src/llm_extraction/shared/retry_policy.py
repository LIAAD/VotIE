"""Deciding what a retry is allowed to overwrite, and counting attempts.

WHY THIS EXISTS. The extraction scripts resume by keeping successful records and
re-running everything that errored. That is the right default, but it silently
throws away information when the retry fails for a *different* reason than the
original.

For example, a run whose failures were recorded as runaway generation or
malformed JSON (model failures, reported in the paper) could otherwise be retried
during a network outage, replacing those diagnoses with connection errors while
the document count stayed the same.

The rule here is one line: **a network failure may never overwrite a recorded
model failure.** Everything else behaves as before — a success always wins, and a
model failure replaces an older model failure.

The distinction matters beyond bookkeeping. A model failure is a property of the
model and belongs in the paper. A network failure is a property of our
infrastructure and must not be reported as either.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Substrings identifying a failure as ours rather than the model's. Matching is
# case-insensitive and deliberately conservative: anything not listed counts as a
# model failure, which never flatters the model.
INFRA_MARKERS = (
    "nodename nor servname",   # macOS resolver gave up — never reached the provider
    "temporary failure in name resolution",
    "could not resolve host",
    "connection error",
    "connection reset",
    "connection refused",
    "rate limit",
    "timed out",
    "timeout",
    "service unavailable",
    "bad gateway",
)

# HTTP statuses are matched separately from the substrings above, because a bare
# "429" is not a safe substring. Two real records were misclassified by it: the
# JSON parser reports the failure offset, and
#   "Unterminated string starting at: line 1108 column 7 (char 20429)"
# contains "429" inside the character offset. That is AMALIA's repetition loop —
# the paper's finding — being filed away as our network dropping.
INFRA_STATUS_CODES = (429, 500, 502, 503, 504)

# Offsets the JSON/HTTP libraries print. Stripped before looking for a status so
# their digits cannot be read as one.
_POSITIONAL_RE = re.compile(
    r"\b(?:line|column|col|char|character|position|pos|offset|byte)s?\s*[:=]?\s*\d+",
    re.IGNORECASE,
)
_STATUS_RE = re.compile(
    r"(?<!\d)(?:" + "|".join(str(c) for c in INFRA_STATUS_CODES) + r")(?!\d)"
)


def classify_failure(error: Optional[str]) -> str:
    """``"ok"``, ``"infra"`` (ours) or ``"model"`` (a genuine result)."""
    if not error:
        return "ok"
    lowered = str(error).lower()
    if any(m in lowered for m in INFRA_MARKERS):
        return "infra"
    return "infra" if _STATUS_RE.search(_POSITIONAL_RE.sub(" ", lowered)) else "model"


def merge_retry(prior: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the record to keep for a document that has now been tried twice.

    Also maintains ``attempts``, because the failure rate the paper reports is
    meaningless without it: these models are not reproducible, so "N% of documents
    failed" is only interpretable as "failed on all K attempts".
    """
    if prior is None:
        return {**new, "attempts": new.get("attempts", 1)}

    attempts = prior.get("attempts", 1) + 1
    prior_kind = classify_failure(prior.get("error"))
    new_kind = classify_failure(new.get("error"))

    # A network error tells us nothing about the model. If we already knew how the
    # model failed here, that diagnosis is worth more than "the VPN dropped".
    if prior_kind == "model" and new_kind == "infra":
        kept = dict(prior)
        kept["attempts"] = attempts
        kept["retry_inconclusive"] = True
        kept["last_retry_error"] = new.get("error")
        return kept

    kept = dict(new)
    kept["attempts"] = attempts
    kept.pop("retry_inconclusive", None)
    kept.pop("last_retry_error", None)
    return kept


def merge_all(prior_by_id: Dict[str, Dict[str, Any]],
              new_records: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply :func:`merge_retry` across a batch, returning the id -> record map."""
    merged = dict(prior_by_id)
    for record in new_records:
        merged[record["id"]] = merge_retry(prior_by_id.get(record["id"]), record)
    return merged


def summarise(records) -> Dict[str, int]:
    """Counts by failure source — the shape every report of these runs needs."""
    counts = {"ok": 0, "infra": 0, "model": 0, "inconclusive_retries": 0}
    for r in records:
        counts[classify_failure(r.get("error"))] += 1
        if r.get("retry_inconclusive"):
            counts["inconclusive_retries"] += 1
    return counts
