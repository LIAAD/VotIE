"""Credential rotation in GptLanguageModel. Run: python tests/test_gpt_rotation.py"""
import os
import sys
from unittest import mock

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.llm_extraction.gpt.config import GptConfig  # noqa: E402
from src.llm_extraction.gpt.langextract_provider import GatewayError, GptLanguageModel  # noqa: E402


def model(n):
    return GptLanguageModel(api_key=None, endpoint="x", channel_id=None,
                            credentials=[(f"k{i}", f"c{i}") for i in range(n)],
                            rate_limit_delay=0, rate_limit_retries=3)


def scripted(lm, outcomes):
    """Each call pops an outcome: an exception to raise, or text to return. Records the key used."""
    used = []

    def call(_prompt):
        used.append(lm.api_key)
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o
    lm._call_once = call
    return used


def rl():
    return GatewayError("Rate limit reached (429)", retryable=True)


def http(status):
    r = requests.Response()
    r.status_code = status
    return requests.HTTPError(f"{status}", response=r)


with mock.patch("time.sleep") as sleep:
    # Rate limit on k0 -> rotates to k1 without sleeping.
    lm = model(3)
    used = scripted(lm, [rl(), "ok"])
    assert lm._call_with_retry("p") == "ok" and used == ["k0", "k1"] and not sleep.called

    # All three limited -> one backoff, then retry continues the cycle.
    lm = model(3)
    used = scripted(lm, [rl(), rl(), rl(), "ok"])
    assert lm._call_with_retry("p") == "ok" and sleep.call_count == 1
    assert used == ["k0", "k1", "k2", "k2"]

    # Gives up after rate_limit_retries rounds.
    lm = model(2)
    scripted(lm, [rl()] * 6)
    try:
        lm._call_with_retry("p")
        raise AssertionError("should have raised")
    except GatewayError:
        pass

    # 401 drops the credential and moves on; the pool shrinks for the rest of the run.
    lm = model(3)
    used = scripted(lm, [http(401), "ok"])
    assert lm._call_with_retry("p") == "ok" and used == ["k0", "k1"]
    assert [k for k, _ in lm.credentials] == ["k1", "k2"]

    # Single credential: 401 is fatal, as before.
    lm = model(1)
    scripted(lm, [http(401)])
    try:
        lm._call_with_retry("p")
        raise AssertionError("should have raised")
    except requests.HTTPError:
        pass

# Config parsing: pool from env; explicit api_key wins.
with mock.patch.dict(os.environ, {"IAEDU_CREDENTIALS": "a:1, b:2", "IAEDU_API_KEY": "z",
                                 "IAEDU_ENDPOINT": "https://example.org/stream"}):
    assert GptConfig().credentials == [("a", "1"), ("b", "2")]
    assert GptConfig(api_key="q").credentials[0][0] == "q"

print("ok")
