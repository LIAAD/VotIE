# Generative standard benchmark — consolidated status

Generated 2026-11-23T06:05:15 by `scripts/llm_extraction/consolidate_standard_benchmark.py`.
Source: `results/generative`  ·  gold: `data/citilink-votie/test.jsonl` (529 documents).

**6 of 6 configurations complete.**  Table 1 can be built.

A run is complete only when it covers every gold document, has no duplicate ids,
has zero *infrastructure* failures, uses a single model id, and captured token usage.
Model failures do not block completion — they are a result, not an error.

| Config | Docs | OK | Infra fail | Model fail | Entities | Discard | Cost | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `gemini/zero_shot` | 529 | 529 | 0 | 0 | 2082 | 1.4% | $15.36 | **COMPLETE** |
| `gemini/few_shot` | 529 | 529 | 0 | 0 | 1994 | 0.2% | $17.75 | **COMPLETE** |
| `gpt5.5/zero_shot` | 529 | 529 | 0 | 0 | 1981 | 0.1% | $6.85 | **COMPLETE** |
| `gpt5.5/few_shot` | 529 | 529 | 0 | 0 | 2038 | 0.1% | $8.68 | **COMPLETE** |
| `amalia/zero_shot` | 529 | 473 | 0 | 56 | 1954 | 16.0% | self-hosted | **COMPLETE** |
| `amalia/few_shot` | 529 | 416 | 0 | 113 | 2515 | 9.8% | self-hosted | **COMPLETE** |
## Complete runs — paper-facing numbers

| Config | Model failure rate | Span discard | Bad-type spans | Latency s/doc | Wall clock |
|---|---:|---:|---:|---:|---:|
| `gemini/zero_shot` | 0.00% | 1.4% | 0.0% | 23.8 | 3.49 h |
| `gemini/few_shot` | 0.00% | 0.2% | 0.0% | 29.0 | 4.26 h |
| `gpt5.5/zero_shot` | 0.00% | 0.1% | 0.0% | 13.6 | 1.99 h |
| `gpt5.5/few_shot` | 0.00% | 0.1% | 0.0% | 13.2 | 1.93 h |
| `amalia/zero_shot` | 10.59% | 16.0% | 23.0% | 68.7 | 10.09 h |
| `amalia/few_shot` | 21.36% | 9.8% | 1.6% | 110.9 | 16.29 h |

| Config | Exact P | Exact R | Exact F1 | Relaxed F1 |
|---|---:|---:|---:|---:|
| `gemini/zero_shot` | 63.0% | 64.3% | 63.1% | 88.0% |
| `gemini/few_shot` | 62.5% | 63.8% | 62.8% | 88.9% |
| `gpt5.5/zero_shot` | 54.7% | 54.5% | 53.9% | 82.7% |
| `gpt5.5/few_shot` | 58.6% | 60.8% | 59.4% | 86.2% |
| `amalia/zero_shot` | 31.0% | 23.9% | 25.9% | 42.7% |
| `amalia/few_shot` | 37.6% | 43.7% | 35.6% | 49.2% |
