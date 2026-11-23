# LOMO Cross-Validation Results — gemini (gemini-2.5-pro)

## Leave-One-Municipality-Out Generalization

Each row shows results when testing on the held-out municipality. Few-shot demonstrations are selected per fold from the five training municipalities only; none originates in the held-out municipality. Provenance for every demonstration is recorded in the per-fold evaluation JSON.

### Entity-Level Span Extraction Results

| Test Municipality | Examples | Precision | Recall | F1 | Gold | TP | FN | FP |
|-------------------|----------|-----------|--------|-----|------|----|----|----|
| M01 | 504 | 0.573 | 0.591 | 0.581 | 1714 | 1301 | 413 | 533 |
| M02 | 398 | 0.436 | 0.440 | 0.437 | 1459 | 979 | 480 | 486 |
| M03 | 720 | 0.434 | 0.433 | 0.433 | 2960 | 1204 | 1756 | 1236 |
| M04 | 235 | 0.391 | 0.418 | 0.404 | 985 | 761 | 224 | 279 |
| M05 | 556 | 0.417 | 0.461 | 0.429 | 1728 | 1041 | 687 | 987 |
| M06 | 472 | 0.557 | 0.496 | 0.522 | 1789 | 1174 | 615 | 522 |
| **Average** | - | **0.468** | **0.473** | **0.468** | - | - | - |

### Per-Entity-Type F1 Scores

| Entity Type | M01 | M02 | M03 | M04 | M05 | M06 | Avg |
|-------------|-------|-------|-------|-------|-------|-------|-----|
| COUNT-BLANK | -- | -- | 1.000 | -- | 1.000 | 0.851 | **0.950** |
| COUNT-FAVOR | -- | -- | -- | -- | -- | 0.844 | **0.844** |
| COUNTING-MAJORITY | 0.987 | 0.986 | -- | 0.964 | 0.800 | 0.442 | **0.836** |
| COUNTING-UNANIMITY | 0.997 | 0.950 | 1.000 | 0.985 | 0.973 | 0.998 | **0.984** |
| SUBJECT | 0.022 | 0.005 | 0.005 | 0.318 | 0.138 | 0.647 | **0.189** |
| VOTER-ABSENT | 0.588 | 0.000 | 0.000 | 0.333 | 0.126 | 0.000 | **0.175** |
| VOTER-ABSTENTION | 0.932 | 0.000 | 0.010 | 0.000 | 0.000 | 0.010 | **0.159** |
| VOTER-AGAINST | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.020 | **0.170** |
| VOTER-FAVOR | 0.991 | 0.985 | 0.975 | 0.976 | 0.225 | 0.000 | **0.692** |
| VOTING | 0.868 | 0.880 | 0.776 | 0.862 | 0.692 | 0.983 | **0.843** |
| VOTING-METHOD | -- | 1.000 | 1.000 | -- | 0.769 | 0.950 | **0.930** |

### Alignment Fidelity

| Municipality | Spans Extracted | Aligned | Alignment Rate | Failed Docs |
|--------------|-----------------|---------|----------------|-------------|
| M01 | 1840 | 1840 | 1.000 | 0 |
| M02 | 1550 | 1550 | 1.000 | 0 |
| M03 | 2830 | 2830 | 1.000 | 0 |
| M04 | 1070 | 1070 | 1.000 | 0 |
| M05 | 2076 | 2076 | 1.000 | 1 |
| M06 | 1776 | 1776 | 1.000 | 5 |

### Experimental Setup

- **Model**: gemini-2.5-pro
- **Pipeline**: langextract (gemini)
- **Strategy**: few_shot (1 example per training municipality, 5 total)
- **Example Selection**: Per-fold, one demonstration per phenomenon, drawn only from the five training municipalities (see src/llm_extraction/lomo_examples.py)
- **Evaluation**: Character-level span matching (exact match)
- **Data Source**: `data/citilink-votie/{train,dev,test}.jsonl`

