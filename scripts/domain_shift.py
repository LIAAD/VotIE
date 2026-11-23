"""Lexical distance between municipalities, per LOMO fold.

Two questions the §5 LOMO paragraph asserts and this measures:

1. Is cross-municipality variation lexical rather than semantic? Reported as
   mean pairwise TF-IDF cosine within vs. across municipalities, with Cohen's d.
2. Is M06 the fold whose surface form is least like its training municipalities?
   Reported per fold as the cosine between the held-out municipality's TF-IDF
   centroid and the centroid of the five training municipalities, which is the
   quantity the LOMO protocol actually exposes a model to.

Segments are de-duplicated by id across train/dev/test, so each of the 2,885
segments is counted once regardless of the oversampling in the training split.

    python scripts/domain_shift.py --out results/domain_shift.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SPLITS = ("train", "dev", "test")


def load_segments(data_dir: Path) -> dict[str, list[str]]:
    """Deduplicated segment text per municipality."""
    seen: set[str] = set()
    by_muni: dict[str, list[str]] = defaultdict(list)
    for split in SPLITS:
        with open(data_dir / f"{split}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                by_muni[row["municipality"]].append(row["text"])
    return dict(sorted(by_muni.items()))


def cohens_d(a: list[float], b: list[float]) -> float:
    """Pooled-variance standardised mean difference."""
    na, nb = len(a), len(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    return (statistics.mean(a) - statistics.mean(b)) / (pooled ** 0.5)


def pairwise_similarity_split(matrix, labels: list[str], rng, max_pairs: int):
    """Sampled within- and across-municipality pairwise cosine similarities.

    All pairs is 4.2M comparisons; sampling keeps the script interactive and the
    estimates are stable to three decimals at this size.
    """
    n = len(labels)
    idx = np.arange(n)
    within, across = [], []
    while len(within) < max_pairs or len(across) < max_pairs:
        i, j = rng.choice(idx, size=2, replace=False)
        sim = float(matrix[i].dot(matrix[j].T).toarray()[0, 0])
        if labels[i] == labels[j]:
            if len(within) < max_pairs:
                within.append(sim)
        elif len(across) < max_pairs:
            across.append(sim)
    return within, across


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/citilink-votie", type=Path)
    parser.add_argument("--out", default="results/domain_shift.json", type=Path)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    by_muni = load_segments(args.data_dir)
    municipalities = list(by_muni)
    texts, labels = [], []
    for muni in municipalities:
        texts.extend(by_muni[muni])
        labels.extend([muni] * len(by_muni[muni]))

    # Sublinear tf and L2 norms so long segments do not dominate; the vocabulary
    # is fit on the whole corpus so every fold is projected into one space.
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, norm="l2")
    matrix = vectorizer.fit_transform(texts)

    rng = np.random.default_rng(args.seed)
    within, across = pairwise_similarity_split(matrix, labels, rng, args.max_pairs)
    d = cohens_d(within, across)

    label_arr = np.array(labels)
    per_fold = {}
    for muni in municipalities:
        held = matrix[label_arr == muni]
        train = matrix[label_arr != muni]
        held_centroid = np.asarray(held.mean(axis=0)).ravel()
        train_centroid = np.asarray(train.mean(axis=0)).ravel()
        cos = float(
            held_centroid @ train_centroid
            / (np.linalg.norm(held_centroid) * np.linalg.norm(train_centroid))
        )
        per_fold[muni] = {
            "segments": int((label_arr == muni).sum()),
            "centroid_cosine_to_training": round(cos, 4),
        }

    ranked = sorted(per_fold, key=lambda m: per_fold[m]["centroid_cosine_to_training"])
    for rank, muni in enumerate(ranked, start=1):
        per_fold[muni]["distance_rank"] = rank

    report = {
        "corpus": str(args.data_dir),
        "segments": len(texts),
        "vocabulary": int(matrix.shape[1]),
        "pairs_sampled_per_group": args.max_pairs,
        "seed": args.seed,
        "pairwise_tfidf_cosine": {
            "within_municipality_mean": round(statistics.mean(within), 4),
            "across_municipality_mean": round(statistics.mean(across), 4),
            "ratio": round(statistics.mean(within) / statistics.mean(across), 2),
            "cohens_d": round(d, 2),
        },
        "per_fold": per_fold,
        "most_distant_fold": ranked[0],
        "least_distant_fold": ranked[-1],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    p = report["pairwise_tfidf_cosine"]
    print(f"segments {report['segments']}  vocabulary {report['vocabulary']}")
    print(
        f"TF-IDF cosine: within {p['within_municipality_mean']:.4f}  "
        f"across {p['across_municipality_mean']:.4f}  "
        f"ratio {p['ratio']}x  Cohen's d = {p['cohens_d']}"
    )
    print("\nfold  segments  cosine to training  rank (1 = most distant)")
    for muni in municipalities:
        f = per_fold[muni]
        print(
            f"{muni}   {f['segments']:>6}   {f['centroid_cosine_to_training']:>16.4f}"
            f"   {f['distance_rank']:>4}"
        )
    print(f"\nmost distant: {ranked[0]}   least distant: {ranked[-1]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
