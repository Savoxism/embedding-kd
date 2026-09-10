#!/usr/bin/env python3
"""Corpus subsets for the exposure-scaling sweep.

exp4 needs several corpus sizes N, because the hypothesis makes a prediction about
N/B and not about B alone. This writes them once, deterministically, so every arm
and every seed at a given N trains on exactly the same texts.

Two properties that the sweep depends on:

*Nested.* The N=5000 corpus is a prefix of the N=13553 corpus, which is a prefix
of the next, and so on. Non-nested subsets would confound "the corpus grew" with
"the corpus changed", and the whole point of the sweep is to vary one number.

*Source-stratified.* The union of the three training sets is unbalanced
(banking77 9k, emotion 16k, tweet 24k), so a naive head would make small N nearly
one domain. Each subset keeps the sources in equal proportion, matching the way
`merged_3_data_5k_each.csv` was built.

The largest size available is the deduplicated union of the three train CSVs --
about 48k texts. Sizes beyond that are refused rather than silently truncated:
the sweep's x-axis is N/B, and a point that is not the N it claims would bend the
curve.

Usage:
    python scripts/exp/make_subsets.py --sizes 5000,13553,25000,48000 \
        --out-dir data/train_set/scaling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOURCES = (
    ("banking77", "data/train_set/banking77_train.csv"),
    ("emotion", "data/train_set/emotion_train.csv"),
    ("tweet", "data/train_set/tweet_train.csv"),
)


def anchor_column(frame: pd.DataFrame) -> str:
    for column in ("text", "premise", "sentence1"):
        if column in frame.columns:
            return column
    raise ValueError(f"no anchor column among {list(frame.columns)}")


def load_pool(repo_root: Path, seed: int) -> pd.DataFrame:
    """One shuffled, deduplicated frame with a `source` column."""
    frames = []
    for name, relative in SOURCES:
        path = repo_root / relative
        if not path.is_file():
            raise SystemExit(f"missing source corpus: {path}")
        frame = pd.read_csv(path)
        column = anchor_column(frame)
        frame = frame[[column]].rename(columns={column: "text"})
        frame["source"] = name
        # Shuffled per source before interleaving, so a subset is not the head of
        # whatever order the source file happened to be written in.
        frames.append(frame.sample(frac=1.0, random_state=seed).reset_index(drop=True))

    # Round-robin interleave: taking the first n rows of the result gives an
    # (almost) equal split across sources for every n, which is what makes the
    # subsets both nested and stratified at the same time.
    interleaved = []
    position = 0
    while any(position < len(frame) for frame in frames):
        for frame in frames:
            if position < len(frame):
                interleaved.append(frame.iloc[position])
        position += 1
    pool = pd.DataFrame(interleaved).reset_index(drop=True)

    # The graph is built on deduplicated anchors, so dedup here too: otherwise a
    # requested N would become a smaller N after the training path drops
    # duplicates, and the axis would be wrong by an unknown amount.
    before = len(pool)
    pool = pool[~pool["text"].astype(str).duplicated(keep="first")].reset_index(drop=True)
    print(f"pool: {before} rows -> {len(pool)} unique texts")
    return pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="5000,13553,25000,48000")
    parser.add_argument("--out-dir", default="data/train_set/scaling")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    pool = load_pool(repo_root, args.seed)
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = sorted(int(part) for part in args.sizes.split(",") if part.strip())
    for size in sizes:
        if size > len(pool):
            raise SystemExit(
                f"requested N={size} but only {len(pool)} unique texts exist across "
                f"{', '.join(name for name, _ in SOURCES)}. The scaling sweep is "
                "capped by the corpora in this repository; a larger N needs an "
                "external corpus, and silently truncating would mislabel the axis"
            )
        subset = pool.iloc[:size]
        path = out_dir / f"scaling_n{size}.csv"
        subset.to_csv(path, index=False)
        counts = subset["source"].value_counts().to_dict()
        print(f"  N={size:>6}  {path}  sources={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
