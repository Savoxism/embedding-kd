"""Merge every benchmark train split into one unlabelled distillation corpus.

Follows the layout of `merged_3_data_5k_each.csv`: one sentence per row, columns
`text` and `source`, and the pandas index saved alongside so a reader sees
`Unnamed: 0`. Datasets built from sentence pairs contribute both sides as
separate rows, kept adjacent so the original pairing can still be recovered.

Unlike `merged_3_data_5k_each.csv` there is no per-dataset cap: every row of
every train split is included. Duplicates are kept by default, matching the
older file (its WiC block holds 5000 rows over 3793 distinct sentences); pass
--dedup to keep the first occurrence of each exact text instead.

Usage:
    python scripts/build_merged_all.py
    python scripts/build_merged_all.py --dedup --out data/train_set/merged_all_unique.csv
"""

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train_set"

# (source label, filename, columns holding the text). Order fixes the row order
# of the output, so rebuilding the file gives a byte-identical result.
SOURCES = [
    ("BANKING77", "banking77_train.csv", ["text"]),
    ("EMOTION", "emotion_train.csv", ["text"]),
    ("TWEET", "tweet_train.csv", ["text"]),
    ("MRPC", "mrpc_train.csv", ["sentence1", "sentence2"]),
    ("SCITAIL", "scitail_train.csv", ["sentence1", "sentence2"]),
    ("SICK", "sick_train.csv", ["sentence1", "sentence2"]),
    ("STS12", "sts12_train.csv", ["sentence1", "sentence2"]),
    ("STSB", "stsb_train.csv", ["sentence1", "sentence2"]),
    ("WIC", "wic_train.csv", ["sentence1", "sentence2"]),
]


def flatten(frame, columns):
    """One row per sentence, both sides of a pair kept next to each other."""
    if len(columns) == 1:
        return frame[columns[0]].astype(str).tolist()
    stacked = frame[columns].astype(str).to_numpy()
    return stacked.reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=TRAIN_DIR / "merged_all.csv")
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="keep only the first occurrence of each exact text",
    )
    args = parser.parse_args()

    missing = [name for _, name, _ in SOURCES if not (TRAIN_DIR / name).is_file()]
    if missing:
        raise SystemExit(
            "Missing train splits: "
            + ", ".join(missing)
            + "\nRun scripts/download_eval_train_splits.py first."
        )

    blocks = []
    for label, filename, columns in SOURCES:
        frame = pd.read_csv(TRAIN_DIR / filename)
        texts = flatten(frame, columns)
        block = pd.DataFrame({"text": texts, "source": label})

        # A pair whose sides are identical, or a stray NaN, adds a row the student
        # would train on as if it were a sentence.
        blank = block["text"].str.strip().isin(["", "nan"])
        if blank.any():
            print(f"  {label}: dropping {int(blank.sum())} blank rows")
            block = block[~blank]

        print(
            f"  {label:<10} {len(frame):>6} source rows x {len(columns)} "
            f"-> {len(block):>6} sentences ({block.text.nunique()} distinct)"
        )
        blocks.append(block)

    merged = pd.concat(blocks, ignore_index=True)
    if args.dedup:
        before = len(merged)
        merged = merged.drop_duplicates(subset="text", keep="first").reset_index(
            drop=True
        )
        print(f"\ndedup: {before} -> {len(merged)} rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=True)

    print(f"\nWrote {args.out.relative_to(BASE_DIR)}")
    print(f"  rows            : {len(merged)}")
    print(f"  distinct texts  : {merged.text.nunique()}")
    print(f"  sources         : {len(SOURCES)}")
    print(merged.source.value_counts().to_string())


if __name__ == "__main__":
    main()
