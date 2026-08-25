"""Fetch the train splits of the evaluation benchmarks that ship without one.

Only the three classification probes read a train file today: `eval_pair_task`
scores cosine against a threshold picked on validation, and `eval_sts_task`
correlates cosine with the gold score, so neither fits anything. The six splits
downloaded here are therefore not required by the current protocol -- they are
here so the benchmarks are complete on disk.

Provenance is not assumed. For every benchmark the script first rebuilds the
validation and test files that are already in the repo from the upstream source
and refuses to write the train file unless they match value-for-value, so a
source that has silently diverged fails loudly instead of landing a train split
that belongs to a different release of the dataset.

Usage:
    python scripts/download_eval_train_splits.py
    python scripts/download_eval_train_splits.py --only sick stsb --force
"""

import argparse
import io
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_DIR = BASE_DIR / "data" / "val_set"
TEST_DIR = BASE_DIR / "data" / "test_set"

SICK_MIRROR = (
    "https://raw.githubusercontent.com/brmson/dataset-sts/master/"
    "data/sts/sick2014/SICK_{part}.txt"
)


def _hf(path, config=None):
    from datasets import load_dataset

    return load_dataset(path, config) if config else load_dataset(path)


def _sick_part(part):
    """One file of the original SemEval-2014 Task 1 release (4500/500/4927).

    The Zenodo redistribution of SICK is the deduplicated 9840-pair version
    (4439/495/4906) and does not reproduce the splits already in this repo.
    """
    request = urllib.request.Request(
        SICK_MIRROR.format(part=part), headers={"User-Agent": "curl/8"}
    )
    raw = urllib.request.urlopen(request, timeout=60).read()
    frame = pd.read_csv(io.BytesIO(raw), sep="\t")
    return frame.rename(
        columns={
            "sentence_A": "sentence1",
            "sentence_B": "sentence2",
            "relatedness_score": "score",
        }
    )[["pair_ID", "sentence1", "sentence2", "score", "entailment_judgment"]]


def mrpc():
    splits = _hf("nyu-mll/glue", "mrpc")
    columns = ["sentence1", "sentence2", "label", "idx"]
    frames = {k: splits[k].to_pandas()[columns] for k in ("train", "validation", "test")}
    return frames, False


def scitail():
    splits = _hf("allenai/scitail", "dgem_format")
    columns = ["sentence1", "sentence2", "label", "hypothesis_graph_structure"]

    def convert(split):
        frame = splits[split].to_pandas().rename(
            columns={"premise": "sentence1", "hypothesis": "sentence2"}
        )
        frame["label"] = frame["label"].map({"entails": 1, "neutral": 0})
        if frame["label"].isna().any():
            raise ValueError("unmapped SciTail label")
        return frame[columns].astype({"label": "int64"})

    frames = {k: convert(k) for k in ("train", "validation", "test")}
    # The sibling files carry a saved pandas index as `Unnamed: 0`; keep it so the
    # three splits stay one schema.
    return frames, True


def wic():
    splits = _hf("Deehan1866/WiC")
    columns = ["phrase1", "phrase2", "sentence1", "sentence2", "label", "idx"]
    frames = {k: splits[k].to_pandas()[columns] for k in ("train", "validation", "test")}
    return frames, False


def sick():
    frames = {
        "train": _sick_part("train"),
        "validation": _sick_part("trial"),
        "test": _sick_part("test_annotated"),
    }
    return frames, False


def sts12():
    splits = _hf("mteb/sts12-sts")
    columns = ["split", "sentence1", "sentence2", "score"]
    train = splits["train"].to_pandas()[columns]
    validation = pd.read_csv(VAL_DIR / "sts12_validation.csv")

    # STS12 upstream has no validation split: this repo's sts12_validation.csv is a
    # 734-row slice of the upstream *train* split. Handing back the full 2234 rows
    # would put those rows on both sides, so the slice is removed here.
    #
    # The key is normalised text only, not text plus score: at least one pair is
    # duplicated upstream with a different capitalisation, and an exact-string key
    # leaves that copy in train while its twin sits in validation.
    def key(frame):
        return frame["sentence1"].map(_normalize) + " ||| " + frame["sentence2"].map(
            _normalize
        )

    held_out = set(key(validation))
    keep = ~key(train).isin(held_out)
    dropped = int((~keep).sum())
    print(
        f"    sts12: dropped {dropped} of {len(train)} upstream train rows that are "
        f"already sts12_validation.csv"
    )
    frames = {"train": train[keep].reset_index(drop=True), "test": splits["test"].to_pandas()[columns]}
    return frames, False


def stsb():
    splits = _hf("mteb/stsbenchmark-sts")
    columns = [
        "split", "genre", "dataset", "year", "sid", "score", "sentence1", "sentence2",
    ]
    frames = {k: splits[k].to_pandas()[columns] for k in ("train", "validation", "test")}
    return frames, False


BENCHMARKS = {
    "mrpc": mrpc,
    "scitail": scitail,
    "wic": wic,
    "sick": sick,
    "sts12": sts12,
    "stsb": stsb,
}


def _normalize(text):
    """Case- and whitespace-insensitive form, for matching pairs across splits."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _column_differences(rebuilt, existing):
    """Count real value differences, ignoring CSV round-trip formatting.

    Upstream stores some numeric ids as zero-padded strings ('0024'); reading the
    repo's CSV back yields the integer 24. Comparing those as text reports every
    row as a difference, so numeric columns are compared as numbers.
    """
    left = pd.to_numeric(rebuilt, errors="coerce")
    right = pd.to_numeric(existing, errors="coerce")
    if not left.isna().any() and not right.isna().any():
        return int((~pd.Series(left).sub(right).abs().le(1e-9)).sum())
    return int((rebuilt.astype(str) != existing.astype(str)).sum())


def check_against_repo(name, split, rebuilt, path):
    """Fail unless the upstream source reproduces a split already on disk."""
    if not path.is_file():
        return f"{split}: no local file to check against"
    existing = pd.read_csv(path)
    columns = [c for c in existing.columns if c != "Unnamed: 0"]
    if len(rebuilt) != len(existing):
        raise SystemExit(
            f"{name} {split}: upstream has {len(rebuilt)} rows, "
            f"{path.name} has {len(existing)}. Refusing to write a train split "
            f"from a source that does not match the rest of the benchmark."
        )
    rebuilt = rebuilt.reset_index(drop=True)
    existing = existing.reset_index(drop=True)
    for column in columns:
        differences = _column_differences(rebuilt[column], existing[column])
        if differences:
            raise SystemExit(
                f"{name} {split}: column {column!r} differs from {path.name} "
                f"on {differences} rows."
            )
    return f"{split}: {len(existing)} rows reproduced exactly"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(BENCHMARKS))
    parser.add_argument(
        "--out-dir", type=Path, default=BASE_DIR / "data" / "train_set"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing train file"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = args.only or sorted(BENCHMARKS)
    summary = []

    for name in selected:
        print(f"[{name}]")
        destination = args.out_dir / f"{name}_train.csv"
        if destination.exists() and not args.force:
            print(f"    {destination} exists; skipping (use --force)")
            continue

        frames, keep_index = BENCHMARKS[name]()
        for split, directory, suffix in (
            ("validation", VAL_DIR, "validation"),
            ("test", TEST_DIR, "test"),
        ):
            if split in frames:
                print(
                    "    "
                    + check_against_repo(
                        name, split, frames[split], directory / f"{name}_{suffix}.csv"
                    )
                )

        train = frames["train"]
        train.to_csv(destination, index=keep_index)
        print(f"    wrote {destination} ({len(train)} rows)")
        summary.append((name, len(train), destination))

    if summary:
        print("\nWrote:")
        for name, rows, destination in summary:
            print(f"  {name:<10} {rows:>6} rows  {destination.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    sys.exit(main())
