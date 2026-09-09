#!/usr/bin/env python3
"""Validate and aggregate a multi-seed GGPKD run into mean +- std per pair.

Reads the run tree written by scripts/ggpkd/run_paper.sh:

    <run_root>/status/<pair>.seed_<seed>.exit     launcher exit code
    <run_root>/runs/<pair>/seed_<seed>/metrics.jsonl
    <run_root>/runs/<pair>/seed_<seed>/weights/student_epoch_<epochs>.pt

Every one of those is checked before a number is used. A run that crashed after
its last epoch record, or one whose metrics.jsonl was appended to by a second
run sharing the save directory, is a wrong number rather than a missing one --
so each is an error here, not a skipped seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_metrics import (  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_SEEDS,
    aggregate_seeds,
    load_run_scores,
    parse_seeds,
)

# Mirrors the pair table in scripts/ggpkd/train.sh; kept here so the summary
# can name the teacher and student a number came from.
GGPKD_PAPER_PAIRS = {
    "qwen3_0_6b_to_minilmv2_h384": {
        "teacher": "Qwen/Qwen3-Embedding-0.6B",
        "student": "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base",
    },
    "bge_m3_to_minilmv2_h768": {
        "teacher": "BAAI/bge-m3",
        "student": "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base",
    },
    "qwen3_4b_to_bert_base": {
        "teacher": "Qwen/Qwen3-Embedding-4B",
        "student": "google-bert/bert-base-uncased",
    },
}


def load_seed_metrics(
    run_root: Path, pair: str, seed: int, epochs: int
) -> dict[str, float]:
    return load_run_scores(
        run_dir=run_root / "runs" / pair / f"seed_{seed}",
        exit_file=run_root / "status" / f"{pair}.seed_{seed}.exit",
        seed=seed,
        epochs=epochs,
    )


def aggregate_run(
    run_root: Path, pairs: tuple[str, ...], seeds: tuple[int, ...], epochs: int
) -> dict:
    return {
        pair: aggregate_seeds(
            {seed: load_seed_metrics(run_root, pair, seed, epochs) for seed in seeds},
            seeds,
        )
        for pair in pairs
    }


def _format_terminal(aggregate: dict, seeds: tuple[int, ...]) -> str:
    lines = []
    for pair, metrics in aggregate.items():
        lines.append(f"\n{pair}")
        lines.append("Metric       mean ± std")
        lines.append("------------ ---------------")
        for metric, values in metrics.items():
            lines.append(f"{metric:<12} {values['mean']:6.2f} ± {values['std']:.2f}")
    lines.append("")
    lines.append(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    return "\n".join(lines).lstrip()


def write_tsv(run_root: Path, aggregate: dict, seeds: tuple[int, ...]) -> Path:
    path = run_root / "summary.tsv"
    seed_columns = "\t".join(f"seed_{seed}" for seed in seeds)
    lines = [f"pair\tmetric\t{seed_columns}\tmean\tstd\n"]
    for pair, metrics in aggregate.items():
        for metric, values in metrics.items():
            cells = "\t".join(f"{values['seeds'][seed]:.6f}" for seed in seeds)
            lines.append(
                f"{pair}\t{metric}\t{cells}\t{values['mean']:.6f}\t{values['std']:.6f}\n"
            )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def write_markdown(
    run_root: Path, aggregate: dict, seeds: tuple[int, ...], epochs: int
) -> Path:
    path = run_root / "summary.md"
    lines = [
        "# GGPKD paper-pair multi-seed results\n\n",
        f"- Run ID: `{run_root.name}`\n",
        f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`\n",
        f"- Checkpoint: final epoch {epochs}\n",
        "- Values: percentage points, mean ± sample standard deviation\n\n",
    ]
    for pair, metrics in aggregate.items():
        preset = GGPKD_PAPER_PAIRS.get(pair)
        lines.append(f"## {pair}\n\n")
        if preset is not None:
            lines.append(f"`{preset['teacher']}` → `{preset['student']}`\n\n")
        header = " | ".join(f"Seed {seed}" for seed in seeds)
        aligns = "|".join("---:" for _ in seeds)
        lines.append(f"| Metric | {header} | Mean ± std |\n")
        lines.append(f"|---|{aligns}|---:|\n")
        for metric, values in metrics.items():
            cells = " | ".join(f"{values['seeds'][seed]:.2f}" for seed in seeds)
            lines.append(
                f"| {metric} | {cells} | "
                f"{values['mean']:.2f} ± {values['std']:.2f} |\n"
            )
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize multi-seed GGPKD paper-pair runs"
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--pairs",
        default=",".join(GGPKD_PAPER_PAIRS),
        help="Comma-separated pair keys to aggregate",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds each pair was trained with",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training budget, used to check the final weight is the last epoch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    pairs = tuple(part for part in args.pairs.split(",") if part)
    seeds = parse_seeds(args.seeds)
    if not pairs:
        raise SystemExit("No pairs to aggregate")
    aggregate = aggregate_run(run_root, pairs, seeds, args.epochs)
    tsv = write_tsv(run_root, aggregate, seeds)
    markdown = write_markdown(run_root, aggregate, seeds, args.epochs)
    print(_format_terminal(aggregate, seeds))
    print(f"\nTSV: {tsv}")
    print(f"Markdown: {markdown}")


if __name__ == "__main__":
    main()
