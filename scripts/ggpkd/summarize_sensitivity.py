#!/usr/bin/env python3
"""Aggregate a GGPKD sensitivity sweep into one table per swept axis.

Reads the run tree written by scripts/ggpkd/sensitivity.sh:

    <run_root>/arms.tsv                             axis, value, label
    <run_root>/status/<label>.seed_<seed>.exit      launcher exit code
    <run_root>/runs/<label>/seed_<seed>/metrics.jsonl
    <run_root>/runs/<label>/seed_<seed>/weights/student_epoch_<epochs>.pt

Validation is the same as the paper-run summary (scripts/ggpkd/run_metrics.py):
a run that did not finish the budget is an error, not a dropped point, because
a sensitivity curve with a silently missing knot reads as a flat one.

Each axis table carries the delta against the default point and the range its
means span. The range is the number the claim rests on -- a knob is insensitive
when moving it across the grid costs less than seed noise, so the per-arm std
belongs beside it rather than in a separate table.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_metrics import (  # noqa: E402
    AVERAGES,
    DEFAULT_EPOCHS,
    DEFAULT_SEEDS,
    METRIC_ORDER,
    aggregate_seeds,
    load_run_scores,
    parse_seeds,
)

HEADLINE = "Avg All"


def read_arms(run_root: Path) -> list[tuple[str, str, str]]:
    """(axis, value, label) rows, in the order sensitivity.sh wrote them."""
    path = run_root / "arms.tsv"
    if not path.is_file():
        raise ValueError(f"Missing arm table: {path}")
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if line_number == 1 and fields[0] == "axis":
            continue
        if len(fields) != 3:
            raise ValueError(f"Malformed arm row in {path} line {line_number}: {line!r}")
        rows.append((fields[0], fields[1], fields[2]))
    if not rows:
        raise ValueError(f"No arms listed in {path}")
    return rows


def _sort_key(value: str) -> tuple[int, float, str]:
    """Numeric where the axis is numeric, lexical otherwise."""
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


def aggregate_sweep(
    run_root: Path, arms: list[tuple[str, str, str]], seeds: tuple[int, ...], epochs: int
) -> tuple[dict[str, dict], dict[str, list[tuple[str, str]]]]:
    # An arm on two axes (the default point) is loaded once.
    by_label: dict[str, dict] = {}
    for _, _, label in arms:
        if label in by_label:
            continue
        seed_scores = {
            seed: load_run_scores(
                run_dir=run_root / "runs" / label / f"seed_{seed}",
                exit_file=run_root / "status" / f"{label}.seed_{seed}.exit",
                seed=seed,
                epochs=epochs,
            )
            for seed in seeds
        }
        by_label[label] = aggregate_seeds(seed_scores, seeds)

    axes: dict[str, list[tuple[str, str]]] = {}
    for axis, value, label in arms:
        axes.setdefault(axis, []).append((value, label))
    for axis, points in axes.items():
        points.sort(key=lambda point: _sort_key(point[0]))
    return by_label, axes


def _axis_lines(
    axis: str,
    points: list[tuple[str, str]],
    by_label: dict[str, dict],
    default_arm: str,
) -> list[str]:
    baseline = (
        by_label[default_arm][HEADLINE]["mean"]
        if default_arm in by_label
        else None
    )
    lines = [
        f"\n{axis}",
        f"{'value':>8}  {'arm':<18} {'Avg All':>15} {'Δ':>7} "
        f"{'Avg In':>15} {'Avg Out':>15}",
        f"{'-' * 8}  {'-' * 18} {'-' * 15} {'-' * 7} {'-' * 15} {'-' * 15}",
    ]
    for value, label in points:
        metrics = by_label[label]
        cells = " ".join(
            f"{metrics[name]['mean']:6.2f} ± {metrics[name]['std']:.2f}"
            for name in ("Avg In", "Avg Out")
        )
        delta = (
            "  ref"
            if label == default_arm or baseline is None
            else f"{metrics[HEADLINE]['mean'] - baseline:+.2f}"
        )
        lines.append(
            f"{value:>8}  {label:<18} "
            f"{metrics[HEADLINE]['mean']:6.2f} ± {metrics[HEADLINE]['std']:.2f} "
            f"{delta:>7} {cells}"
        )
    means = [by_label[label][HEADLINE]["mean"] for _, label in points]
    stds = [by_label[label][HEADLINE]["std"] for _, label in points]
    lines.append(
        f"{'range':>8}  {HEADLINE} spans {max(means) - min(means):.2f} points "
        f"across {len(points)} values; largest per-arm seed std {max(stds):.2f}"
    )
    return lines


def format_terminal(
    by_label: dict[str, dict],
    axes: dict[str, list[tuple[str, str]]],
    seeds: tuple[int, ...],
    default_arm: str,
) -> str:
    lines: list[str] = []
    for axis, points in axes.items():
        lines.extend(_axis_lines(axis, points, by_label, default_arm))
    lines.append("")
    lines.append(
        f"Seeds: {', '.join(str(seed) for seed in seeds)}; "
        f"default arm: {default_arm}"
    )
    return "\n".join(lines).lstrip()


def write_tsv(
    run_root: Path,
    by_label: dict[str, dict],
    arms: list[tuple[str, str, str]],
    seeds: tuple[int, ...],
) -> Path:
    """Long format, one row per (axis, arm, metric): every number the sweep produced."""
    path = run_root / "sensitivity.tsv"
    seed_columns = "\t".join(f"seed_{seed}" for seed in seeds)
    lines = [f"axis\tvalue\tarm\tmetric\t{seed_columns}\tmean\tstd\n"]
    for axis, value, label in arms:
        for metric in METRIC_ORDER:
            values = by_label[label][metric]
            cells = "\t".join(f"{values['seeds'][seed]:.6f}" for seed in seeds)
            lines.append(
                f"{axis}\t{value}\t{label}\t{metric}\t{cells}\t"
                f"{values['mean']:.6f}\t{values['std']:.6f}\n"
            )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def write_markdown(
    run_root: Path,
    by_label: dict[str, dict],
    axes: dict[str, list[tuple[str, str]]],
    seeds: tuple[int, ...],
    epochs: int,
    default_arm: str,
    pair: str,
) -> Path:
    path = run_root / "sensitivity.md"
    lines = [
        "# GGPKD hyperparameter sensitivity\n\n",
        f"- Run ID: `{run_root.name}`\n",
        f"- Pair: `{pair}`\n",
        f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`\n",
        f"- Checkpoint: final epoch {epochs}\n",
        f"- Default point: `{default_arm}`\n",
        "- Values: percentage points, mean ± sample standard deviation over seeds\n\n",
    ]
    baseline = (
        by_label[default_arm][HEADLINE]["mean"] if default_arm in by_label else None
    )
    for axis, points in axes.items():
        lines.append(f"## {axis}\n\n")
        header = " | ".join(AVERAGES)
        lines.append(f"| {axis} | Arm | {header} | Δ Avg All |\n")
        lines.append("|---|---|---:|---:|---:|---:|\n")
        for value, label in points:
            metrics = by_label[label]
            cells = " | ".join(
                f"{metrics[name]['mean']:.2f} ± {metrics[name]['std']:.2f}"
                for name in AVERAGES
            )
            delta = (
                "ref"
                if label == default_arm or baseline is None
                else f"{metrics[HEADLINE]['mean'] - baseline:+.2f}"
            )
            lines.append(f"| {value} | `{label}` | {cells} | {delta} |\n")
        means = [by_label[label][HEADLINE]["mean"] for _, label in points]
        stds = [by_label[label][HEADLINE]["std"] for _, label in points]
        lines.append(
            f"\nAvg All spans **{max(means) - min(means):.2f}** points across "
            f"{len(points)} values; largest per-arm seed std is "
            f"{max(stds):.2f}, mean seed std {statistics.fmean(stds):.2f}.\n\n"
        )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def read_pair(run_root: Path) -> str:
    config = run_root / "run_config.tsv"
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("\t")
            if key == "pair":
                return value
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate a GGPKD sensitivity sweep into per-axis tables"
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds each arm was trained with",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training budget, used to check the final weight is the last epoch",
    )
    parser.add_argument(
        "--default-arm",
        default="default",
        help="Arm the deltas are measured against",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    seeds = parse_seeds(args.seeds)
    arms = read_arms(run_root)
    by_label, axes = aggregate_sweep(run_root, arms, seeds, args.epochs)
    tsv = write_tsv(run_root, by_label, arms, seeds)
    markdown = write_markdown(
        run_root, by_label, axes, seeds, args.epochs, args.default_arm,
        read_pair(run_root),
    )
    print(format_terminal(by_label, axes, seeds, args.default_arm))
    print(f"\nTSV: {tsv}")
    print(f"Markdown: {markdown}")


if __name__ == "__main__":
    main()
