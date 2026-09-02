#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.talas_config import TALAS_PAPER_PAIRS


SEEDS = (42, 43, 44)
TASKS = (
    ("Banking77", "classification", "banking77_test", "f1"),
    ("Tweet", "classification", "tweet_test", "f1"),
    ("Emotion", "classification", "emotion_test", "f1"),
    ("MRPC", "pair", "mrpc_test", "average_precision"),
    ("SciTail", "pair", "scitail_test", "average_precision"),
    ("WiC", "pair", "wic_test", "average_precision"),
    ("SICK", "sts", "sick_test", "spearman"),
    ("STS12", "sts", "sts12_test", "spearman"),
    ("STS-B", "sts", "stsb_test", "spearman"),
)
IN_DOMAIN = ("Emotion", "WiC", "STS-B")
OUT_DOMAIN = ("Banking77", "Tweet", "MRPC", "SciTail", "SICK", "STS12")


def _read_exit_code(path: Path) -> int:
    if not path.is_file():
        raise ValueError(f"Missing exit-code file: {path}")
    try:
        code = int(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError(f"Invalid exit-code file: {path}") from exc
    if code != 0:
        raise ValueError(f"Run failed with exit code {code}: {path}")
    return code


def _final_test_record(metrics_path: Path) -> dict:
    if not metrics_path.is_file():
        raise ValueError(f"Missing metrics file: {metrics_path}")
    records = []
    for line_number, line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {metrics_path} at line {line_number}"
            ) from exc
        if "test" in record:
            records.append(record)
    if len(records) != 1:
        raise ValueError(
            f"Expected exactly one final test record in {metrics_path}, got {len(records)}"
        )
    return records[0]


def _metric_from_family(
    test: dict,
    family: str,
    benchmark_stem: str,
    metric_name: str,
) -> float:
    family_values = test.get(family)
    if not isinstance(family_values, dict):
        raise ValueError(f"Missing test family {family!r}")
    matches = [
        value
        for path, value in family_values.items()
        if Path(path).stem == benchmark_stem
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {family}/{benchmark_stem} result, got {len(matches)}"
        )
    value = matches[0]
    if family == "sts" and isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, dict) and metric_name in value:
        score = float(value[metric_name])
    else:
        raise ValueError(f"Missing metric {metric_name!r} for {benchmark_stem}")
    if not math.isfinite(score):
        raise ValueError(f"Non-finite metric for {benchmark_stem}: {score}")
    return 100.0 * score


def _validate_final_weight(run_dir: Path) -> None:
    weights = sorted((run_dir / "weights").glob("student_epoch_*.pt"))
    if len(weights) != 1:
        raise ValueError(
            f"Expected exactly one final student weight in {run_dir / 'weights'}, "
            f"got {len(weights)}"
        )
    payload = torch.load(weights[0], map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != 5:
        raise ValueError(f"Final weight payload is not epoch 5: {weights[0]}")


def load_seed_metrics(run_root: Path, pair: str, seed: int) -> dict[str, float]:
    _read_exit_code(run_root / "status" / f"{pair}.seed_{seed}.exit")
    run_dir = run_root / "runs" / pair / f"seed_{seed}"
    _validate_final_weight(run_dir)
    record = _final_test_record(run_dir / "metrics.jsonl")
    if int(record.get("seed", -1)) != seed:
        raise ValueError(
            f"Seed mismatch in {run_dir / 'metrics.jsonl'}: "
            f"expected {seed}, got {record.get('seed')}"
        )
    test = record["test"]
    scores = {
        label: _metric_from_family(test, family, stem, metric)
        for label, family, stem, metric in TASKS
    }
    scores["Avg In"] = statistics.fmean(scores[name] for name in IN_DOMAIN)
    scores["Avg Out"] = statistics.fmean(scores[name] for name in OUT_DOMAIN)
    scores["Avg All"] = statistics.fmean(
        scores[label] for label, *_ in TASKS
    )
    return scores


def aggregate_run(run_root: Path) -> dict[str, dict[str, dict[str, object]]]:
    aggregate = {}
    for pair in TALAS_PAPER_PAIRS:
        seed_scores = {
            seed: load_seed_metrics(run_root, pair, seed) for seed in SEEDS
        }
        pair_values = {}
        for metric in (*[task[0] for task in TASKS], "Avg In", "Avg Out", "Avg All"):
            values = [seed_scores[seed][metric] for seed in SEEDS]
            pair_values[metric] = {
                "seeds": dict(zip(SEEDS, values)),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values),
            }
        aggregate[pair] = pair_values
    return aggregate


def _format_terminal(aggregate: dict) -> str:
    lines = []
    for pair, metrics in aggregate.items():
        lines.append(f"\n{pair}")
        lines.append("Metric       mean ± std")
        lines.append("------------ ---------------")
        for metric, values in metrics.items():
            lines.append(
                f"{metric:<12} {values['mean']:6.2f} ± {values['std']:.2f}"
            )
    return "\n".join(lines).lstrip()


def write_tsv(run_root: Path, aggregate: dict) -> Path:
    path = run_root / "summary.tsv"
    lines = ["pair\tmetric\tseed_42\tseed_43\tseed_44\tmean\tstd\n"]
    for pair, metrics in aggregate.items():
        for metric, values in metrics.items():
            seeds = values["seeds"]
            lines.append(
                f"{pair}\t{metric}\t{seeds[42]:.6f}\t{seeds[43]:.6f}\t"
                f"{seeds[44]:.6f}\t{values['mean']:.6f}\t{values['std']:.6f}\n"
            )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def write_markdown(run_root: Path, aggregate: dict) -> Path:
    path = run_root / "summary.md"
    lines = [
        "# TALAS paper-pair three-seed results\n\n",
        f"- Run ID: `{run_root.name}`\n",
        "- Server: `H200_Tensara`\n",
        "- Seeds: `42, 43, 44`\n",
        "- Checkpoint: final epoch 5\n",
        "- Values: percentage points, mean ± sample standard deviation\n\n",
    ]
    for pair, metrics in aggregate.items():
        preset = TALAS_PAPER_PAIRS[pair]
        lines.extend(
            [
                f"## {pair}\n\n",
                f"`{preset['teacher']}` → `{preset['student']}`\n\n",
                "| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± std |\n",
                "|---|---:|---:|---:|---:|\n",
            ]
        )
        for metric, values in metrics.items():
            seeds = values["seeds"]
            lines.append(
                f"| {metric} | {seeds[42]:.2f} | {seeds[43]:.2f} | "
                f"{seeds[44]:.2f} | {values['mean']:.2f} ± {values['std']:.2f} |\n"
            )
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize three-seed TALAS paper-pair runs"
    )
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    aggregate = aggregate_run(run_root)
    tsv = write_tsv(run_root, aggregate)
    markdown = write_markdown(run_root, aggregate)
    print(_format_terminal(aggregate))
    print(f"\nTSV: {tsv}")
    print(f"Markdown: {markdown}")


if __name__ == "__main__":
    main()
