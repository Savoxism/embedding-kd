"""Shared reader for the run trees written by the GGPKD runners.

Both aggregators -- summarize.py (per pair) and summarize_sensitivity.py (per
arm) -- read the same three files per training run and apply the same
validation, so that logic lives here rather than being kept in step across two
copies. Only the directory layout differs, which is why the callers pass paths
rather than a pair or an arm name.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch

# (label, result family, benchmark file stem, metric key)
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
TASK_LABELS = tuple(task[0] for task in TASKS)
IN_DOMAIN = ("Emotion", "WiC", "STS-B")
OUT_DOMAIN = ("Banking77", "Tweet", "MRPC", "SciTail", "SICK", "STS12")
AVERAGES = ("Avg In", "Avg Out", "Avg All")
METRIC_ORDER = (*TASK_LABELS, *AVERAGES)

DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_EPOCHS = 5


def read_exit_code(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Missing exit-code file: {path}")
    try:
        code = int(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError(f"Invalid exit-code file: {path}") from exc
    if code != 0:
        raise ValueError(f"Run failed with exit code {code}: {path}")


def final_test_record(metrics_path: Path) -> dict:
    """The one record carrying benchmark scores.

    The shared training loop appends a per-epoch record with `"test": null`
    whenever per-epoch evaluation is off (GGPKD's default), then one final
    record after the last epoch. Selecting on the key alone would therefore
    match all six, so this selects on the value being a result dict.
    """
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
        if isinstance(record.get("test"), dict):
            records.append(record)
    if len(records) != 1:
        raise ValueError(
            f"Expected exactly one final test record in {metrics_path}, "
            f"got {len(records)}"
        )
    return records[0]


def metric_from_family(
    test: dict, family: str, benchmark_stem: str, metric_name: str
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


def validate_final_weight(run_dir: Path, epochs: int) -> None:
    """The reported scores must come from the end of the fixed budget.

    `--final_weights_only` writes exactly one file, so more than one means the
    directory was reused by another run and the metrics.jsonl beside it cannot
    be trusted either.
    """
    weights_dir = run_dir / "weights"
    weights = sorted(weights_dir.glob("student_epoch_*.pt"))
    if len(weights) != 1:
        raise ValueError(
            f"Expected exactly one final student weight in {weights_dir}, "
            f"got {len(weights)}"
        )
    payload = torch.load(weights[0], map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != epochs:
        raise ValueError(f"Final weight payload is not epoch {epochs}: {weights[0]}")


def load_run_scores(
    run_dir: Path, exit_file: Path, seed: int, epochs: int
) -> dict[str, float]:
    """Validate one training run and return its benchmark scores in points."""
    read_exit_code(exit_file)
    validate_final_weight(run_dir, epochs)
    record = final_test_record(run_dir / "metrics.jsonl")
    if int(record.get("seed", -1)) != seed:
        raise ValueError(
            f"Seed mismatch in {run_dir / 'metrics.jsonl'}: "
            f"expected {seed}, got {record.get('seed')}"
        )
    if record.get("method") != "ggpkd":
        raise ValueError(
            f"Not a GGPKD record in {run_dir / 'metrics.jsonl'}: "
            f"method={record.get('method')!r}"
        )
    test = record["test"]
    scores = {
        label: metric_from_family(test, family, stem, metric)
        for label, family, stem, metric in TASKS
    }
    scores["Avg In"] = statistics.fmean(scores[name] for name in IN_DOMAIN)
    scores["Avg Out"] = statistics.fmean(scores[name] for name in OUT_DOMAIN)
    scores["Avg All"] = statistics.fmean(scores[label] for label in TASK_LABELS)
    return scores


def aggregate_seeds(
    seed_scores: dict[int, dict[str, float]], seeds: tuple[int, ...]
) -> dict[str, dict]:
    """Mean and sample standard deviation across seeds, per metric."""
    aggregate = {}
    for metric in METRIC_ORDER:
        values = [seed_scores[seed][metric] for seed in seeds]
        aggregate[metric] = {
            "seeds": dict(zip(seeds, values)),
            "mean": statistics.fmean(values),
            # Sample standard deviation, the spread the paper reports; it needs
            # at least two seeds to mean anything.
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return aggregate


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part) for part in text.split(",") if part.strip())
    if not seeds:
        raise SystemExit("No seeds to aggregate")
    return seeds
