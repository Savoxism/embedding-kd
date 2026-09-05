"""Validate and aggregate the locked R=1 paper experiment suite.

The collector reads only one experiment key, requires the exact 66-run matrix by
default, and writes one CSV per paper table from the compact per-run result
files. It never scans the legacy ``ablations/results.csv``.

Usage:
    python scripts/ablation/collect.py
    python scripts/ablation/collect.py --allow-incomplete
    python scripts/ablation/collect.py --root /path/to/runs --experiment my_key
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.distill.benchmarks import ALL_BENCHMARKS, primary_scores

PRIMARY_PAIR = "qwen3_0_6b_to_minilmv2_h384"
MAIN_PAIRS = (
    PRIMARY_PAIR,
    "bge_m3_to_minilmv2_h768",
    "qwen3_4b_to_bert_base",
)
SEED_DEFAULT = (42, 43, 44)

# (display label, pair, group, arm). A run may appear in more than one table by
# design; it is still trained only once.
TABLES: dict[str, list[tuple[str, str, str, str]]] = {
    "table1_main": [(pair, pair, "full", "full") for pair in MAIN_PAIRS],
    "table2_support": [
        ("topk", PRIMARY_PAIR, "full", "full"),
        ("proportional", PRIMARY_PAIR, "support", "proportional"),
        ("uniform_pool", PRIMARY_PAIR, "support", "uniform_pool"),
        ("uniform_corpus", PRIMARY_PAIR, "shared", "no_graph_support"),
        ("batch_local", PRIMARY_PAIR, "support", "batch_local"),
    ],
    "table3_components": [
        ("full", PRIMARY_PAIR, "full", "full"),
        ("no_ambient", PRIMARY_PAIR, "components", "no_ambient"),
        ("no_row", PRIMARY_PAIR, "components", "no_row"),
        ("no_ambient_no_row", PRIMARY_PAIR, "components", "no_ambient_no_row"),
        ("no_graph_and_diffusion", PRIMARY_PAIR, "shared", "no_graph_support"),
    ],
    "table4_radius": [
        ("r_1", PRIMARY_PAIR, "full", "full"),
        ("r_1_2", PRIMARY_PAIR, "radius", "r_1_2"),
        ("r_1_2_4", PRIMARY_PAIR, "radius", "r_1_2_4"),
    ],
    "table5_sensitivity": [
        ("topk:0.5", PRIMARY_PAIR, "sensitivity", "topk_0_5x"),
        ("topk:0.75", PRIMARY_PAIR, "sensitivity", "topk_0_75x"),
        ("topk:1", PRIMARY_PAIR, "full", "full"),
        ("topk:1.5", PRIMARY_PAIR, "sensitivity", "topk_1_5x"),
        ("topk:2", PRIMARY_PAIR, "sensitivity", "topk_2x"),
        ("topk:2.5", PRIMARY_PAIR, "sensitivity", "topk_2_5x"),
        ("topk:3", PRIMARY_PAIR, "sensitivity", "topk_3x"),
        ("row_weight:0", PRIMARY_PAIR, "components", "no_row"),
        ("row_weight:0.1", PRIMARY_PAIR, "sensitivity", "row_0_1"),
        ("row_weight:0.25", PRIMARY_PAIR, "sensitivity", "row_0_25"),
        ("row_weight:0.5", PRIMARY_PAIR, "sensitivity", "row_0_5"),
        ("row_weight:0.75", PRIMARY_PAIR, "sensitivity", "row_0_75"),
        ("row_weight:1", PRIMARY_PAIR, "full", "full"),
    ],
    "table6_efficiency": [(pair, pair, "full", "full") for pair in MAIN_PAIRS],
}

METRICS = (
    "sts_avg",
    "pair_avg",
    "cls_avg",
    "avg",
    "avg_in",
    "avg_out",
    "teacher_weighted_distortion",
    "teacher_student_spearman",
    "encoded_texts_cum",
    "encoded_tokens_cum",
    "peak_memory_mb",
    "mean_step_seconds",
    "wall_clock_seconds",
) + tuple(ALL_BENCHMARKS)
FAMILIES = {
    "sts": ("sick", "sts12", "stsb"),
    "pair": ("mrpc", "scitail", "wic"),
    "cls": ("banking77", "tweet", "emotion"),
}


def _jsonl(path: Path, run_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") == run_id:
                rows.append(row)
    return rows


def _final_test(save_dir: Path, run_id: str) -> dict[str, Any] | None:
    for filename in ("metrics.jsonl", "epochs.jsonl"):
        for row in reversed(_jsonl(save_dir / filename, run_id)):
            if isinstance(row.get("test"), dict):
                return row["test"]
    return None


def load_run(
    save_dir: Path, experiment: str, *, require_done: bool = True
) -> dict[str, Any]:
    required = ("request.json", "arm.json", "run.json")
    if require_done:
        required = (".done", *required)
    missing = [name for name in required if not (save_dir / name).exists()]
    if missing:
        raise ValueError(f"{save_dir}: missing {', '.join(missing)}")

    arm = json.loads((save_dir / "arm.json").read_text(encoding="utf-8"))
    request = json.loads((save_dir / "request.json").read_text(encoding="utf-8"))
    manifest = json.loads((save_dir / "run.json").read_text(encoding="utf-8"))
    run_id = str(manifest["run_id"])
    config = manifest.get("config") or {}

    if request.get("experiment") != experiment:
        raise ValueError(
            f"{save_dir}: request experiment={request.get('experiment')!r}, "
            f"expected {experiment!r}"
        )
    if int(config.get("eval_every", -1)) != 0:
        raise ValueError(f"{save_dir}: Table 1 protocol requires eval_every=0")
    if config.get("final_weights_only") is not True:
        raise ValueError(f"{save_dir}: protocol requires final_weights_only=true")

    test = _final_test(save_dir, run_id)
    if test is None:
        raise ValueError(f"{save_dir}: no final test record for run_id={run_id}")
    scores = primary_scores(test)
    absent = sorted(set(ALL_BENCHMARKS) - set(scores))
    if absent:
        raise ValueError(f"{save_dir}: final test is missing {absent}")

    parts = save_dir.resolve().parts
    try:
        experiment_index = len(parts) - 1 - parts[::-1].index(experiment)
        pair = parts[experiment_index - 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"{save_dir}: cannot resolve pair before {experiment}"
        ) from exc

    epochs = _jsonl(save_dir / "epochs.jsonl", run_id)
    epoch = epochs[-1] if epochs else {}
    train = epoch.get("train") or {}
    geometry = epoch.get("geometry") or {}
    step_seconds = [
        float(row["step_seconds"])
        for row in _jsonl(save_dir / "step_metrics.jsonl", run_id)
        if row.get("step_seconds") is not None
    ]
    row: dict[str, Any] = {
        "pair": pair,
        "experiment": experiment,
        "group": arm.get("group", arm.get("ablation")),
        "arm": arm["arm"],
        "seed": int(arm["seed"]),
        "run_id": run_id,
        "save_dir": str(save_dir),
        "git_sha": (manifest.get("git") or {}).get("sha"),
        "git_dirty": (manifest.get("git") or {}).get("dirty"),
        "gpu": (manifest.get("env") or {}).get("gpu"),
        "graph_key": arm.get("graph_key"),
        "wall_clock_seconds": arm.get("wall_clock_seconds"),
        "teacher_cache_warm_before": arm.get("teacher_cache_warm_before"),
        "graph_cache_warm_before": arm.get("graph_cache_warm_before"),
        "started_at": arm.get("started_at"),
        "timing_scope": arm.get("timing_scope"),
        "request_sha256": hashlib.sha256(
            (save_dir / "request.json").read_bytes()
        ).hexdigest(),
        "code_sha256": request.get("code_sha256"),
    }
    for family, names in FAMILIES.items():
        row[f"{family}_avg"] = 100.0 * statistics.fmean(scores[name] for name in names)
    row["avg"] = 100.0 * statistics.fmean(scores[name] for name in ALL_BENCHMARKS)
    row["avg_in"] = test.get("avg_in")
    row["avg_out"] = test.get("avg_out")
    for name in ALL_BENCHMARKS:
        row[name] = 100.0 * scores[name]
    for key in (
        "teacher_weighted_distortion",
        "teacher_student_spearman",
    ):
        row[key] = geometry.get(key)
    for key in ("encoded_texts_cum", "encoded_tokens_cum"):
        row[key] = train.get(key)
    epoch_peaks = [
        float(record["train"]["peak_memory_mb"])
        for record in epochs
        if isinstance(record.get("train"), dict)
        and record["train"].get("peak_memory_mb") is not None
    ]
    row["peak_memory_mb"] = max(epoch_peaks) if epoch_peaks else None
    row["mean_step_seconds"] = (
        statistics.fmean(step_seconds)
        if step_seconds
        else train.get("mean_step_seconds")
    )
    for key in (
        "support_policy",
        "relation_target",
        "use_ambient",
        "row_weight",
        "diffusion_scales",
        "diffusion_quota",
        "hard_neg_k",
        "random_neg_k",
        "learning_rate",
        "epochs",
        "batch_size",
        "eval_every",
        "final_weights_only",
    ):
        row[key] = config.get(key)
    return row


_FLOAT_FIELDS = set(METRICS) | {"row_weight", "learning_rate"}
_INT_FIELDS = {
    "seed",
    "diffusion_quota",
    "hard_neg_k",
    "random_neg_k",
    "epochs",
    "batch_size",
    "eval_every",
}
_BOOL_FIELDS = {
    "git_dirty",
    "teacher_cache_warm_before",
    "graph_cache_warm_before",
    "use_ambient",
    "final_weights_only",
}


def read_compact_result(path: Path, experiment: str) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one result row, found {len(rows)}")
    row: dict[str, Any] = dict(rows[0])
    if row.get("experiment") != experiment:
        raise ValueError(
            f"{path}: experiment={row.get('experiment')!r}, expected {experiment!r}"
        )
    for key in _FLOAT_FIELDS:
        if row.get(key) not in (None, ""):
            row[key] = float(row[key])
        else:
            row[key] = None
    for key in _INT_FIELDS:
        if row.get(key) not in (None, ""):
            row[key] = int(row[key])
        else:
            row[key] = None
    for key in _BOOL_FIELDS:
        if row.get(key) not in (None, ""):
            if row[key] not in {"True", "False"}:
                raise ValueError(f"{path}: invalid boolean {key}={row[key]!r}")
            row[key] = row[key] == "True"
        else:
            row[key] = None
    if row.get("eval_every") != 0 or row.get("final_weights_only") is not True:
        raise ValueError(f"{path}: not a final-only, eval_every=0 paper run")
    return row


def discover(root: Path, experiment: str) -> list[dict[str, Any]]:
    candidates = sorted(root.glob(f"*/{experiment}/*/*/seed*/result.csv"))
    return [read_compact_result(path, experiment) for path in candidates]


def _mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return statistics.fmean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0


def _paired_delta(
    rows: list[dict[str, Any]], baseline: dict[int, dict[str, Any]], metric: str
) -> tuple[float | None, float | None]:
    values = [
        row[metric] - baseline[row["seed"]][metric]
        for row in rows
        if row["seed"] in baseline
        and row.get(metric) is not None
        and baseline[row["seed"]].get(metric) is not None
    ]
    return _mean_std(values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def _coverage_lookup(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["policy"], []).append(row)
    result = {}
    for policy, rows in grouped.items():
        final_epoch = max(int(row["epoch"]) for row in rows)
        final = [row for row in rows if int(row["epoch"]) == final_epoch]
        result[policy] = {
            "coverage_final_mean": statistics.fmean(
                float(row["coverage_mean"]) for row in final
            ),
            "restriction_error_final_mean": statistics.fmean(
                float(row["epsilon_mean"]) for row in final
            ),
        }
    return result


def aggregate_table(
    name: str,
    specs: list[tuple[str, str, str, str]],
    runs_by_key: dict[tuple[str, str, str], list[dict[str, Any]]],
    seeds: tuple[int, ...],
    coverage: dict[str, dict[str, float]],
    allow_incomplete: bool,
) -> list[dict[str, Any]]:
    output = []
    baseline_rows = runs_by_key.get((PRIMARY_PAIR, "full", "full"), [])
    baseline = {row["seed"]: row for row in baseline_rows}
    for label, pair, group, arm in specs:
        rows = runs_by_key.get((pair, group, arm), [])
        found = {row["seed"] for row in rows}
        missing = sorted(set(seeds) - found)
        if missing and not allow_incomplete:
            raise ValueError(f"{name}/{label}: missing seeds {missing}")
        if not rows:
            continue
        record: dict[str, Any] = {
            "variant": label,
            "pair": pair,
            "group": group,
            "arm": arm,
            "n": len(rows),
            "seeds": ",".join(
                str(row["seed"]) for row in sorted(rows, key=lambda x: x["seed"])
            ),
        }
        if name == "table5_sensitivity":
            record["axis"], record["value"] = label.split(":")
        for metric in METRICS:
            mean, std = _mean_std([row.get(metric) for row in rows])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        if pair == PRIMARY_PAIR:
            delta, delta_std = _paired_delta(rows, baseline, "avg")
            record["avg_paired_delta_mean"] = delta
            record["avg_paired_delta_std"] = delta_std
        if name == "table2_support":
            coverage_key = "uniform" if label == "uniform_pool" else label
            if coverage_key in coverage:
                record.update(coverage[coverage_key])
            elif label in {"uniform_corpus", "batch_local"}:
                record["coverage_final_mean"] = 0.0
                record["restriction_error_final_mean"] = None
        if name == "table6_efficiency":
            warm = [
                row
                for row in rows
                if row.get("teacher_cache_warm_before") is True
                and row.get("graph_cache_warm_before") is True
            ]
            warm_mean, warm_std = _mean_std(
                [row.get("wall_clock_seconds") for row in warm]
            )
            record["warm_end_to_end_n"] = len(warm)
            record["warm_end_to_end_seconds_mean"] = warm_mean
            record["warm_end_to_end_seconds_std"] = warm_std
        output.append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--experiment", default="paper_r1_v2")
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in SEED_DEFAULT)
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    out = args.out or args.root / "tables" / args.experiment

    runs = discover(args.root, args.experiment)
    if not runs:
        raise SystemExit(
            f"no completed runs found under {args.root} for {args.experiment}"
        )
    runs_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in runs:
        runs_by_key.setdefault((row["pair"], row["group"], row["arm"]), []).append(row)

    expected_unique = {
        (pair, group, arm) for specs in TABLES.values() for _, pair, group, arm in specs
    }
    if not args.allow_incomplete:
        unexpected = sorted(set(runs_by_key) - expected_unique)
        if unexpected:
            raise ValueError(f"unexpected configurations in suite: {unexpected}")

    _write_csv(out / "runs.csv", runs)
    coverage = _coverage_lookup(
        args.root / PRIMARY_PAIR / args.experiment / "tables" / "table2_coverage.csv"
    )
    for name, specs in TABLES.items():
        table = aggregate_table(
            name, specs, runs_by_key, seeds, coverage, args.allow_incomplete
        )
        _write_csv(out / f"{name}.csv", table)

    expected_run_count = len(expected_unique) * len(seeds)
    if not args.allow_incomplete and len(runs) != expected_run_count:
        raise ValueError(
            f"suite has {len(runs)} runs; expected {expected_run_count} "
            f"({len(expected_unique)} configurations x {len(seeds)} seeds)"
        )
    print(f"validated {len(runs)} runs for experiment={args.experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
