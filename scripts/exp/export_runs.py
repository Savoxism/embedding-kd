#!/usr/bin/env python3
"""Every run tree under one experiment root, flattened into one tidy CSV.

One row per (arm, seed). Four families of column, and each is there because an
analysis that lacks it can reach a wrong conclusion:

    results      the nine benchmarks plus Avg-In / Avg-Out / Avg-All, in points.
    config       what the arm actually was, read from each run's own `run.json`
                 rather than from the label in the runner. A mislabelled arm is
                 the failure mode that survives every other check.
    cost         wall seconds and peak memory, plus the encoder counters the
                 GGPKD step records. Support arms are matched on support *size*,
                 not on compute -- a corpus-uniform draw deduplicates far worse
                 than a teacher draw and so encodes more texts per step -- so the
                 comparison is only honest if that difference is on the table.
    diagnostics  the final epoch's losses and geometry probe, so a suspicious
                 downstream number can be traced to a degenerate objective
                 without re-running anything.

Failed and unfinished runs become rows with `status` set and blank metrics, so a
partially completed sweep still exports and the gaps are visible. Nothing here
aborts on one bad run.

Usage:
    python scripts/exp/export_runs.py results/exp2/<run_id> \
        --experiment exp2_support --out runs/exp2/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ggpkd.run_metrics import (
    IN_DOMAIN,
    OUT_DOMAIN,
    TASK_LABELS,
    TASKS,
    metric_from_family,
)

# Config fields worth carrying into the CSV. Deliberately a fixed list rather than
# the whole snapshot: an arm comparison is read on these, and dumping 80 columns
# makes the two that differ hard to find.
CONFIG_FIELDS = (
    "distill_method",
    "support_policy",
    "relation_target",
    "use_ambient",
    "row_weight",
    "batch_sampler",
    "batch_size",
    "diffusion_quota",
    "graph_k",
    "knn_mode",
    "neighbor_source",
    "row_centers",
    "holdout_edge_frac",
    "holdout_seed",
    "epochs",
    "learning_rate",
    "seed",
    "student_model_name",
    "teacher_model_name",
    "train_data_path",
)

# Training diagnostics from the last epoch record.
TRAIN_FIELDS = (
    "loss",
    "loss_rel",
    "loss_amb",
    "loss_nbr",
    "loss_row",
    "row_exposed_mass",
    "grad_norm",
    "encoded_texts_cum",
    "encoded_tokens_cum",
    "candidates_per_anchor",
    "mean_step_seconds",
)

# Geometry probe from the last epoch record.
GEOMETRY_FIELDS = (
    "teacher_student_spearman",
    "teacher_weighted_distortion",
    "cosine_rmse",
    "anisotropy",
    "effective_rank",
    "uniformity",
    "separation",
    "alignment",
)


def _records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _flatten(prefix: str, source: dict | None, fields) -> dict[str, object]:
    values: dict[str, object] = {f"{prefix}{name}": "" for name in fields}
    if not isinstance(source, dict):
        return values
    for name in fields:
        if name not in source:
            continue
        value = source[name]
        # bool before int: `bool` is a subclass of `int`, so the numeric branch
        # would render use_ambient=False as 0.0 and an analysis grouping on it
        # would read a float column where it expected a flag. int before float
        # for the same reason in reverse: a quota of 15 must not become "15.0".
        if isinstance(value, bool):
            values[f"{prefix}{name}"] = str(value)
        elif isinstance(value, int):
            values[f"{prefix}{name}"] = value
        elif isinstance(value, float):
            values[f"{prefix}{name}"] = round(value, 6) if math.isfinite(value) else ""
        elif isinstance(value, str):
            values[f"{prefix}{name}"] = value
    return values


def _scores(record: dict) -> tuple[dict[str, float], str]:
    """Benchmark scores in points, or a reason they could not be read."""
    test = record.get("test")
    if not isinstance(test, dict):
        return {}, "no_test_record"
    try:
        scores = {
            label: metric_from_family(test, family, stem, metric)
            for label, family, stem, metric in TASKS
        }
    except ValueError as error:
        return {}, f"bad_test_record: {error}"
    scores["Avg In"] = statistics.fmean(scores[name] for name in IN_DOMAIN)
    scores["Avg Out"] = statistics.fmean(scores[name] for name in OUT_DOMAIN)
    scores["Avg All"] = statistics.fmean(scores[label] for label in TASK_LABELS)
    return scores, ""


def collect(run_dir: Path) -> dict[str, object]:
    """One row for one run directory, whatever state it is in."""
    row: dict[str, object] = {}
    status = "ok"

    exit_file = run_dir / "exit"
    manifest = run_dir / "run.json"
    config = {}
    artifact_meta = {}
    if manifest.is_file():
        blob = json.loads(manifest.read_text(encoding="utf-8"))
        config = blob.get("config", {}) or {}
        artifact_meta = (blob.get("artifact") or {}).get("metadata", {}) or {}
        row["run_id"] = blob.get("run_id", "")
        # run.json records the commit as `sha`; reading `commit` left the column
        # empty for every run.
        git = blob.get("git") or {}
        row["git_commit"] = git.get("sha") or git.get("commit") or ""
    else:
        status = "no_manifest"

    # The artifact's own metadata wins for the graph fields: the config records
    # what was requested, the artifact records what was actually trained against.
    merged_config = {
        **config,
        **{k: v for k, v in artifact_meta.items() if k in CONFIG_FIELDS},
    }
    row.update(_flatten("cfg_", merged_config, CONFIG_FIELDS))

    stats_path = run_dir / "run_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for key in ("wall_seconds", "peak_host_rss_mib", "peak_gpu_mib", "exit_code"):
            value = stats.get(key)
            row[f"cost_{key}"] = value if value is not None else ""
    else:
        for key in ("wall_seconds", "peak_host_rss_mib", "peak_gpu_mib", "exit_code"):
            row[f"cost_{key}"] = ""

    # The two files carry different halves of a run and neither is a superset of
    # the other: `metrics.jsonl` holds the one record with benchmark scores,
    # while the per-epoch training means and the geometry probe are appended to
    # `epochs.jsonl`. Reading only the first is how the geometry columns come out
    # empty on a run that measured them perfectly well.
    records = _records(run_dir / "metrics.jsonl")
    epochs = _records(run_dir / "epochs.jsonl")
    final = next(
        (
            record
            for record in reversed(records)
            if isinstance(record.get("test"), dict)
        ),
        None,
    )
    if final is None:
        status = "no_final_test" if status == "ok" else status
        scores = {}
    else:
        scores, reason = _scores(final)
        if reason:
            status = reason

    row.update({f"score_{label}": round(scores[label], 4) for label in scores})
    for label in (*TASK_LABELS, "Avg In", "Avg Out", "Avg All"):
        row.setdefault(f"score_{label}", "")

    last_epoch = epochs[-1] if epochs else (records[-1] if records else None)
    row.update(_flatten("train_", (last_epoch or {}).get("train"), TRAIN_FIELDS))
    row.update(_flatten("geom_", (last_epoch or {}).get("geometry"), GEOMETRY_FIELDS))
    row["epoch"] = (last_epoch or {}).get("epoch", "")
    row["n_epoch_records"] = len(epochs) or len(records)

    if exit_file.is_file():
        code = exit_file.read_text(encoding="utf-8").strip()
        row["exit_code"] = code
        if code not in ("0", ""):
            status = f"exit_{code}"
    else:
        row["exit_code"] = ""
    row["status"] = status
    return row


def _row_key(row: dict) -> tuple[str, str, str, str]:
    return tuple(str(row.get(name, "")) for name in ("experiment", "pair", "arm", "seed"))


def _run_config(root: Path) -> dict[str, str]:
    """The runner's run_config.tsv as a dict; empty when absent."""
    path = root / "run_config.tsv"
    if not path.is_file():
        return {}
    pairs = (line.split("\t", 1) for line in path.read_text(encoding="utf-8").splitlines())
    return {key: value for key, value in (p for p in pairs if len(p) == 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", help="<RESULT_BASE>/<RUN_ID> written by a runner")
    parser.add_argument("--out", required=True)
    parser.add_argument("--experiment", default="")
    parser.add_argument(
        "--pair",
        default="",
        help="teacher->student setting, carried as a column so several pairs "
        "can be concatenated into one analysis frame",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        help="where the runner wrote per-task exit codes; defaults to <root>/status",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="keep rows already in --out whose (experiment, pair, arm, seed) this "
        "export does not produce, instead of overwriting the file. A stage that "
        "runs several pairs, or re-runs one arm, then adds to one table",
    )
    args = parser.parse_args()

    root = Path(args.run_root)
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"no runs/ under {root}")

    status_dir = Path(args.status_file) if args.status_file else root / "status"
    fallback_commit = _run_config(root).get("commit", "")
    if fallback_commit == "unknown":
        fallback_commit = ""
    rows = []
    for seed_dir in sorted(runs_dir.glob("*/seed_*")):
        arm = seed_dir.parent.name
        seed = seed_dir.name.replace("seed_", "")
        row = {
            "experiment": args.experiment,
            "pair": args.pair,
            "arm": arm,
            "seed": seed,
            "run_dir": str(seed_dir),
        }
        # The runner's exit file lives outside the run directory; copy it in so a
        # crashed run is distinguishable from one that never started.
        exit_file = status_dir / f"{arm}.seed_{seed}.exit"
        if exit_file.is_file():
            (seed_dir / "exit").write_text(
                exit_file.read_text(encoding="utf-8"), encoding="utf-8"
            )
        row.update(collect(seed_dir))
        if not row.get("git_commit"):
            row["git_commit"] = fallback_commit
        rows.append(row)

    if not rows:
        raise SystemExit(f"no run directories under {runs_dir}")
    fresh = rows

    out_path = Path(args.out)
    if args.merge and out_path.is_file():
        with out_path.open(newline="", encoding="utf-8") as handle:
            previous = list(csv.DictReader(handle))
        replaced = {_row_key(row) for row in rows}
        kept = [row for row in previous if _row_key(row) not in replaced]
        rows = kept + rows
        print(
            f"merge: kept {len(kept)} existing rows, replaced "
            f"{len(previous) - len(kept)}, added {len(fresh)}"
        )

    # Union of keys, with the identifying columns pinned to the front so the CSV
    # is readable without a spreadsheet.
    lead = ["experiment", "pair", "arm", "seed", "status", "run_id", "git_commit"]
    rest = sorted({key for row in rows for key in row} - set(lead))
    fieldnames = lead + rest

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(out_path)

    ok = sum(1 for row in fresh if row["status"] == "ok")
    print(
        f"wrote {out_path}: {len(fresh)} runs, {ok} ok, {len(fresh) - ok} incomplete"
        f" ({len(rows)} rows in file)"
    )
    for row in fresh:
        if row["status"] != "ok":
            print(f"  {row['arm']} seed {row['seed']}: {row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
