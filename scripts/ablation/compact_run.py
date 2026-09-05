"""Compact one successful training directory to a single result.csv.

The launcher calls this only after the final evaluation has been validated.
Failed runs are intentionally left untouched so their logs remain debuggable.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ablation.collect import load_run


def compact(run_dir: Path, experiment: str, graph_log_dir: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or not run_dir.name.startswith("seed"):
        raise ValueError(f"refusing to compact unexpected run directory: {run_dir}")

    row = load_run(run_dir, experiment, require_done=False)
    output = run_dir / "result.csv"
    temporary = run_dir / ".result.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)

    # The target has been resolved and checked above. Keep the compact CSV and
    # remove only its siblings inside this exact seed directory.
    for child in run_dir.iterdir():
        if child == output:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    if graph_log_dir is not None:
        graph_log_dir = graph_log_dir.resolve()
        # Graph logs are diagnostics duplicated in the cached artifact's
        # graph_stats. Restrict deletion to the exact per-graph directory passed
        # by the launcher; graph cache tensors remain untouched.
        if graph_log_dir.is_dir() and graph_log_dir.name.startswith("graph_"):
            shutil.rmtree(graph_log_dir)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--graph-log-dir", type=Path)
    args = parser.parse_args()
    output = compact(args.run_dir, args.experiment, args.graph_log_dir)
    print(f"compacted successful run to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
