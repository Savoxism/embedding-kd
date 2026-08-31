"""Run provenance and the per-epoch record.

Two problems this exists to fix.

*Runs were not identifiable.* `metrics.jsonl` carried only the method and the
seed, every file was opened in append mode, and nothing recorded the code or the
graph a run was produced by -- so recovering what a number meant took reading
notebook output. Every file written now carries a `run_id`, and `run.json` pins
the config, the git commit, the graph artifact and the environment.

*Runs produced one number.* With `eval_every = 0` a five-epoch run reported a
single score at the end, which is a poor trade when an epoch costs ~23 s: there
was no way to see whether the model had converged, which epoch was best, or which
benchmark an arm moved. `epochs.jsonl` records train means, test scores and
embedding geometry once per epoch, so a run answers those without being re-run.
"""

import json
import os
import platform
import subprocess
import time
import uuid
from typing import Any


def new_run_id() -> str:
    """Sortable and unique: a timestamp for reading, 6 hex for collisions."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                args, stderr=subprocess.DEVNULL, text=True, timeout=10
            ).strip()
        except Exception:
            return None

    sha = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "sha": sha,
        # A dirty tree means the code that ran is not the code the sha names, and
        # a result from it cannot be reproduced from the commit alone.
        "dirty": None if status is None else bool(status.strip()),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


def _config_snapshot(config: Any) -> dict[str, Any]:
    """Every public attribute, JSON-safe. Defaults included on purpose.

    Recording only the overrides would leave a run's meaning dependent on what the
    config class happened to contain that week.
    """
    snapshot: dict[str, Any] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name, None)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            snapshot[name] = value
        elif isinstance(value, (list, tuple)):
            snapshot[name] = [
                v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                for v in value
            ]
        else:
            snapshot[name] = str(value)
    return snapshot


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 1
            )
    except Exception:
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except Exception:
        pass
    return env


def write_run_manifest(
    save_dir: str,
    run_id: str,
    config: Any,
    artifact: dict | None = None,
    extra: dict | None = None,
) -> str | None:
    """Write `run.json` once, at the start of training."""
    if not save_dir:
        return None
    os.makedirs(save_dir, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _git_state(),
        "env": _environment(),
        "config": _config_snapshot(config),
    }
    if artifact is not None:
        # The graph is half the method: two runs with identical configs but
        # different graphs are different experiments, and only the fingerprint
        # and the build stats say which graph was used.
        manifest["artifact"] = {
            "metadata": artifact.get("metadata", {}),
            "graph_stats": {
                key: value
                for key, value in artifact.get("graph_stats", {}).items()
                if isinstance(value, (int, float, str, bool))
            },
        }
    if extra:
        manifest.update(extra)

    path = os.path.join(save_dir, "run.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True, default=str)
    print(f"Run manifest: {path} (run_id={run_id})")
    return path


def append_epoch_record(save_dir: str, run_id: str, record: dict[str, Any]) -> None:
    """Append one line to `epochs.jsonl`.

    Append mode is deliberate -- a resumed or re-run experiment should extend the
    history rather than erase it -- which is exactly why every line carries the
    run_id that separates them.
    """
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "epochs.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"run_id": run_id, **record}, sort_keys=True, default=float)
            + "\n"
        )


def load_epochs(save_dir: str, run_id: str | None = None) -> list[dict[str, Any]]:
    """Read `epochs.jsonl` back, optionally filtered to one run."""
    path = os.path.join(save_dir, "epochs.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if run_id is None or row.get("run_id") == run_id:
                rows.append(row)
    return rows
