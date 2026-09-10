import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"


@pytest.mark.parametrize(
    ("script", "expected_runs"),
    [
        ("exp1_batch_intervention.sh", 27),
        ("exp2_support_intervention.sh", 15),
        ("exp4_exposure_scaling.sh", 84),
    ],
)
def test_experiment_dry_run_matrix(script, expected_runs):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "GPUS": "0,1,2,3",
        "PYTHON_BIN": str(PYTHON_BIN),
        "RUN_ID": "runner-test",
    }
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "exp" / script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"runs:   {expected_runs}" in result.stdout


def test_runner_rejects_an_unknown_graph_key():
    command = f"""
source {REPO_ROOT / 'scripts/exp/lib/run_arms.sh'}
GRAPH_SPEC='main|data/train_set/merged_3_data_5k_each.csv|'
ARMS_SPEC='bad|missing|ggpkd|'
PAIR=qwen3_0_6b_to_minilmv2_h384
SEEDS=42
GPUS=0
DRY_RUN=1
PYTHON_BIN={PYTHON_BIN}
run_arms {REPO_ROOT} runner_validation
"""
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "references unknown graph_key" in result.stderr
