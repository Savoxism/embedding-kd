"""The fixes behind the 2026-09-12 sweep's broken tables.

* student_knn trained on the teacher graph, rebuilt under the student's name;
* pair_order read exactly 1.0 or 0.0 on a rectangular, sparse held-out mask;
* Stage 1.1b and 1.3 had no code;
* the runner overwrote multi-pair CSVs and could not re-run a single arm.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from config import GGPKDConfig
from src.criterions.ggpkd_distillation import GGPKDDistillation
from src.distill.geometry import pair_order_accuracy
from src.ggpkd.graph_builder import build_or_load_ggpkd_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"


def _inputs(batch=4, candidates=12, dim=16, n_items=60):
    torch.manual_seed(0)
    probs = torch.rand(batch, 1, candidates)
    probs[:, :, candidates // 2 :] = 0.0
    probs /= probs.sum(-1, keepdim=True)
    candidate_idx = torch.stack(
        [torch.randperm(n_items)[:candidates] for _ in range(batch)]
    )
    for row in range(batch):
        candidate_idx[row][candidate_idx[row] == row] = (row + 31) % n_items
    return {
        "teacher": torch.randn(n_items, dim),
        "anchor": torch.randn(batch, dim, requires_grad=True),
        "candidates": torch.randn(batch * candidates, dim, requires_grad=True),
        "probs": probs,
        "candidate_idx": candidate_idx,
        "anchor_idx": torch.arange(batch),
        "graph": {
            "transition_neighbors": torch.randint(0, n_items, (n_items, 8)).int(),
            "transition_probs": torch.rand(n_items, 8).softmax(-1),
            "row_temps": torch.full((n_items,), 0.05),
        },
    }


def _forward(criterion, data):
    criterion.use_row_loss = True
    return criterion(
        data["anchor"],
        data["candidates"],
        data["probs"],
        candidate_idx=data["candidate_idx"],
        anchor_idx=data["anchor_idx"],
    )


# --------------------------------------------------------------------------- #
# Stage 1.3 -- uniform target
# --------------------------------------------------------------------------- #


def test_uniform_target_changes_the_values_not_the_columns():
    data = _inputs()
    metrics = {}
    for target in ("transition", "uniform"):
        criterion = GGPKDDistillation(
            teacher_embeddings=data["teacher"],
            relation_target=target,
            row_weight=1.0,
            **data["graph"],
        )
        loss, metrics[target] = _forward(criterion, data)
        assert torch.isfinite(loss)
    assert metrics["uniform"]["candidates_per_anchor"] == pytest.approx(
        metrics["transition"]["candidates_per_anchor"]
    )
    assert metrics["uniform"]["loss_amb"] == pytest.approx(
        metrics["transition"]["loss_amb"]
    )
    assert metrics["uniform"]["loss_nbr"] != pytest.approx(
        metrics["transition"]["loss_nbr"]
    )


def test_uniform_target_is_differentiable():
    data = _inputs()
    criterion = GGPKDDistillation(
        teacher_embeddings=data["teacher"],
        relation_target="uniform",
        row_weight=1.0,
        **data["graph"],
    )
    loss, _ = _forward(criterion, data)
    loss.backward()
    assert torch.isfinite(data["anchor"].grad).all()


# --------------------------------------------------------------------------- #
# Stage 1.1b -- random row centers
# --------------------------------------------------------------------------- #


def test_random_row_centers_keep_each_rows_width_and_never_pick_the_row_itself():
    data = _inputs()
    criterion = GGPKDDistillation(
        teacher_embeddings=data["teacher"],
        row_weight=1.0,
        row_centers="random",
        **data["graph"],
    )
    allowed = torch.zeros(3, 10, dtype=torch.bool)
    allowed[0, [1, 2]] = True
    allowed[1, [0, 3, 4, 5]] = True
    allowed[2, [7, 8, 9]] = True
    source_positions = torch.tensor([0, 1, 2])
    torch.manual_seed(1)
    drawn, target = criterion._random_row_columns(
        source_positions,
        torch.tensor([5, 6, 7]),
        torch.arange(20, 30),
        allowed,
        torch.full((3, 1), 0.05),
    )
    assert drawn.sum(dim=1).tolist() == [2, 4, 3]
    assert not drawn[torch.arange(3), source_positions].any()
    torch.testing.assert_close(target.sum(dim=1), torch.ones(3))
    assert bool((target[~drawn] == 0).all())


def test_random_row_centers_need_the_teacher_bank():
    data = _inputs()
    with pytest.raises(ValueError, match="teacher bank"):
        GGPKDDistillation(
            teacher_embeddings=None,
            row_weight=1.0,
            row_centers="random",
            calibration_mode="none",
            **data["graph"],
        )


def test_random_row_centers_move_only_the_row_term():
    data = _inputs()
    metrics = {}
    for centers in ("teacher", "random"):
        criterion = GGPKDDistillation(
            teacher_embeddings=data["teacher"],
            row_weight=1.0,
            row_centers=centers,
            **data["graph"],
        )
        torch.manual_seed(2)
        loss, metrics[centers] = _forward(criterion, data)
        assert torch.isfinite(loss)
    assert metrics["random"]["row_count"] > 0
    assert metrics["random"]["loss_nbr"] == pytest.approx(
        metrics["teacher"]["loss_nbr"]
    )
    assert metrics["random"]["loss_row"] != pytest.approx(
        metrics["teacher"]["loss_row"]
    )


# --------------------------------------------------------------------------- #
# Config guards
# --------------------------------------------------------------------------- #


def test_student_neighbours_require_teacher_targets_and_no_row_term():
    GGPKDConfig(
        neighbor_source="student",
        relation_target="direct",
        calibration_mode="none",
        row_weight=0.0,
    )
    with pytest.raises(ValueError, match="relation_target='direct'"):
        GGPKDConfig(neighbor_source="student", row_weight=0.0)
    with pytest.raises(ValueError, match="row_weight=0"):
        GGPKDConfig(neighbor_source="student", relation_target="direct")


def test_random_row_centers_are_refused_without_the_row_term():
    with pytest.raises(ValueError, match="row_weight > 0"):
        GGPKDConfig(row_centers="random", row_weight=0.0)


# --------------------------------------------------------------------------- #
# Stage 2B -- student_knn graph
# --------------------------------------------------------------------------- #


def _sorted_neighbours(artifact):
    return artifact["transition_neighbors"].sort(dim=1).values


def test_student_graph_takes_columns_from_the_student_and_temperatures_from_the_teacher(
    tmp_path,
):
    torch.manual_seed(0)
    teacher = torch.randn(80, 16)
    student = torch.randn(80, 12)
    common = {"log_dir": str(tmp_path / "logs"), "graph_k": 8, "knn_mode": "directed"}
    teacher_graph = build_or_load_ggpkd_artifact(
        teacher, cache_path=str(tmp_path / "teacher.pt"), **common
    )
    student_graph = build_or_load_ggpkd_artifact(
        teacher,
        cache_path=str(tmp_path / "student.pt"),
        neighbor_source="student:test",
        neighbor_embeddings=lambda: student,
        **common,
    )
    normalized = F.normalize(student, dim=-1)
    cosine = normalized @ normalized.t()
    cosine.fill_diagonal_(-float("inf"))
    expected = cosine.topk(8, dim=1).indices.sort(dim=1).values
    assert torch.equal(_sorted_neighbours(student_graph), expected)
    assert not torch.equal(
        _sorted_neighbours(student_graph), _sorted_neighbours(teacher_graph)
    )
    torch.testing.assert_close(student_graph["row_temps"], teacher_graph["row_temps"])


def test_a_teacher_graph_never_loads_under_the_student_name(tmp_path):
    """The 2026-09-12 failure: a teacher artifact sat at the student arm's path."""
    torch.manual_seed(0)
    teacher = torch.randn(80, 16)
    student = torch.randn(80, 12)
    path = tmp_path / "graph_student_knn.pt"
    common = {"log_dir": str(tmp_path / "logs"), "graph_k": 8, "knn_mode": "directed"}
    teacher_graph = build_or_load_ggpkd_artifact(teacher, cache_path=str(path), **common)
    # An artifact written before the key existed carries no neighbour source.
    stale = torch.load(path, weights_only=False)
    stale["metadata"].pop("neighbor_source")
    torch.save(stale, path)

    calls = []

    def encode():
        calls.append(1)
        return student

    rebuilt = build_or_load_ggpkd_artifact(
        teacher,
        cache_path=str(path),
        neighbor_source="student:test",
        neighbor_embeddings=encode,
        **common,
    )
    assert calls == [1]
    assert rebuilt["metadata"]["neighbor_source"] == "student:test"
    assert not torch.equal(_sorted_neighbours(rebuilt), _sorted_neighbours(teacher_graph))

    # A matching artifact loads without encoding the student again.
    build_or_load_ggpkd_artifact(
        teacher,
        cache_path=str(path),
        neighbor_source="student:test",
        neighbor_embeddings=encode,
        **common,
    )
    assert calls == [1]


# --------------------------------------------------------------------------- #
# Stage 2E -- pair_order on [anchors, corpus]
# --------------------------------------------------------------------------- #


def _rectangular(seed):
    torch.manual_seed(seed)
    corpus = F.normalize(torch.randn(2000, 16), dim=-1)
    anchors = torch.arange(0, 2000, 40)
    mask = torch.zeros(len(anchors), 2000, dtype=torch.bool)
    for row in range(len(anchors)):
        # Almost every admissible column lies past the first 50, which is all the
        # old sampler could reach.
        mask[row, torch.randperm(2000)[:12]] = True
    return corpus, anchors, mask


def test_pair_order_reads_a_sparse_mask_on_a_rectangular_matrix():
    corpus, anchors, mask = _rectangular(0)
    teacher_rows = corpus[anchors] @ corpus.t()
    assert (
        pair_order_accuracy(teacher_rows, teacher_rows, restrict=mask, n_triplets=20000)
        == 1.0
    )
    other = F.normalize(torch.randn(2000, 16), dim=-1)
    student_rows = other[anchors] @ other.t()
    accuracy = pair_order_accuracy(
        teacher_rows, student_rows, restrict=mask, n_triplets=20000
    )
    assert 0.4 < accuracy < 0.6


def test_pair_order_scores_only_the_masked_columns_of_each_row():
    corpus, anchors, mask = _rectangular(1)
    teacher_rows = corpus[anchors] @ corpus.t()
    corrupted = teacher_rows + torch.randn_like(teacher_rows) * 5.0
    student_rows = torch.where(mask, teacher_rows, corrupted)
    assert (
        pair_order_accuracy(teacher_rows, student_rows, restrict=mask, n_triplets=20000)
        == 1.0
    )


# --------------------------------------------------------------------------- #
# Runner and export
# --------------------------------------------------------------------------- #


def test_export_merges_rows_by_pair_arm_and_seed(tmp_path):
    root = tmp_path / "run"
    (root / "runs" / "student_knn" / "seed_42").mkdir(parents=True)
    (root / "run_config.tsv").write_text("commit\tabc123\n", encoding="utf-8")
    out = tmp_path / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "pair", "arm", "seed", "status"])
        writer.writeheader()
        for arm in ("student_knn", "teacher"):
            writer.writerow(
                {"experiment": "stage2b_ladder", "pair": "p", "arm": arm, "seed": "42", "status": "ok"}
            )
    subprocess.run(
        [
            sys.executable,
            "scripts/exp/export_runs.py",
            str(root),
            "--experiment",
            "stage2b_ladder",
            "--pair",
            "p",
            "--out",
            str(out),
            "--merge",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert sorted(row["arm"] for row in rows) == ["student_knn", "teacher"]
    fresh = next(row for row in rows if row["arm"] == "student_knn")
    assert fresh["status"] == "no_manifest"
    assert fresh["git_commit"] == "abc123"


def _run_arms(arms):
    corpus = "data/train_set/merged_3_data_5k_each.csv"
    command = f"""
source {REPO_ROOT / 'scripts/exp/lib/run_arms.sh'}
GRAPH_SPEC='main|{corpus}|
other|{corpus}|'
ARMS_SPEC='a|main|ggpkd|
b|other|ggpkd|'
PAIR=qwen3_0_6b_to_minilmv2_h384
SEEDS=42,43
GPUS=0
DRY_RUN=1
ARMS={arms}
PYTHON_BIN={PYTHON_BIN}
run_arms {REPO_ROOT} runner_validation
"""
    return subprocess.run(
        ["bash", "-c", command], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def test_runner_runs_only_the_named_arms_and_their_graphs():
    result = _run_arms("b")
    assert result.returncode == 0, result.stderr
    assert "runs:   2" in result.stdout
    assert "graphs: 1" in result.stdout
    assert "graph=other" in result.stdout and "graph=main" not in result.stdout


def test_runner_rejects_an_arm_the_sweep_does_not_define():
    result = _run_arms("zzz")
    assert result.returncode == 2
    assert "does not define" in result.stderr


@pytest.mark.parametrize(
    ("script", "extra_env", "expected", "absent"),
    [
        (
            "stage1_deletions.sh",
            {},
            ["runs:   12", "row_centers_random", "uniform_target"],
            ["skipping arm"],
        ),
        (
            "stage2b_ladder.sh",
            {},
            ["runs:   12", "graph=student_knn"],
            ["qwen3_4b_to_bert_base"],
        ),
        (
            "stage2c_dose_response.sh",
            {"HALF": "c2"},
            ["runs:   12", "--batch_size 256 --epochs 20", "--batch_size 1024 --epochs 80"],
            [],
        ),
        (
            "stage2d_components.sh",
            {},
            ["runs:   12", "full", "graph=main_holdout"],
            [],
        ),
        ("stage3_main_table.sh", {}, ["pairs run here: none"], ["main table -- pair"]),
    ],
)
def test_stage_dry_runs_plan_the_fixed_matrix(script, extra_env, expected, absent):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "GPUS": "0",
        "GRAPH_K": "200",
        "PYTHON_BIN": str(PYTHON_BIN),
        "RUN_ID": "stage-fix-test",
        **extra_env,
    }
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "exp" / script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for text in expected:
        assert text in output
    for text in absent:
        assert text not in output
