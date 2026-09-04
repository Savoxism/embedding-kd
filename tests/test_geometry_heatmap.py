from pathlib import Path

import numpy as np
import torch

from scripts.ablation.geometry_heatmap import latest_checkpoints, teacher_weighted_error
from scripts.ablation.geometry_heatmap import teacher_graph_order


def test_teacher_weighted_error_is_zero_for_identical_geometry():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
    )
    error, e_hat = teacher_weighted_error(embeddings, embeddings)

    assert error.shape == (3, 3)
    assert torch.count_nonzero(error) == 0
    assert e_hat == 0.0


def test_teacher_weighted_error_mean_matches_row_sum_definition():
    teacher = torch.eye(3)
    student = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    error, e_hat = teacher_weighted_error(teacher, student, temperature=0.2)

    assert torch.allclose(torch.diag(error), torch.zeros(3))
    assert np.isclose(e_hat, float(error.sum(dim=-1).mean()))
    assert e_hat > 0


def test_latest_checkpoints_selects_highest_epoch_per_seed(tmp_path: Path):
    for seed, epochs in {42: (1, 5), 43: (2, 4)}.items():
        directory = tmp_path / "full" / "full" / f"seed{seed}" / "weights"
        directory.mkdir(parents=True)
        for epoch in epochs:
            (directory / f"student_epoch_{epoch}.pt").touch()

    found = latest_checkpoints(
        tmp_path,
        "full/full/seed*/weights/student_epoch_*.pt",
        (42, 43),
    )

    assert found[42].name == "student_epoch_5.pt"
    assert found[43].name == "student_epoch_4.pt"


def test_teacher_graph_order_returns_probe_permutation(tmp_path: Path):
    artifact = {
        "transition_neighbors": torch.tensor(
            [[1, -1], [0, 2], [1, 3], [2, -1]], dtype=torch.long
        ),
        "transition_probs": torch.tensor(
            [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [1.0, 0.0]]
        ),
    }
    path = tmp_path / "graph.pt"
    torch.save(artifact, path)

    order = teacher_graph_order(path, np.arange(4), corpus_size=4)

    assert sorted(order.tolist()) == [0, 1, 2, 3]
