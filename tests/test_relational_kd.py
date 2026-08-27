"""Checks that the RKD baseline reproduces Park et al. (CVPR 2019) verbatim.

Each test pins one equation of the paper against a naive reference written
straight from the formula, so a refactor that quietly changes the objective
fails here rather than in a training run.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from main import get_config, parse_args
from src.criterions.relational_kd import RelationalKD, RKdAngle, RKdDistance, pdist


def huber(x: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Eq. 5: l_delta(x, y)."""
    difference = (x - y).abs()
    return torch.where(
        difference <= delta,
        0.5 * difference.pow(2) / delta,
        difference - 0.5 * delta,
    )


def reference_distance_loss(student, teacher):
    """Eq. 3-4, written directly from the paper."""
    teacher_distance = torch.cdist(teacher, teacher, p=2)
    teacher_distance = teacher_distance / teacher_distance[teacher_distance > 0].mean()

    student_distance = torch.cdist(student, student, p=2)
    student_distance = student_distance / student_distance[student_distance > 0].mean()

    return huber(student_distance, teacher_distance).mean()


def reference_angle_loss(student, teacher):
    """Eq. 6-7, enumerated one triplet at a time."""

    def potential(embeddings, i, j, k):
        e_ij = F.normalize(embeddings[i] - embeddings[j], p=2, dim=0)
        e_kj = F.normalize(embeddings[k] - embeddings[j], p=2, dim=0)
        return torch.dot(e_ij, e_kj)

    n = student.size(0)
    terms = [
        huber(potential(student, i, j, k), potential(teacher, i, j, k))
        for i in range(n)
        for j in range(n)
        for k in range(n)
    ]
    return torch.stack(terms).mean()


@pytest.fixture
def embeddings():
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(6, 8, generator=generator)
    teacher = torch.randn(6, 16, generator=generator) * 3.0
    return student, teacher


def test_pdist_matches_euclidean_distance_matrix(embeddings):
    student, _ = embeddings

    assert torch.allclose(
        pdist(student, squared=False), torch.cdist(student, student, p=2), atol=1e-5
    )
    assert torch.allclose(
        pdist(student).diagonal(), torch.zeros(student.size(0)), atol=0
    )


def test_distance_loss_matches_paper_equation(embeddings):
    student, teacher = embeddings

    assert RKdDistance()(student, teacher).item() == pytest.approx(
        reference_distance_loss(student, teacher).item(), abs=1e-5
    )


def test_angle_loss_matches_paper_equation(embeddings):
    student, teacher = embeddings

    assert RKdAngle()(student, teacher).item() == pytest.approx(
        reference_angle_loss(student, teacher).item(), abs=1e-5
    )


def test_both_potentials_vanish_when_relations_already_agree(embeddings):
    _, teacher = embeddings

    assert RKdDistance()(teacher.clone(), teacher).item() == pytest.approx(0.0, abs=1e-6)
    assert RKdAngle()(teacher.clone(), teacher).item() == pytest.approx(0.0, abs=1e-6)


def test_distance_potential_is_scale_invariant(embeddings):
    """The mu normalisation of Eq. 3 is what makes the loss scale-free."""
    student, teacher = embeddings

    baseline = RKdDistance()(student, teacher)
    rescaled = RKdDistance()(student * 17.0, teacher)

    assert rescaled.item() == pytest.approx(baseline.item(), abs=1e-5)


def test_criterion_needs_no_projection_between_mismatched_dimensions(embeddings):
    student, teacher = embeddings
    criterion = RelationalKD()

    assert student.size(-1) != teacher.size(-1)
    assert list(criterion.parameters()) == []

    loss, _ = criterion(student, teacher, task_loss=torch.zeros(()))
    assert torch.isfinite(loss)


def test_total_loss_is_the_paper_weighted_sum(embeddings):
    student, teacher = embeddings
    task_loss = torch.tensor(0.75)
    criterion = RelationalKD(w_task=1.0, dist_ratio=1.0, angle_ratio=2.0)

    loss, metrics = criterion(student, teacher, task_loss=task_loss)

    expected = (
        1.0 * task_loss.item()
        + 1.0 * reference_distance_loss(student, teacher).item()
        + 2.0 * reference_angle_loss(student, teacher).item()
    )
    assert loss.item() == pytest.approx(expected, abs=1e-5)
    assert metrics["loss_task"] == pytest.approx(task_loss.item())
    assert metrics["loss_rkd_d"] == pytest.approx(
        reference_distance_loss(student, teacher).item(), abs=1e-5
    )
    assert metrics["loss_rkd_a"] == pytest.approx(
        reference_angle_loss(student, teacher).item(), abs=1e-5
    )


@pytest.mark.parametrize(
    ("dist_ratio", "angle_ratio", "dropped_metric"),
    [(1.0, 0.0, "loss_rkd_a"), (0.0, 1.0, "loss_rkd_d")],
)
def test_single_potential_variants(embeddings, dist_ratio, angle_ratio, dropped_metric):
    """RKD-D and RKD-A are the ratio=0 corners of the combined objective."""
    student, teacher = embeddings
    criterion = RelationalKD(
        w_task=0.0, dist_ratio=dist_ratio, angle_ratio=angle_ratio
    )

    _, metrics = criterion(student, teacher, task_loss=torch.zeros(()))
    assert metrics[dropped_metric] == 0.0


def test_gradients_reach_the_student_only(embeddings):
    student, teacher = embeddings
    student = student.clone().requires_grad_(True)
    teacher = teacher.clone().requires_grad_(True)

    loss, _ = RelationalKD()(student, teacher, task_loss=torch.zeros(()))
    loss.backward()

    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_cli_defaults_follow_the_paper(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--method", "rkd"])
    config = get_config(parse_args().method, parse_args())

    assert config.distill_method == "rkd"
    assert config.dist_ratio == 1.0
    assert config.angle_ratio == 2.0
    assert config.huber_delta == 1.0


def test_cli_overrides_the_loss_ratios(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "rkd", "--dist_ratio", "1", "--angle_ratio", "0"],
    )
    args = parse_args()
    config = get_config(args.method, args)

    assert (config.dist_ratio, config.angle_ratio) == (1.0, 0.0)
