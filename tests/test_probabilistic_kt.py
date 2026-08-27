"""Checks that the PKT baseline reproduces Passalis & Tefas verbatim.

Each test pins one equation against a naive reference written straight from the
paper, and one test pins the released-code variant, so the two conventions stay
distinguishable.
"""

import sys

import pytest
import torch

from main import get_config, parse_args
from src.criterions.probabilistic_kt import (
    ProbabilisticKT,
    cosine_kernel,
    gaussian_kernel,
)


def reference_conditional(embeddings, exclude_self=True, eps=1e-7):
    """Eq. 3 / Eq. 4, one entry at a time."""
    n = embeddings.size(0)
    rows = []
    for j in range(n):
        kernels = []
        for i in range(n):
            if exclude_self and i == j:
                kernels.append(torch.zeros(()))
                continue
            a, b = embeddings[i], embeddings[j]
            kernels.append(
                0.5 * (torch.dot(a, b) / (a.norm() * b.norm()) + 1.0)
            )
        kernels = torch.stack(kernels)
        rows.append(kernels / (kernels.sum() + eps))
    # Row j of the reference is the distribution conditioned on j, which is how
    # the module lays it out too.
    return torch.stack(rows)


def reference_loss(student, teacher, exclude_self=True, eps=1e-7):
    """Eq. 8: sum over i of the KL from p(.|i) to q(.|i)."""
    p = reference_conditional(teacher, exclude_self, eps)
    q = reference_conditional(student, exclude_self, eps)
    return (p * torch.log((p + eps) / (q + eps))).sum()


@pytest.fixture
def embeddings():
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(6, 8, generator=generator)
    teacher = torch.randn(6, 16, generator=generator) * 3.0
    return student, teacher


def test_cosine_kernel_matches_equation_six(embeddings):
    student, _ = embeddings
    kernel = cosine_kernel(student)

    normalized = student / student.norm(dim=1, keepdim=True)
    expected = (normalized @ normalized.t() + 1.0) / 2.0

    assert torch.allclose(kernel, expected, atol=1e-5)
    assert kernel.min() >= 0.0 and kernel.max() <= 1.0 + 1e-6
    assert torch.allclose(kernel.diagonal(), torch.ones(student.size(0)), atol=1e-5)


def test_gaussian_kernel_matches_equation_five(embeddings):
    student, _ = embeddings
    sigma = 2.0
    kernel = gaussian_kernel(student, sigma=sigma)

    expected = torch.exp(-torch.cdist(student, student, p=2).pow(2) / sigma)
    assert torch.allclose(kernel, expected, atol=1e-5)


def test_conditional_excludes_self_and_sums_to_one(embeddings):
    student, _ = embeddings
    criterion = ProbabilisticKT(exclude_self=True)

    conditional = criterion.conditional(student)

    assert torch.allclose(conditional, reference_conditional(student), atol=1e-5)
    assert torch.allclose(
        conditional.diagonal(), torch.zeros(student.size(0)), atol=0
    )
    assert torch.allclose(
        conditional.sum(dim=1), torch.ones(student.size(0)), atol=1e-5
    )


def test_divergence_matches_equation_eight(embeddings):
    student, teacher = embeddings
    criterion = ProbabilisticKT(exclude_self=True, reduction="sum")

    assert criterion.divergence(student, teacher).item() == pytest.approx(
        reference_loss(student, teacher).item(), abs=1e-5
    )


def test_reductions_differ_only_by_batch_size(embeddings):
    student, teacher = embeddings
    n = student.size(0)

    def divergence(reduction):
        return ProbabilisticKT(reduction=reduction).divergence(student, teacher).item()

    total = divergence("sum")
    assert divergence("batchmean") == pytest.approx(total / n, abs=1e-6)
    assert divergence("mean") == pytest.approx(total / n**2, abs=1e-6)


def test_released_code_variant_keeps_the_self_term(embeddings):
    """The authors' nn/pkt.py normalises over the full row, diagonal included."""
    student, teacher = embeddings
    criterion = ProbabilisticKT(exclude_self=False, reduction="mean")

    eps = 1e-7

    def released_code_conditional(x):
        x = x / (torch.sqrt(torch.sum(x**2, dim=1, keepdim=True)) + eps)
        similarity = (x @ x.t() + 1.0) / 2.0
        return similarity / torch.sum(similarity, dim=1, keepdim=True)

    p = released_code_conditional(teacher)
    q = released_code_conditional(student)
    expected = torch.mean(p * torch.log((p + eps) / (q + eps)))

    assert criterion.divergence(student, teacher).item() == pytest.approx(
        expected.item(), abs=1e-6
    )
    assert criterion.conditional(student).diagonal().min() > 0.0


def test_divergence_vanishes_when_affinities_already_agree(embeddings):
    _, teacher = embeddings

    assert ProbabilisticKT().divergence(teacher.clone(), teacher).item() == (
        pytest.approx(0.0, abs=1e-6)
    )


def test_divergence_is_scale_invariant_under_the_cosine_kernel(embeddings):
    student, teacher = embeddings
    criterion = ProbabilisticKT()

    baseline = criterion.divergence(student, teacher).item()
    assert criterion.divergence(student * 17.0, teacher).item() == pytest.approx(
        baseline, abs=1e-5
    )


def test_criterion_needs_no_projection_between_mismatched_dimensions(embeddings):
    student, teacher = embeddings
    criterion = ProbabilisticKT()

    assert student.size(-1) != teacher.size(-1)
    assert list(criterion.parameters()) == []

    loss, _ = criterion(student, teacher, task_loss=torch.zeros(()))
    assert torch.isfinite(loss)


def test_total_loss_is_the_weighted_sum(embeddings):
    student, teacher = embeddings
    task_loss = torch.tensor(0.75)
    criterion = ProbabilisticKT(w_task=0.5, w_pkt=2.0)

    loss, metrics = criterion(student, teacher, task_loss=task_loss)

    expected = 0.5 * task_loss.item() + 2.0 * criterion.divergence(
        student.float(), teacher.float()
    ).item()
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert metrics["loss_task"] == pytest.approx(task_loss.item())


def test_gradients_reach_the_student_only(embeddings):
    student, teacher = embeddings
    student = student.clone().requires_grad_(True)
    teacher = teacher.clone().requires_grad_(True)

    loss, _ = ProbabilisticKT()(student, teacher, task_loss=torch.zeros(()))
    loss.backward()

    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


@pytest.mark.parametrize("bad", [{"kernel": "laplacian"}, {"reduction": "none"}])
def test_unsupported_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        ProbabilisticKT(**bad)


def test_cli_defaults_follow_the_paper(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--method", "pkt"])
    args = parse_args()
    config = get_config(args.method, args)

    assert config.distill_method == "pkt"
    assert config.kernel == "cosine"
    assert config.exclude_self is True
    assert config.w_task == 0.0


def test_cli_overrides_pkt_settings(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "pkt", "--w_pkt", "30000", "--pkt_kernel", "gaussian"],
    )
    args = parse_args()
    config = get_config(args.method, args)

    assert config.w_pkt == 30000.0
    assert config.kernel == "gaussian"
