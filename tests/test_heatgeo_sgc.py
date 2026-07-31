import torch
import torch.nn.functional as F

from src.criterions.heatgeo_distillation import HeatGeoDistillation


def _criterion(weight: float) -> HeatGeoDistillation:
    teacher_embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        dim=-1,
    )
    return HeatGeoDistillation(
        student_dim=3,
        teacher_dim=3,
        scale_weights=(1.0,),
        scale_temps=(0.07,),
        share_in_batch=False,
        teacher_embeddings=teacher_embeddings,
        direct_weight=1.0,
        sgc_weight=weight,
        sgc_huber_delta=0.10,
    )


def test_sgc_is_zero_for_matching_origin_and_detects_row_shift():
    criterion = _criterion(weight=0.05)
    teacher = torch.tensor([[0.7, 0.4, 0.1], [0.6, 0.2, -0.1]])
    weights = torch.tensor([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]])

    matching_loss, matching_stats = criterion._sgc_loss(teacher, teacher, weights)
    assert matching_loss.item() == 0.0
    assert matching_stats[0].item() == 0.0

    student = (teacher + torch.tensor([[0.1], [-0.05]])).requires_grad_(True)
    assert torch.allclose(
        F.softmax(student / 0.07, dim=-1),
        F.softmax(teacher / 0.07, dim=-1),
        atol=1e-6,
    )
    shifted_loss, shifted_stats = criterion._sgc_loss(student, teacher, weights)
    shifted_loss.backward()

    assert shifted_loss.item() > 0.0
    assert torch.isclose(shifted_stats[0], torch.tensor(0.075))
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_forward_adds_weighted_sgc_and_weight_zero_is_compatible():
    torch.manual_seed(5)
    anchors = torch.randn(2, 3, requires_grad=True)
    candidates = torch.randn(6, 3)
    teacher_probs = torch.rand(2, 1, 3)
    anchor_idx = torch.tensor([0, 1])
    candidate_idx = torch.tensor([[1, 2, 3], [0, 2, 3]])

    criterion = _criterion(weight=0.05)
    total, metrics = criterion(
        anchor_embeddings=anchors,
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
        candidate_idx=candidate_idx,
        anchor_idx=anchor_idx,
    )
    expected = metrics["loss_diff"] + 0.05 * metrics["loss_sgc"]
    assert abs(metrics["loss_total"] - expected) < 1e-6
    total.backward()
    assert torch.isfinite(anchors.grad).all()

    baseline = _criterion(weight=0.0)
    baseline_total, baseline_metrics = baseline(
        anchor_embeddings=anchors.detach(),
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
        candidate_idx=candidate_idx,
        anchor_idx=anchor_idx,
    )
    assert baseline_metrics["loss_sgc"] == 0.0
    assert torch.isclose(
        baseline_total, torch.tensor(baseline_metrics["loss_diff"])
    )
