from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler

from distiller import KnowledgeDistiller
from src.criterions.relational_kd import (
    RKDAngleLoss,
    RKDDistanceLoss,
    RelationalKnowledgeDistillation,
    pairwise_distance,
)


def _official_pdist(embeddings, squared=False, eps=1e-12):
    squared_norm = embeddings.pow(2).sum(dim=1)
    distances = (
        squared_norm.unsqueeze(1)
        + squared_norm.unsqueeze(0)
        - 2 * embeddings @ embeddings.t()
    ).clamp(min=eps)
    if not squared:
        distances = distances.sqrt()
    distances = distances.clone()
    distances[range(len(embeddings)), range(len(embeddings))] = 0
    return distances


def _official_distance(student, teacher):
    teacher_distances = _official_pdist(teacher)
    teacher_distances = teacher_distances / teacher_distances[teacher_distances > 0].mean()
    student_distances = _official_pdist(student)
    student_distances = student_distances / student_distances[student_distances > 0].mean()
    return F.smooth_l1_loss(student_distances, teacher_distances)


def _official_angle(student, teacher):
    teacher_differences = teacher.unsqueeze(0) - teacher.unsqueeze(1)
    teacher_directions = F.normalize(teacher_differences, p=2, dim=2)
    teacher_angles = torch.bmm(
        teacher_directions, teacher_directions.transpose(1, 2)
    ).reshape(-1)

    student_differences = student.unsqueeze(0) - student.unsqueeze(1)
    student_directions = F.normalize(student_differences, p=2, dim=2)
    student_angles = torch.bmm(
        student_directions, student_directions.transpose(1, 2)
    ).reshape(-1)
    return F.smooth_l1_loss(student_angles, teacher_angles)


def test_rkd_losses_match_official_implementation():
    torch.manual_seed(7)
    student = torch.randn(8, 5, requires_grad=True)
    teacher = torch.randn(8, 11)

    distance, _ = RKDDistanceLoss()(student, teacher)
    angle = RKDAngleLoss()(student, teacher)

    torch.testing.assert_close(distance, _official_distance(student, teacher))
    torch.testing.assert_close(angle, _official_angle(student, teacher))


def test_rkd_default_objective_uses_one_to_two_weights():
    torch.manual_seed(11)
    student = torch.randn(6, 4, requires_grad=True)
    teacher = torch.randn(6, 9)
    criterion = RelationalKnowledgeDistillation()

    total, metrics = criterion(student, teacher)
    distance, _ = criterion.distance_loss(student, teacher)
    angle = criterion.angle_loss(student, teacher)

    torch.testing.assert_close(total, distance + 2.0 * angle)
    assert metrics["loss_task"] == 0.0


def test_rkd_is_zero_for_identical_embeddings_and_detaches_teacher():
    torch.manual_seed(13)
    teacher = torch.randn(7, 6, requires_grad=True)
    student = teacher.detach().clone().requires_grad_(True)

    loss, _ = RelationalKnowledgeDistillation()(student, teacher)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0)
    loss.backward()

    assert teacher.grad is None
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_rkd_relations_are_translation_and_scale_invariant():
    torch.manual_seed(17)
    teacher = torch.randn(6, 8)
    translated_scaled = teacher * 3.5 + 12.0

    distance, _ = RKDDistanceLoss()(translated_scaled, teacher)
    angle = RKDAngleLoss()(translated_scaled, teacher)

    torch.testing.assert_close(distance, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(angle, torch.tensor(0.0), atol=1e-6, rtol=0)


def test_pairwise_distance_and_rkd_stay_finite_with_duplicates():
    student = torch.tensor([[1.0, 2.0], [1.0, 2.0], [2.0, 4.0]], requires_grad=True)
    teacher = torch.tensor([[3.0, 1.0], [3.0, 1.0], [5.0, 2.0]])

    assert torch.isfinite(pairwise_distance(student)).all()
    loss, _ = RelationalKnowledgeDistillation()(student, teacher)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all()


class _TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 6)

    def forward(self, input_ids, attention_mask, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_rkd_train_step_updates_the_student(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = SimpleNamespace(
        distill_method="rkd",
        w_task=0.0,
        temperature=0.07,
    )
    distiller.device_s = torch.device("cpu")
    distiller.model_student = _TinyStudent()
    distiller.criterion = RelationalKnowledgeDistillation()
    distiller.optimizer = torch.optim.Adam(
        distiller.model_student.parameters(), lr=1e-3
    )
    distiller.scheduler = torch.optim.lr_scheduler.LambdaLR(
        distiller.optimizer, lambda _: 1.0
    )
    distiller.scaler = GradScaler("cuda", enabled=False)
    distiller.current_epoch = 0
    distiller.current_step = 0

    token_ids = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.long
    )
    batch = {
        "input_ids1_stu": token_ids,
        "attention_mask1_stu": torch.ones_like(token_ids),
        "input_ids2_stu": token_ids.clone(),
        "attention_mask2_stu": torch.ones_like(token_ids),
        "teacher_cls": torch.randn(4, 10),
    }
    before = distiller.model_student.embedding.weight.detach().clone()

    loss, metrics = distiller.train_step(batch)

    assert torch.isfinite(loss)
    assert metrics["loss_rkd_distance"] >= 0
    assert metrics["loss_rkd_angle"] >= 0
    assert not torch.equal(before, distiller.model_student.embedding.weight.detach())
