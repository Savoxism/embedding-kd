"""Relational Knowledge Distillation losses.

This module follows the public RKD implementation from Park et al. (CVPR 2019):
pairwise distances are normalized by their non-zero batch mean and both the
distance-wise and angle-wise relations are matched with Smooth L1 loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def pairwise_distance(
    embeddings: torch.Tensor,
    *,
    squared: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return the full pairwise Euclidean-distance matrix used by RKD."""
    if embeddings.ndim != 2:
        raise ValueError(
            f"RKD embeddings must have shape [batch, dim], got {tuple(embeddings.shape)}"
        )

    embeddings = embeddings.float()
    squared_norm = embeddings.pow(2).sum(dim=1)
    distances = (
        squared_norm.unsqueeze(1)
        + squared_norm.unsqueeze(0)
        - 2.0 * embeddings @ embeddings.t()
    ).clamp_min(eps)
    if not squared:
        distances = distances.sqrt()

    distances = distances.clone()
    diagonal = torch.arange(embeddings.shape[0], device=embeddings.device)
    distances[diagonal, diagonal] = 0.0
    return distances


class RKDDistanceLoss(nn.Module):
    """Second-order RKD loss over normalized pairwise distances."""

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if student.shape[0] != teacher.shape[0]:
            raise ValueError(
                "RKD student and teacher batches must have the same size, got "
                f"{student.shape[0]} and {teacher.shape[0]}"
            )
        if student.shape[0] < 2:
            raise ValueError("RKD distance loss requires a batch size of at least 2")

        with torch.no_grad():
            teacher_distances = pairwise_distance(teacher, eps=self.eps)
            teacher_mean = teacher_distances[teacher_distances > 0].mean()
            teacher_relations = teacher_distances / teacher_mean.clamp_min(self.eps)

        student_distances = pairwise_distance(student, eps=self.eps)
        student_mean = student_distances[student_distances > 0].mean()
        student_relations = student_distances / student_mean.clamp_min(self.eps)

        loss = F.smooth_l1_loss(student_relations, teacher_relations)
        metrics = {
            "teacher_mean_distance": float(teacher_mean.detach()),
            "student_mean_distance": float(student_mean.detach()),
        }
        return loss, metrics


class RKDAngleLoss(nn.Module):
    """Third-order RKD loss over angles formed by embedding triplets."""

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def _relations(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.float()
        differences = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
        directions = F.normalize(differences, p=2, dim=2, eps=self.eps)
        return torch.bmm(directions, directions.transpose(1, 2)).reshape(-1)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.shape[0] != teacher.shape[0]:
            raise ValueError(
                "RKD student and teacher batches must have the same size, got "
                f"{student.shape[0]} and {teacher.shape[0]}"
            )
        if student.shape[0] < 2:
            raise ValueError("RKD angle loss requires a batch size of at least 2")

        with torch.no_grad():
            teacher_relations = self._relations(teacher)
        student_relations = self._relations(student)
        return F.smooth_l1_loss(student_relations, teacher_relations)


class RelationalKnowledgeDistillation(nn.Module):
    """Paper-default RKD-DA objective: one distance loss plus two angle losses."""

    def __init__(
        self,
        *,
        distance_weight: float = 1.0,
        angle_weight: float = 2.0,
        task_weight: float = 0.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        if distance_weight < 0 or angle_weight < 0 or task_weight < 0:
            raise ValueError("RKD loss weights must be non-negative")
        if distance_weight == 0 and angle_weight == 0 and task_weight == 0:
            raise ValueError("At least one RKD loss weight must be positive")

        self.distance_weight = float(distance_weight)
        self.angle_weight = float(angle_weight)
        self.task_weight = float(task_weight)
        self.distance_loss = RKDDistanceLoss(eps=eps)
        self.angle_loss = RKDAngleLoss(eps=eps)

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        task_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = student.float().sum() * 0.0
        distance = zero
        angle = zero
        distance_metrics: dict[str, float] = {}

        if self.distance_weight > 0:
            distance, distance_metrics = self.distance_loss(student, teacher)
        if self.angle_weight > 0:
            angle = self.angle_loss(student, teacher)

        if self.task_weight > 0:
            if task_loss is None:
                raise ValueError("RKD task_weight > 0 requires task_loss")
            task = task_loss.float()
        else:
            task = zero

        total = (
            self.distance_weight * distance
            + self.angle_weight * angle
            + self.task_weight * task
        )
        metrics = {
            "loss_total": float(total.detach()),
            "loss_task": float(task.detach()),
            "loss_rkd_distance": float(distance.detach()),
            "loss_rkd_angle": float(angle.detach()),
            **distance_metrics,
        }
        return total, metrics
