"""Relational Knowledge Distillation (Park et al., CVPR 2019).

Baseline reproduction of the paper's objective, unmodified:

    L_RKD = sum_{(x_1..x_n) in X^n} l( psi(t_1..t_n), psi(s_1..s_n) )   (Eq. 2)

with the paper's two relational potentials.

Distance-wise (Sec. 3.1)::

    psi_D(t_i, t_j) = (1 / mu) * || t_i - t_j ||_2 ,                     (Eq. 3)
    mu = (1 / |X^2|) * sum_{(x_i,x_j) in X^2} || t_i - t_j ||_2
    L_RKD-D = sum_{(x_i,x_j) in X^2} l_delta( psi_D(t_i,t_j),
                                              psi_D(s_i,s_j) )           (Eq. 4)

Angle-wise (Sec. 3.2)::

    psi_A(t_i,t_j,t_k) = <e_ij, e_kj>,
        e_ij = (t_i - t_j) / || t_i - t_j ||_2,
        e_kj = (t_k - t_j) / || t_k - t_j ||_2                           (Eq. 6)
    L_RKD-A = sum_{(x_i,x_j,x_k) in X^3} l_delta( psi_A(t_i,t_j,t_k),
                                                  psi_A(s_i,s_j,s_k) )   (Eq. 7)

Both use the Huber loss of Eq. 5 with delta = 1::

    l_delta(x, y) = 0.5 (x - y)^2            if |x - y| <= 1
                    |x - y| - 0.5            otherwise

which is exactly ``F.smooth_l1_loss`` with ``beta = 1``.

The tensor-level choices below -- the ``eps`` clamp inside ``pdist``, the zeroed
diagonal, ``mu`` taken over the non-zero pairwise distances, and angles
enumerated over the full N^3 grid -- follow the authors' released implementation
(https://github.com/lenscloth/RKD, ``metric/loss.py``), which is the reference
for the equations above.

The final objective is the paper's Sec. 3.3 combination::

    L = L_task + lambda_RKD-D * L_RKD-D + lambda_RKD-A * L_RKD-A

RKD compares only *relations* between examples, so the teacher and student
embeddings never have to share a dimensionality and no projection head is
learned. This module therefore holds no parameters.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist(e: torch.Tensor, squared: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """Pairwise Euclidean distance matrix of the rows of ``e``. [N, N]"""
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKdDistance(nn.Module):
    """Distance-wise distillation loss, Eq. 3-4."""

    def __init__(self, huber_delta: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.huber_delta = huber_delta
        self.eps = eps

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_d = pdist(teacher, squared=False, eps=self.eps)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False, eps=self.eps)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        return F.smooth_l1_loss(d, t_d, reduction="mean", beta=self.huber_delta)


class RKdAngle(nn.Module):
    """Angle-wise distillation loss, Eq. 6-7."""

    def __init__(self, huber_delta: float = 1.0):
        super().__init__()
        self.huber_delta = huber_delta

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        # N x C -> N x N x C -> N x N x N
        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        return F.smooth_l1_loss(
            s_angle, t_angle, reduction="mean", beta=self.huber_delta
        )


class RelationalKD(nn.Module):
    """L = w_task * L_task + dist_ratio * L_RKD-D + angle_ratio * L_RKD-A.

    ``dist_ratio`` / ``angle_ratio`` are the paper's lambda_RKD-D / lambda_RKD-A.
    Setting one of them to 0 recovers the paper's single-potential variants
    (RKD-D or RKD-A); both non-zero is RKD-DA.
    """

    def __init__(
        self,
        w_task: float = 1.0,
        dist_ratio: float = 1.0,
        angle_ratio: float = 2.0,
        huber_delta: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.w_task = w_task
        self.dist_ratio = dist_ratio
        self.angle_ratio = angle_ratio

        self.dist_criterion = RKdDistance(huber_delta=huber_delta, eps=eps)
        self.angle_criterion = RKdAngle(huber_delta=huber_delta)

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        task_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # The relations are second-order statistics of the embeddings, and the
        # teacher here runs in bfloat16 under autocast. Accumulating the squared
        # norms and the N^3 angle grid at that precision loses most of the signal
        # the Huber loss is meant to measure, so both potentials are evaluated in
        # float32. This is a precision choice only -- the objective is unchanged.
        student_embeddings = student_embeddings.float()
        teacher_embeddings = teacher_embeddings.float()

        zero = torch.zeros((), device=student_embeddings.device, dtype=torch.float32)

        loss_dist = (
            self.dist_criterion(student_embeddings, teacher_embeddings)
            if self.dist_ratio != 0.0
            else zero
        )
        loss_angle = (
            self.angle_criterion(student_embeddings, teacher_embeddings)
            if self.angle_ratio != 0.0
            else zero
        )

        total_loss = (
            self.w_task * task_loss
            + self.dist_ratio * loss_dist
            + self.angle_ratio * loss_angle
        )

        metrics = {
            "loss_total": float(total_loss.detach()),
            "loss_task": float(task_loss.detach()),
            "loss_rkd_d": float(loss_dist.detach()),
            "loss_rkd_a": float(loss_angle.detach()),
        }

        return total_loss, metrics
