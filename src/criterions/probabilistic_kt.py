"""Probabilistic Knowledge Transfer (PKT).

Baseline reproduction of Passalis & Tefas, *Learning Deep Representations with
Probabilistic Knowledge Transfer* (ECCV 2018), extended as Passalis, Tzelepi &
Tefas, *Probabilistic Knowledge Transfer for Lightweight Deep Representation
Learning* (IEEE TNNLS 32(5):2030-2039, 2021) -- the version this repo cites.

PKT matches the *conditional affinity distribution* that each sample induces
over the other samples, rather than the representations themselves.

Teacher and student conditionals (Eq. 3-4)::

    p_{i|j} = K(x_i, x_j) / sum_{k != j} K(x_k, x_j)
    q_{i|j} = K(y_i, y_j) / sum_{k != j} K(y_k, y_j)

with, per the paper, `sum_{i != j} p_{i|j} = 1` -- the self term is excluded
from both the normaliser and the sum.

Kernels::

    K_gaussian(a, b; sigma) = exp( -||a - b||_2^2 / sigma )                (Eq. 5)
    K_cosine(a, b) = 0.5 * ( a^T b / (||a||_2 ||b||_2) + 1 )  in [0, 1]    (Eq. 6)

Eq. 6 is the kernel "used in this work": it needs no bandwidth, which Sec. 3
gives as the reason for preferring it over Eq. 5.

Objective (Eq. 8), a forward KL from the teacher's conditional to the
student's::

    L = sum_{i=1}^{N} sum_{j != i} p_{j|i} log( p_{j|i} / q_{j|i} )

Because only relations between samples enter, teacher and student spaces need
not share a dimensionality and no projection is learned -- Sec. 2 gives this as
an explicit advantage over hint-based transfer. This module holds no parameters.


Two deliberate deviations, both switchable
------------------------------------------

1. ``exclude_self``. Eq. 3, Eq. 4 and Eq. 8 all restrict the index to `k != j` /
   `j != i`, but the authors' released code
   (https://github.com/passalis/probabilistic_kt, ``nn/pkt.py``) normalises over
   the full row, self term included. With a cosine kernel the self term is the
   largest possible affinity (1.0 after the [0,1] rescaling), so for a batch of
   32 it takes a substantial share of every row's mass and dilutes the
   neighbour signal. The default here follows the *paper*; set
   ``exclude_self=False`` to reproduce the released code.

2. ``reduction``. Eq. 8 is a double sum, the released code takes a mean over all
   N^2 entries. The three settings differ only by a constant factor::

       "sum"       -> Eq. 8 verbatim
       "batchmean" -> Eq. 8 / N   (default: mean per-anchor KL, in nats)
       "mean"      -> Eq. 8 / N^2 (the released code)

   A loss weight tuned for one is off by a factor of N for the next.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


def cosine_kernel(embeddings: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Eq. 6, evaluated for every pair of rows. [N, N] in [0, 1]."""
    norm = torch.sqrt(torch.sum(embeddings.pow(2), dim=1, keepdim=True))
    normalized = embeddings / (norm + eps)
    # The released code zeroes any NaN left by a zero-norm row before the
    # product; `norm + eps` already rules that out, so this only forwards NaN
    # that was present in the input.
    normalized = torch.where(
        torch.isnan(normalized), torch.zeros_like(normalized), normalized
    )
    return (normalized @ normalized.t() + 1.0) / 2.0


def gaussian_kernel(embeddings: torch.Tensor, sigma: float) -> torch.Tensor:
    """Eq. 5, evaluated for every pair of rows. [N, N]."""
    squared_distance = torch.cdist(embeddings, embeddings, p=2).pow(2)
    return torch.exp(-squared_distance / sigma)


class ProbabilisticKT(nn.Module):
    """L = w_task * L_task + w_pkt * L_PKT, with L_PKT from Eq. 8."""

    KERNELS = ("cosine", "gaussian")
    REDUCTIONS = ("sum", "batchmean", "mean")

    def __init__(
        self,
        w_task: float = 0.0,
        w_pkt: float = 1.0,
        kernel: str = "cosine",
        gaussian_sigma: float = 1.0,
        exclude_self: bool = True,
        reduction: str = "batchmean",
        eps: float = 1e-7,
    ):
        super().__init__()
        if kernel not in self.KERNELS:
            raise ValueError(
                f"Unsupported PKT kernel={kernel!r}; expected one of {self.KERNELS}"
            )
        if reduction not in self.REDUCTIONS:
            raise ValueError(
                f"Unsupported PKT reduction={reduction!r}; "
                f"expected one of {self.REDUCTIONS}"
            )

        self.w_task = w_task
        self.w_pkt = w_pkt
        self.kernel = kernel
        self.gaussian_sigma = gaussian_sigma
        self.exclude_self = exclude_self
        self.reduction = reduction
        self.eps = eps

    def affinity(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.kernel == "cosine":
            return cosine_kernel(embeddings, eps=self.eps)
        return gaussian_kernel(embeddings, sigma=self.gaussian_sigma)

    def conditional(self, embeddings: torch.Tensor) -> torch.Tensor:
        """The conditional distribution of Eq. 3 / Eq. 4. Rows sum to 1."""
        affinity = self.affinity(embeddings)
        if self.exclude_self:
            affinity = affinity * (
                1.0 - torch.eye(affinity.size(0), device=affinity.device)
            )
        return affinity / (torch.sum(affinity, dim=1, keepdim=True) + self.eps)

    def divergence(
        self, student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Eq. 8 under the configured reduction."""
        q = self.conditional(student_embeddings)
        with torch.no_grad():
            p = self.conditional(teacher_embeddings)

        # Excluded pairs carry p = 0, so their term vanishes whatever q is.
        terms = p * torch.log((p + self.eps) / (q + self.eps))

        if self.reduction == "sum":
            return terms.sum()
        if self.reduction == "batchmean":
            return terms.sum(dim=1).mean()
        return terms.mean()

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        task_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Row-normalised affinities over a bfloat16 teacher lose the tail of
        # every distribution, which is the part the KL weighs least but the part
        # that carries the neighbourhood structure. Both conditionals are formed
        # in float32; the objective is unchanged.
        student_embeddings = student_embeddings.float()
        teacher_embeddings = teacher_embeddings.float()

        loss_pkt = self.divergence(student_embeddings, teacher_embeddings)
        total_loss = self.w_task * task_loss + self.w_pkt * loss_pkt

        metrics = {
            "loss_total": float(total_loss.detach()),
            "loss_task": float(task_loss.detach()),
            "loss_pkt": float(loss_pkt.detach()),
        }

        return total_loss, metrics
