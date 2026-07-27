from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _assert_finite_tensors(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> None:
    finite_status = None
    for _, tensor in named_tensors:
        current = torch.isfinite(tensor).all()
        finite_status = current if finite_status is None else finite_status & current

    if finite_status is None or bool(finite_status.item()):
        return

    for name, tensor in named_tensors:
        if bool(torch.isfinite(tensor).all().item()):
            continue
        if tensor.is_floating_point() or tensor.is_complex():
            nan_count = int(torch.isnan(tensor).sum().item())
            inf_count = int(torch.isinf(tensor).sum().item())
        else:
            nan_count = 0
            inf_count = 0
        raise RuntimeError(
            f"HeatGeo non-finite tensor {name!r}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}, "
            f"nan_count={nan_count}, inf_count={inf_count}"
        )


class HeatGeoDistillation(nn.Module):
    """Two-term objective: multi-scale diffusion matching + a pointwise teacher anchor.

    L = L_diff + lambda_anchor * L_anchor

    The spectral term was removed: matching heat diffusion at several scales already
    matches the graph spectrum with an increasingly strong low-pass, so a separate
    eigenvector-regression term distils the same information a second time (and, with
    unit-norm eigenvectors over N rows, at a magnitude of ~1/N it contributed nothing
    to the gradient anyway). The InfoNCE task term was removed for the same reason it
    had no effect at w_task=1e-3: it is orthogonal to what this loss is meant to test.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        scale_weights: Sequence[float],
        lambda_anchor: float = 0.05,
        student_temp: float = 0.07,
        eps_norm: float = 1e-8,
        diag_topk: int = 8,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.lambda_anchor = lambda_anchor
        self.student_temp = student_temp
        self.eps_norm = eps_norm
        self.diag_topk = diag_topk

        self.anchor_proj = nn.Linear(student_dim, teacher_dim, bias=False)

        weights = torch.tensor(list(scale_weights), dtype=torch.float32)
        if weights.numel() == 0:
            weights = torch.ones(1, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-12)
        self.register_buffer("scale_weights", weights)

        nn.init.normal_(self.anchor_proj.weight, mean=0.0, std=1e-3)

    def _resolved_weights(self, n_scales: int) -> torch.Tensor:
        if self.scale_weights.numel() < n_scales:
            pad = self.scale_weights[-1:].repeat(n_scales - self.scale_weights.numel())
            weights = torch.cat([self.scale_weights, pad], dim=0)
        else:
            weights = self.scale_weights[:n_scales]
        return weights / weights.sum().clamp_min(1e-12)

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        teacher_probs: torch.Tensor,
        teacher_cls: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        _assert_finite_tensors(
            (
                ("anchor_embeddings", anchor_embeddings),
                ("candidate_embeddings", candidate_embeddings),
                ("teacher_probs", teacher_probs),
                ("teacher_cls", teacher_cls),
            )
        )

        batch_size = anchor_embeddings.size(0)
        candidate_size = teacher_probs.size(-1)
        n_scales = teacher_probs.size(1)

        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1, eps=self.eps_norm)
        candidate_embeddings = candidate_embeddings.view(batch_size, candidate_size, -1)
        candidate_norm = F.normalize(
            candidate_embeddings, p=2, dim=-1, eps=self.eps_norm
        )

        logits = (
            torch.einsum("bd,bcd->bc", anchor_norm, candidate_norm) / self.student_temp
        )
        log_probs_student = F.log_softmax(logits, dim=-1)

        teacher_probs = teacher_probs.clamp_min(1e-12)
        teacher_probs = teacher_probs / teacher_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        weights = self._resolved_weights(n_scales)

        log_teacher = teacher_probs.log()
        kl_per_scale = (
            teacher_probs * (log_teacher - log_probs_student.unsqueeze(1))
        ).sum(dim=-1)
        loss_diff = (kl_per_scale * weights.view(1, -1)).sum(dim=-1).mean()

        projected = self.anchor_proj(anchor_embeddings)
        teacher_norm = F.normalize(teacher_cls, p=2, dim=-1, eps=self.eps_norm)
        loss_anchor = 1.0 - F.cosine_similarity(projected, teacher_norm, dim=-1).mean()

        total_loss = loss_diff + self.lambda_anchor * loss_anchor
        _assert_finite_tensors(
            (
                ("loss_diff", loss_diff),
                ("loss_anchor", loss_anchor),
                ("total_loss", total_loss),
            )
        )

        metrics = self._diagnostics(
            total_loss=total_loss,
            loss_diff=loss_diff,
            loss_anchor=loss_anchor,
            kl_per_scale=kl_per_scale,
            log_probs_student=log_probs_student,
            teacher_probs=teacher_probs,
        )
        return total_loss, metrics

    @torch.no_grad()
    def _diagnostics(
        self,
        total_loss: torch.Tensor,
        loss_diff: torch.Tensor,
        loss_anchor: torch.Tensor,
        kl_per_scale: torch.Tensor,
        log_probs_student: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> dict[str, float]:
        """Loss value alone cannot distinguish "learned the geometry" from "went uniform"."""
        probs_student = log_probs_student.exp()
        candidate_size = probs_student.size(-1)
        student_entropy = -(probs_student * log_probs_student).sum(dim=-1).mean()
        uniform_entropy = torch.log(
            torch.tensor(float(candidate_size), device=probs_student.device)
        )

        sharpest = teacher_probs[:, 0, :]
        k = min(self.diag_topk, candidate_size)
        teacher_top = sharpest.topk(k, dim=-1).indices
        mass_on_teacher_top = probs_student.gather(-1, teacher_top).sum(dim=-1).mean()

        scalars = [
            total_loss.detach(),
            loss_diff.detach(),
            loss_anchor.detach(),
            (self.lambda_anchor * loss_anchor).detach(),
            student_entropy,
            student_entropy / uniform_entropy,
            probs_student.max(dim=-1).values.mean(),
            sharpest.max(dim=-1).values.mean(),
            mass_on_teacher_top,
        ]
        names = [
            "loss_total",
            "loss_diff",
            "loss_anchor",
            "weighted_anchor",
            "student_entropy",
            "student_entropy_ratio",
            "student_top1",
            "target_top1",
            f"student_mass_on_teacher_top{k}",
        ]
        per_scale = list(kl_per_scale.mean(dim=0).detach())
        # One device sync for all logged scalars instead of one sync per scalar.
        values = torch.stack(
            [value.float().reshape(()) for value in scalars + per_scale]
        ).tolist()
        metrics = dict(zip(names, values[: len(names)]))
        for scale_idx, value in enumerate(values[len(names) :]):
            metrics[f"kl_scale{scale_idx}"] = value
        return metrics
