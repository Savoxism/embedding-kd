import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Sequence, Tuple


def _assert_finite_tensors(named_tensors: Sequence[Tuple[str, torch.Tensor]]) -> None:
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
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        spectral_dim: int,
        scale_weights: Sequence[float],
        w_task: float = 0.001,
        lambda_diff: float = 1.0,
        lambda_spec: float = 0.1,
        lambda_anchor: float = 0.05,
        student_temp: float = 0.07,
        eps_norm: float = 1e-8,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.spectral_dim = spectral_dim
        self.w_task = w_task
        self.lambda_diff = lambda_diff
        self.lambda_spec = lambda_spec
        self.lambda_anchor = lambda_anchor
        self.student_temp = student_temp
        self.eps_norm = eps_norm

        self.anchor_proj = nn.Linear(student_dim, teacher_dim, bias=False)
        if spectral_dim > 0:
            self.spec_proj = nn.Linear(student_dim, spectral_dim, bias=False)
        else:
            self.spec_proj = None

        weights = torch.tensor(list(scale_weights), dtype=torch.float32)
        if weights.numel() == 0:
            weights = torch.ones(1, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-12)
        self.register_buffer("scale_weights", weights)

        nn.init.normal_(self.anchor_proj.weight, mean=0.0, std=1e-3)
        if self.spec_proj is not None:
            nn.init.normal_(self.spec_proj.weight, mean=0.0, std=1e-3)

    def _active_scales(self, epoch: int, n_scales: int) -> int:
        if epoch <= 0:
            return min(1, n_scales)
        if epoch == 1:
            return min(2, n_scales)
        return n_scales

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        teacher_probs: torch.Tensor,
        teacher_cls: torch.Tensor,
        spectral_target: torch.Tensor,
        task_loss: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        _assert_finite_tensors(
            (
                ("anchor_embeddings", anchor_embeddings),
                ("candidate_embeddings", candidate_embeddings),
                ("teacher_probs", teacher_probs),
                ("teacher_cls", teacher_cls),
                ("spectral_target", spectral_target),
                ("task_loss", task_loss),
            )
        )

        batch_size = anchor_embeddings.size(0)
        candidate_size = teacher_probs.size(-1)
        n_scales = teacher_probs.size(1)

        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1, eps=self.eps_norm)
        candidate_embeddings = candidate_embeddings.view(batch_size, candidate_size, -1)
        candidate_norm = F.normalize(candidate_embeddings, p=2, dim=-1, eps=self.eps_norm)

        logits = torch.einsum("bd,bcd->bc", anchor_norm, candidate_norm) / self.student_temp
        log_probs_student = F.log_softmax(logits, dim=-1)

        active_scales = self._active_scales(epoch, n_scales)
        teacher_probs = teacher_probs[:, :active_scales, :].clamp_min(1e-12)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.scale_weights.numel() < active_scales:
            pad = self.scale_weights[-1:].repeat(active_scales - self.scale_weights.numel())
            weights = torch.cat([self.scale_weights, pad], dim=0)
        else:
            weights = self.scale_weights[:active_scales]
        weights = weights / weights.sum().clamp_min(1e-12)

        log_teacher = teacher_probs.log()
        kl_per_scale = (teacher_probs * (log_teacher - log_probs_student.unsqueeze(1))).sum(dim=-1)
        loss_diff = (kl_per_scale * weights.view(1, -1)).sum(dim=-1).mean()

        projected = self.anchor_proj(anchor_embeddings)
        projected = F.normalize(projected, p=2, dim=-1, eps=self.eps_norm)
        teacher_norm = F.normalize(teacher_cls, p=2, dim=-1, eps=self.eps_norm)
        loss_anchor = 1.0 - F.cosine_similarity(projected, teacher_norm, dim=-1).mean()

        use_spec = (
            self.spec_proj is not None
            and spectral_target.numel() > 0
            and spectral_target.size(-1) == self.spectral_dim
            and epoch >= 2
        )
        if use_spec:
            spectral_pred = self.spec_proj(anchor_embeddings)
            loss_spec = F.mse_loss(spectral_pred, spectral_target)
        else:
            loss_spec = torch.tensor(0.0, device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)

        total_loss = (
            self.w_task * task_loss
            + self.lambda_diff * loss_diff
            + self.lambda_spec * loss_spec
            + self.lambda_anchor * loss_anchor
        )
        _assert_finite_tensors(
            (
                ("task_loss", task_loss),
                ("loss_diff", loss_diff),
                ("loss_spec", loss_spec),
                ("loss_anchor", loss_anchor),
                ("total_loss", total_loss),
            )
        )
        metrics = {
            "loss_total": float(total_loss.detach()),
            "loss_task": float(task_loss.detach()),
            "loss_diff": float(loss_diff.detach()),
            "loss_spec": float(loss_spec.detach()),
            "loss_anchor": float(loss_anchor.detach()),
            "weighted_task": float((self.w_task * task_loss).detach()),
            "weighted_diff": float((self.lambda_diff * loss_diff).detach()),
            "weighted_spec": float((self.lambda_spec * loss_spec).detach()),
            "weighted_anchor": float((self.lambda_anchor * loss_anchor).detach()),
            "active_scales": float(active_scales),
        }
        return total_loss, metrics
