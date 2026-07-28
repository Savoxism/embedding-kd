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
    r"""Multi-resolution diffusion matching.

    L = sum_r omega_r KL(p^T_r || p^S_r),  p^S_r(j) = softmax_j(cos(s_i,s_j) / tau_r)

    **Why the student distribution is scale-dependent.** With one shared student
    distribution p^S the objective collapses exactly onto a single-scale one:
    cross-entropy is linear in the target, so

        sum_r omega_r KL(p^T_r || p^S)
            = -sum_r omega_r H(p^T_r) + CE(pbar, p^S),   pbar = sum_r omega_r p^T_r,

    and the first term is a precomputed constant. Every choice of scale set that
    yields the same mixture pbar produces the same gradient, so "multi-scale" would
    be nothing more than target smoothing. Two further consequences follow:

    * the loss can never drop below JS_omega(p^T_1, ..., p^T_R) = H(pbar) -
      sum_r omega_r H(p^T_r); that floor is computable offline from the artifact,
      and a loss curve that flattens near it means the objective is exhausted, not
      that optimization stalled;
    * a single softmax pins cos(s_i, s_j) only up to a per-anchor additive constant,
      so nothing constrains similarity levels *across* anchors -- which is exactly
      what STS Spearman and a single global cosine threshold need.

    Giving each scale its own temperature tau_r removes both problems. Matching the
    same candidate similarities at several resolutions is over-determined, so the
    student has to reproduce the teacher's cosine *gaps*, not just its ranking, and
    the scales stop being redundant.

    **In-batch candidate sharing.** Every anchor is scored against the union of all
    candidates in the batch, deduplicated by corpus index and with the anchor's own
    row masked out. Encoding cost is unchanged, negatives per anchor go up by a
    factor of the batch size, and the shared columns couple anchors, which is what
    makes similarity levels comparable across the batch.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        scale_weights: Sequence[float],
        scale_temps: Sequence[float] | None = None,
        lambda_anchor: float = 0.0,
        student_temp: float = 0.07,
        eps_norm: float = 1e-8,
        diag_topk: int = 8,
        share_in_batch: bool = True,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.lambda_anchor = lambda_anchor
        self.student_temp = student_temp
        self.eps_norm = eps_norm
        self.diag_topk = diag_topk
        self.share_in_batch = share_in_batch

        self.use_anchor = lambda_anchor > 0.0
        if self.use_anchor:
            self.anchor_proj = nn.Linear(student_dim, teacher_dim, bias=False)
            nn.init.normal_(self.anchor_proj.weight, mean=0.0, std=1e-3)
        else:
            self.anchor_proj = None

        weights = torch.tensor(list(scale_weights), dtype=torch.float32)
        if weights.numel() == 0:
            weights = torch.ones(1, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-12)
        self.register_buffer("scale_weights", weights)

        if scale_temps is None or len(tuple(scale_temps)) == 0:
            temps = torch.full_like(weights, float(student_temp))
        else:
            temps = torch.tensor(list(scale_temps), dtype=torch.float32)
        if (temps <= 0).any():
            raise ValueError(f"scale_temps must be positive, got {temps.tolist()}")
        self.register_buffer("scale_temps", temps)
        self.temps_tied = bool(temps.numel() == 1 or torch.allclose(temps, temps[0]))

    def _resolved(self, buffer: torch.Tensor, n_scales: int) -> torch.Tensor:
        if buffer.numel() < n_scales:
            pad = buffer[-1:].repeat(n_scales - buffer.numel())
            return torch.cat([buffer, pad], dim=0)
        return buffer[:n_scales]

    def _resolved_weights(self, n_scales: int) -> torch.Tensor:
        weights = self._resolved(self.scale_weights, n_scales)
        return weights / weights.sum().clamp_min(1e-12)

    def _build_shared_pool(
        self,
        candidate_embeddings: torch.Tensor,
        teacher_probs: torch.Tensor,
        candidate_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_scales, candidate_size = teacher_probs.shape
        flat_idx = candidate_idx.reshape(-1)
        unique_idx, inverse = torch.unique(flat_idx, return_inverse=True)
        pool_size = unique_idx.numel()

        # One representative embedding per corpus index. Duplicates across anchors are
        # the same text, so any occurrence is the same vector up to padding.
        representative = torch.zeros(
            pool_size, dtype=torch.long, device=candidate_embeddings.device
        )
        representative.scatter_(
            0,
            inverse,
            torch.arange(
                flat_idx.numel(), device=candidate_embeddings.device, dtype=torch.long
            ),
        )
        pool_embeddings = candidate_embeddings.index_select(0, representative)

        target = torch.zeros(
            batch_size,
            n_scales,
            pool_size,
            dtype=teacher_probs.dtype,
            device=teacher_probs.device,
        )
        scatter_index = (
            inverse.view(batch_size, 1, candidate_size)
            .expand(batch_size, n_scales, candidate_size)
        )
        target.scatter_add_(2, scatter_index, teacher_probs)

        self_mask = unique_idx.view(1, -1) == anchor_idx.view(-1, 1)
        return pool_embeddings, target, self_mask

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        teacher_probs: torch.Tensor,
        candidate_idx: torch.Tensor | None = None,
        anchor_idx: torch.Tensor | None = None,
        teacher_cls: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        checked = [
            ("anchor_embeddings", anchor_embeddings),
            ("candidate_embeddings", candidate_embeddings),
            ("teacher_probs", teacher_probs),
        ]
        if self.use_anchor:
            if teacher_cls is None:
                raise ValueError("lambda_anchor > 0 requires teacher_cls")
            checked.append(("teacher_cls", teacher_cls))
        _assert_finite_tensors(tuple(checked))

        batch_size = anchor_embeddings.size(0)
        candidate_size = teacher_probs.size(-1)
        n_scales = teacher_probs.size(1)

        teacher_probs = teacher_probs.clamp_min(0.0)
        teacher_probs = teacher_probs / teacher_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1, eps=self.eps_norm)
        candidate_embeddings = candidate_embeddings.reshape(
            batch_size * candidate_size, -1
        )

        share = self.share_in_batch and candidate_idx is not None and anchor_idx is not None
        if share:
            pool_embeddings, target, self_mask = self._build_shared_pool(
                candidate_embeddings, teacher_probs, candidate_idx, anchor_idx
            )
            pool_norm = F.normalize(pool_embeddings, p=2, dim=-1, eps=self.eps_norm)
            similarity = anchor_norm @ pool_norm.t()
            target = target.masked_fill(self_mask.unsqueeze(1), 0.0)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            candidate_norm = F.normalize(
                candidate_embeddings.view(batch_size, candidate_size, -1),
                p=2,
                dim=-1,
                eps=self.eps_norm,
            )
            similarity = torch.einsum("bd,bcd->bc", anchor_norm, candidate_norm)
            target = teacher_probs
            self_mask = torch.zeros_like(similarity, dtype=torch.bool)

        weights = self._resolved_weights(n_scales)
        temps = self._resolved(self.scale_temps, n_scales)

        log_target = torch.where(
            target > 0, target.clamp_min(1e-12).log(), torch.zeros_like(target)
        )
        target_entropy = -(target * log_target).sum(dim=-1)

        kl_per_scale = []
        log_probs_per_scale = []
        for scale_idx in range(n_scales):
            logits = similarity / temps[scale_idx]
            logits = logits.masked_fill(self_mask, float("-inf"))
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_per_scale.append(log_probs)
            scale_target = target[:, scale_idx, :]
            # Masked columns carry log_probs = -inf and target = 0, and 0 * inf is
            # NaN, not the 0 the KL sum needs. Zero-target terms are dropped rather
            # than multiplied out.
            contribution = torch.where(
                scale_target > 0,
                scale_target * (log_target[:, scale_idx, :] - log_probs),
                torch.zeros_like(scale_target),
            )
            kl_per_scale.append(contribution.sum(dim=-1))
        kl_per_scale = torch.stack(kl_per_scale, dim=1)
        loss_diff = (kl_per_scale * weights.view(1, -1)).sum(dim=-1).mean()

        if self.use_anchor:
            projected = self.anchor_proj(anchor_embeddings)
            teacher_norm = F.normalize(teacher_cls, p=2, dim=-1, eps=self.eps_norm)
            loss_anchor = (
                1.0 - F.cosine_similarity(projected, teacher_norm, dim=-1).mean()
            )
            total_loss = loss_diff + self.lambda_anchor * loss_anchor
        else:
            loss_anchor = None
            total_loss = loss_diff
        _assert_finite_tensors((("loss_diff", loss_diff), ("total_loss", total_loss)))

        metrics = self._diagnostics(
            total_loss=total_loss,
            loss_diff=loss_diff,
            loss_anchor=loss_anchor,
            kl_per_scale=kl_per_scale,
            log_probs_sharpest=log_probs_per_scale[0],
            target=target,
            target_entropy=target_entropy,
            weights=weights,
            self_mask=self_mask,
        )
        return total_loss, metrics

    @torch.no_grad()
    def _diagnostics(
        self,
        total_loss: torch.Tensor,
        loss_diff: torch.Tensor,
        loss_anchor: torch.Tensor | None,
        kl_per_scale: torch.Tensor,
        log_probs_sharpest: torch.Tensor,
        target: torch.Tensor,
        target_entropy: torch.Tensor,
        weights: torch.Tensor,
        self_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Loss value alone cannot distinguish "learned the geometry" from "went uniform".

        It also cannot distinguish "still learning" from "sitting on the irreducible
        floor", which is why the Jensen-Shannon term and the excess above it are
        logged next to the raw loss.
        """
        probs_student = log_probs_sharpest.exp()
        n_columns = (~self_mask).sum(dim=-1).clamp_min(1).float()
        student_entropy = -(
            probs_student * torch.where(
                probs_student > 0,
                log_probs_sharpest,
                torch.zeros_like(log_probs_sharpest),
            )
        ).sum(dim=-1)
        uniform_entropy = n_columns.log().clamp_min(1e-12)

        # Irreducible floor: JS_omega(p_1..p_R) = H(pbar) - sum_r omega_r H(p_r).
        # Exactly the minimum of L_diff when all scales share one temperature; a
        # lower bound (and a scale-disagreement statistic) when they do not.
        mixture = (target * weights.view(1, -1, 1)).sum(dim=1)
        log_mixture = torch.where(
            mixture > 0, mixture.clamp_min(1e-12).log(), torch.zeros_like(mixture)
        )
        mixture_entropy = -(mixture * log_mixture).sum(dim=-1)
        weighted_entropy = (target_entropy * weights.view(1, -1)).sum(dim=-1)
        js_floor = (mixture_entropy - weighted_entropy).clamp_min(0.0)

        sharpest = target[:, 0, :]
        k = min(self.diag_topk, sharpest.size(-1))
        teacher_top = sharpest.topk(k, dim=-1).indices
        mass_on_teacher_top = probs_student.gather(-1, teacher_top).sum(dim=-1).mean()

        scalars = [
            total_loss.detach(),
            loss_diff.detach(),
            js_floor.mean(),
            (loss_diff - js_floor.mean()).detach(),
            (loss_diff + weighted_entropy.mean()).detach(),
            weighted_entropy.mean(),
            student_entropy.mean(),
            (student_entropy / uniform_entropy).mean(),
            probs_student.max(dim=-1).values.mean(),
            sharpest.max(dim=-1).values.mean(),
            mass_on_teacher_top,
            n_columns.mean(),
        ]
        names = [
            "loss_total",
            "loss_diff",
            "js_floor",
            "loss_excess",
            "loss_cross_entropy",
            "target_entropy",
            "student_entropy",
            "student_entropy_ratio",
            "student_top1",
            "target_top1",
            f"student_mass_on_teacher_top{k}",
            "candidates_per_anchor",
        ]
        if loss_anchor is not None:
            scalars += [loss_anchor.detach(), (self.lambda_anchor * loss_anchor).detach()]
            names += ["loss_anchor", "weighted_anchor"]
        per_scale = list(kl_per_scale.mean(dim=0).detach())
        # One device sync for all logged scalars instead of one sync per scalar.
        values = torch.stack(
            [value.float().reshape(()) for value in scalars + per_scale]
        ).tolist()
        metrics = dict(zip(names, values[: len(names)]))
        for scale_idx, value in enumerate(values[len(names) :]):
            metrics[f"kl_scale{scale_idx}"] = value
        metrics["excess_is_exact"] = float(self.temps_tied)
        return metrics
