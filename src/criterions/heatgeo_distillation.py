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

    L = L_diff + lambda_SGC L_SGC

    L_diff = sum_r omega_r KL(p^T_r || p^S_r),
    p^S_r(j) = softmax_j(cos(s_i,s_j) / tau_r)

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

    **The direct scale, and why it is not optional.** The diffusion targets are the
    graph's mass renormalized over the scored columns, so every column outside the
    anchor's diffusion pool receives target *exactly zero*. That is not a neutral
    "no information" value: the gradient of the cross-entropy at such a column is
    +p^S(j), which pushes cos(s_i, s_j) down without bound. Under in-batch sharing
    roughly 97% of the columns an anchor sees are zero-target, and they include the
    hard negatives -- same-source, top-200 by teacher cosine, excluded from the
    mutual-kNN graph -- whose true teacher similarity is high. The objective is
    therefore actively training the student to drive apart pairs the teacher calls
    similar, which is exactly the calibration STS and a global cosine threshold
    depend on.

    Scale r=0 fixes this by targeting the teacher's own similarity over the full
    column set,

        p^T_0(j) = softmax_j( cos(t_i, t_j) / tau_t ),

    which is dense: every scored column gets its true teacher mass instead of a
    false zero. It costs nothing (the teacher embeddings are already cached), and it
    supplies the absolute calibration the objective needs.

    **Separate column domains, and why adding r=0 was not enough on its own.**
    Adding r=0 alongside diffusion scales that still softmax over the whole shared
    pool does not remove the false zeros -- it adds a term that argues with them. The
    diffusion scales keep pushing ~900 of ~965 columns down; r=0 spends its weight
    pulling the same columns back up. On the 13.5k-row run the two sides disagreed by
    JS = 0.42 nats, 61% of the maximum possible for a two-way split, and that
    disagreement accounted for 94% of an irreducible loss floor of 0.45. Against a
    total loss of 0.84 that left only 0.39 nats reachable, the student closed 99.4% of
    it inside one epoch, and every benchmark was flat from epoch 1 onward.

    The fix is to give the two families different column sets. Diffusion scales
    softmax over the anchor's *own* candidate draw, where their zeros are real
    teacher judgements (a hard negative genuinely carries no diffusion mass); r=0
    softmaxes over the full shared pool, where every column carries real teacher mass.
    Neither term has an opinion the other contradicts: diffusion ranks within the
    neighbourhood, r=0 calibrates across the batch.

    **Similarity Gauge Calibration (SGC).** Every softmax term is invariant to a
    row-wise translation of its similarities:

        softmax((c_i + a_i) / tau) = softmax(c_i / tau).

    L_diff therefore identifies relative heat flow but leaves one scalar similarity
    origin per anchor unconstrained. SGC fixes only that missing degree of freedom.
    Using the direct teacher distribution q_i over the already-scored columns,

        L_SGC = mean_i Huber(sum_j q_ij c^S_ij - sum_j q_ij c^T_ij).

    It adds no parameters, projections, pair construction, or model forward pass.
    Teacher weighting keeps the calibration focused on semantically relevant columns
    instead of letting the many easy in-batch negatives dominate the mean.

    **Why there is no pointwise anchor term.** An earlier version added
    lambda_anchor * (1 - cos(W_a s_i, t_i)) with a free linear map W_a. That term is
    invariant to any invertible transform of the student space -- W_a simply absorbs
    it -- so it cannot pin absolute similarity levels no matter how it is weighted,
    which is the one thing it was introduced to do. The direct scale supplies that
    calibration properly, by comparing the teacher's *relative* similarities over a
    shared column set rather than comparing cosines across two different metrics. The
    term has been removed rather than left at weight 0: it also made the criterion
    carry trainable parameters, and a knob that cannot work is worse than no knob.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        scale_weights: Sequence[float],
        scale_temps: Sequence[float] | None = None,
        student_temp: float = 0.07,
        eps_norm: float = 1e-8,
        diag_topk: int = 8,
        share_in_batch: bool = True,
        teacher_embeddings: torch.Tensor | None = None,
        direct_weight: float = 1.0,
        direct_temp: float = 0.10,
        direct_student_temp: float = 0.10,
        sgc_weight: float = 0.0,
        sgc_huber_delta: float = 0.10,
        use_diffusion_loss: bool = True,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.student_temp = student_temp
        self.eps_norm = eps_norm
        self.diag_topk = diag_topk
        self.share_in_batch = share_in_batch
        self.use_diffusion_loss = bool(use_diffusion_loss)
        self.sgc_weight = float(sgc_weight)
        self.sgc_huber_delta = float(sgc_huber_delta)
        if self.sgc_weight < 0.0:
            raise ValueError(f"sgc_weight must be non-negative, got {sgc_weight}")
        if self.sgc_huber_delta <= 0.0:
            raise ValueError(
                f"sgc_huber_delta must be positive, got {sgc_huber_delta}"
            )

        self.use_direct = teacher_embeddings is not None and direct_weight > 0.0
        self.use_sgc = self.sgc_weight > 0.0
        if not self.use_diffusion_loss and not self.use_direct:
            raise ValueError(
                "direct-only mode requires teacher_embeddings and direct_weight > 0"
            )
        if self.use_sgc and not self.use_direct:
            raise ValueError("SGC requires the direct teacher scale")
        if self.use_direct:
            if direct_temp <= 0.0 or direct_student_temp <= 0.0:
                raise ValueError("direct temperatures must be positive")
            # Stored normalized and in half precision: the only operation it feeds is
            # a cosine, and at corpus scale this buffer is the largest thing the
            # criterion owns (N x 2560).
            normalized = F.normalize(
                teacher_embeddings.float(), p=2, dim=-1, eps=eps_norm
            ).half()
            self.register_buffer("teacher_bank", normalized, persistent=False)
            self.direct_temp = float(direct_temp)
            self.register_buffer(
                "direct_weight", torch.tensor([float(direct_weight)])
            )
            self.register_buffer(
                "direct_student_temp", torch.tensor([float(direct_student_temp)])
            )
        else:
            self.teacher_bank = None
            self.direct_temp = float(direct_temp)

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

    def _build_shared_pool(
        self,
        candidate_embeddings: torch.Tensor,
        teacher_probs: torch.Tensor,
        candidate_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Which pool columns belong to this anchor's own candidate draw. The diffusion
        # scales only have an opinion inside this set; everything else in the shared
        # pool was drawn for a different anchor and carries target 0 for reasons that
        # have nothing to do with the teacher.
        own_mask = torch.zeros(
            batch_size, pool_size, dtype=torch.bool, device=teacher_probs.device
        )
        own_mask.scatter_(1, inverse.view(batch_size, candidate_size), True)

        self_mask = unique_idx.view(1, -1) == anchor_idx.view(-1, 1)
        return pool_embeddings, target, self_mask, own_mask, unique_idx

    def _build_direct_columns(
        self,
        candidate_embeddings: torch.Tensor,
        candidate_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        batch_size, candidate_size = candidate_idx.shape
        flat_embeddings = candidate_embeddings.reshape(
            batch_size * candidate_size, -1
        )
        if self.share_in_batch:
            flat_idx = candidate_idx.reshape(-1)
            unique_idx, inverse = torch.unique(flat_idx, return_inverse=True)
            representative = torch.zeros(
                unique_idx.numel(), dtype=torch.long, device=flat_idx.device
            )
            representative.scatter_(
                0,
                inverse,
                torch.arange(
                    flat_idx.numel(), device=flat_idx.device, dtype=torch.long
                ),
            )
            columns = flat_embeddings.index_select(0, representative)
            self_mask = unique_idx.view(1, -1) == anchor_idx.view(-1, 1)
            return columns, self_mask, unique_idx, True

        columns = flat_embeddings.view(batch_size, candidate_size, -1)
        self_mask = candidate_idx == anchor_idx.view(-1, 1)
        return columns, self_mask, candidate_idx, False

    @torch.no_grad()
    def _direct_target(
        self,
        anchor_idx: torch.Tensor,
        column_idx: torch.Tensor,
        self_mask: torch.Tensor,
        shared: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher similarity over every scored column, not just the graph pool."""
        bank = self.teacher_bank
        t_anchor = bank.index_select(0, anchor_idx).float()
        if shared:
            t_columns = bank.index_select(0, column_idx).float()
            cosine = t_anchor @ t_columns.t()
        else:
            batch_size, candidate_size = column_idx.shape
            t_columns = bank.index_select(0, column_idx.reshape(-1)).float()
            t_columns = t_columns.view(batch_size, candidate_size, -1)
            cosine = torch.einsum("bd,bcd->bc", t_anchor, t_columns)
        logits = cosine / self.direct_temp
        logits = logits.masked_fill(self_mask, float("-inf"))
        return F.softmax(logits, dim=-1), cosine

    def _sgc_loss(
        self,
        student_similarity: torch.Tensor,
        teacher_similarity: torch.Tensor,
        teacher_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        weights = teacher_weights.detach().float()
        student_origin = (weights * student_similarity.float()).sum(dim=-1)
        teacher_origin = (weights * teacher_similarity.float()).sum(dim=-1)
        gap = student_origin - teacher_origin
        loss = F.smooth_l1_loss(
            gap,
            torch.zeros_like(gap),
            beta=self.sgc_huber_delta,
        )
        stats = (
            gap.abs().mean().detach(),
            student_origin.mean().detach(),
            teacher_origin.mean().detach(),
        )
        return loss, stats

    def _forward_direct_only(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_idx: torch.Tensor | None,
        anchor_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if candidate_idx is None or anchor_idx is None:
            raise ValueError(
                "candidate_idx and anchor_idx are required in direct-only mode"
            )
        if candidate_idx.ndim != 2 or anchor_idx.ndim != 1:
            raise ValueError("candidate_idx must be [B, C] and anchor_idx must be [B]")
        if candidate_idx.size(0) != anchor_embeddings.size(0):
            raise ValueError("candidate_idx batch size does not match anchor embeddings")
        expected_rows = candidate_idx.numel()
        if candidate_embeddings.reshape(-1, candidate_embeddings.size(-1)).size(0) != expected_rows:
            raise ValueError(
                f"candidate embeddings contain the wrong number of rows: "
                f"expected {expected_rows}"
            )

        _assert_finite_tensors(
            (
                ("anchor_embeddings", anchor_embeddings),
                ("candidate_embeddings", candidate_embeddings),
            )
        )
        anchor_norm = F.normalize(
            anchor_embeddings, p=2, dim=-1, eps=self.eps_norm
        )
        columns, self_mask, column_idx, shared = self._build_direct_columns(
            candidate_embeddings, candidate_idx, anchor_idx
        )
        column_norm = F.normalize(columns, p=2, dim=-1, eps=self.eps_norm)
        if shared:
            similarity = anchor_norm @ column_norm.t()
        else:
            similarity = torch.einsum("bd,bcd->bc", anchor_norm, column_norm)

        direct, teacher_similarity = self._direct_target(
            anchor_idx, column_idx, self_mask, shared
        )
        logits = similarity / self.direct_student_temp.to(similarity.dtype)
        logits = logits.masked_fill(self_mask, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)
        log_direct = torch.where(
            direct > 0, direct.clamp_min(1e-12).log(), torch.zeros_like(direct)
        )
        contribution = torch.where(
            direct > 0,
            direct * (log_direct - log_probs),
            torch.zeros_like(direct),
        )
        loss_direct = contribution.sum(dim=-1).mean()

        loss_sgc = anchor_norm.sum() * 0.0
        zero = loss_sgc.detach()
        sgc_stats = (zero, zero, zero)
        if self.use_sgc:
            loss_sgc, sgc_stats = self._sgc_loss(
                similarity, teacher_similarity, direct
            )
        loss_sgc_weighted = self.sgc_weight * loss_sgc
        total_loss = loss_direct + loss_sgc_weighted
        _assert_finite_tensors(
            (
                ("loss_direct", loss_direct),
                ("loss_sgc", loss_sgc),
                ("loss_total", total_loss),
            )
        )

        with torch.no_grad():
            student_probs = log_probs.exp()
            teacher_entropy = -(direct * log_direct).sum(dim=-1).mean()
            student_entropy = -(
                student_probs
                * torch.where(
                    student_probs > 0, log_probs, torch.zeros_like(log_probs)
                )
            ).sum(dim=-1).mean()
            available = (~self_mask).sum(dim=-1).float()
            entropy_ratio = student_entropy / available.clamp_min(2.0).log().mean()
            k = min(self.diag_topk, direct.size(-1))
            teacher_top = direct.topk(k, dim=-1).indices
            scalars = torch.stack(
                [
                    total_loss.detach(),
                    loss_direct.detach(),
                    loss_sgc.detach(),
                    loss_sgc_weighted.detach(),
                    *sgc_stats,
                    teacher_entropy,
                    student_entropy,
                    entropy_ratio,
                    student_probs.max(dim=-1).values.mean(),
                    direct.max(dim=-1).values.mean(),
                    student_probs.gather(-1, teacher_top).sum(dim=-1).mean(),
                    available.mean(),
                ]
            ).float().tolist()
        names = (
            "loss_total",
            "loss_direct",
            "loss_sgc",
            "loss_sgc_weighted",
            "sgc_abs_origin_gap",
            "sgc_student_origin",
            "sgc_teacher_origin",
            "teacher_entropy_direct",
            "student_entropy_direct",
            "student_entropy_ratio_direct",
            "student_top1_direct",
            "target_top1_direct",
            f"student_mass_on_teacher_top{k}_direct",
            "pool_columns_direct",
        )
        metrics = dict(zip(names, scalars))
        metrics["kl_direct"] = metrics["loss_direct"]
        metrics["candidates_per_anchor"] = float(candidate_idx.size(1))
        metrics["diffusion_loss_active"] = 0.0
        self.direct_active = True
        return total_loss, metrics

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        teacher_probs: torch.Tensor | None,
        candidate_idx: torch.Tensor | None = None,
        anchor_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.use_diffusion_loss:
            return self._forward_direct_only(
                anchor_embeddings,
                candidate_embeddings,
                candidate_idx,
                anchor_idx,
            )
        if teacher_probs is None:
            raise ValueError("teacher_probs is required when diffusion loss is active")
        _assert_finite_tensors(
            (
                ("anchor_embeddings", anchor_embeddings),
                ("candidate_embeddings", candidate_embeddings),
                ("teacher_probs", teacher_probs),
            )
        )

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
            (
                pool_embeddings,
                target,
                self_mask,
                own_mask,
                column_idx,
            ) = self._build_shared_pool(
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
            column_idx = candidate_idx
            # Without sharing every column already belongs to this anchor, so the
            # two domains coincide. The anchor is still masked explicitly rather than
            # assumed absent: it is excluded upstream by construction, but nothing
            # here enforces it, and an anchor scored against itself lands at cos = 1
            # with the sharpest temperature behind it.
            own_mask = torch.ones_like(similarity, dtype=torch.bool)
            if column_idx is not None and anchor_idx is not None:
                self_mask = column_idx == anchor_idx.view(-1, 1)
            else:
                self_mask = torch.zeros_like(similarity, dtype=torch.bool)

        weights = self._resolved(self.scale_weights, n_scales)
        temps = self._resolved(self.scale_temps, n_scales)

        # Scale r=0: the teacher's own similarity over every scored column. Without
        # it, every column outside the anchor's diffusion pool carries target 0 and
        # is pushed toward maximal dissimilarity regardless of what the teacher says.
        self.direct_active = self.use_direct and anchor_idx is not None and column_idx is not None
        if self.use_sgc and not self.direct_active:
            raise ValueError(
                "anchor_idx and candidate_idx are required when SGC is active"
            )
        loss_sgc = anchor_norm.sum() * 0.0
        sgc_stats = (
            loss_sgc.detach(),
            loss_sgc.detach(),
            loss_sgc.detach(),
        )
        if self.direct_active:
            direct, teacher_similarity = self._direct_target(
                anchor_idx, column_idx, self_mask, share
            )
            if self.use_sgc:
                loss_sgc, sgc_stats = self._sgc_loss(
                    similarity, teacher_similarity, direct
                )
            target = torch.cat([direct.unsqueeze(1).to(target.dtype), target], dim=1)
            weights = torch.cat([self.direct_weight.to(weights.dtype), weights])
            temps = torch.cat([self.direct_student_temp.to(temps.dtype), temps])
            n_scales += 1
        weights = weights / weights.sum().clamp_min(1e-12)

        log_target = torch.where(
            target > 0, target.clamp_min(1e-12).log(), torch.zeros_like(target)
        )
        target_entropy = -(target * log_target).sum(dim=-1)

        # Column domain per scale. The diffusion targets are the graph's mass
        # renormalized over *this anchor's* draw, so outside that draw their zeros are
        # an artefact of who else happened to be in the batch, not a teacher judgement.
        # Softmaxing them over the whole shared pool turns those artefacts into a
        # gradient that pushes ~900 of ~965 cosines down, and the direct scale then
        # spends its weight pulling the same columns back up. Measured on the 13.5k
        # run, the two halves disagreed by 0.42 nats -- 61% of the maximum possible --
        # and that disagreement was 94% of the irreducible loss floor, which is why
        # the objective saturated after one epoch.
        #
        # Restricting the diffusion softmax to the anchor's own columns makes the
        # gradient on every other column exactly zero, so the diffusion scales rank
        # within the neighbourhood and the direct scale owns absolute calibration
        # across the full pool. Complementary instead of opposed.
        diffusion_mask = self_mask | ~own_mask

        kl_per_scale = []
        log_probs_per_scale = []
        for scale_idx in range(n_scales):
            is_direct = self.direct_active and scale_idx == 0
            logits = similarity / temps[scale_idx]
            logits = logits.masked_fill(self_mask if is_direct else diffusion_mask,
                                        float("-inf"))
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
        _assert_finite_tensors((("loss_diff", loss_diff),))
        sgc_weighted = self.sgc_weight * loss_sgc
        total_loss = loss_diff + sgc_weighted
        _assert_finite_tensors(
            (
                ("loss_sgc", loss_sgc),
                ("loss_total", total_loss),
            )
        )

        metrics = self._diagnostics(
            total_loss=total_loss,
            loss_diff=loss_diff,
            kl_per_scale=kl_per_scale,
            log_probs_per_scale=log_probs_per_scale,
            target=target,
            target_entropy=target_entropy,
            weights=weights,
            self_mask=self_mask,
            diffusion_mask=diffusion_mask,
            temps=temps,
        )
        sgc_values = torch.stack(
            [
                loss_sgc.detach(),
                sgc_weighted.detach(),
                *sgc_stats,
            ]
        ).float().tolist()
        metrics.update(
            dict(
                zip(
                    (
                        "loss_sgc",
                        "loss_sgc_weighted",
                        "sgc_abs_origin_gap",
                        "sgc_student_origin",
                        "sgc_teacher_origin",
                    ),
                    sgc_values,
                )
            )
        )
        return total_loss, metrics

    @torch.no_grad()
    def _diagnostics(
        self,
        total_loss: torch.Tensor,
        loss_diff: torch.Tensor,
        kl_per_scale: torch.Tensor,
        log_probs_per_scale: list[torch.Tensor],
        target: torch.Tensor,
        target_entropy: torch.Tensor,
        weights: torch.Tensor,
        self_mask: torch.Tensor,
        diffusion_mask: torch.Tensor,
        temps: torch.Tensor,
    ) -> dict[str, float]:
        """Loss value alone cannot distinguish "learned the geometry" from "went uniform".

        It also cannot distinguish "still learning" from "sitting on the irreducible
        floor", which is why the Jensen-Shannon term and the excess above it are
        logged next to the raw loss.

        Unsuffixed metrics describe the *sharpest diffusion* scale, over the anchor's
        own candidate columns. An earlier version indexed scale 0, which stopped being
        that scale the moment the direct target was prepended to the stack: the curves
        kept their names and silently started reporting the direct scale at a
        different temperature over a 15x larger column set. The direct scale now
        reports under its own `*_direct` names.
        """
        offset = 1 if getattr(self, "direct_active", False) else 0
        k = min(self.diag_topk, target.size(-1))

        def _distribution_stats(
            log_probs: torch.Tensor, scale_target: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            probs_student = log_probs.exp()
            student_entropy = -(
                probs_student
                * torch.where(probs_student > 0, log_probs, torch.zeros_like(log_probs))
            ).sum(dim=-1)
            # Two columns is the smallest set on which a softmax has any freedom, so
            # it is the smallest denominator for which the ratio means anything.
            n_columns = (~mask).sum(dim=-1).float()
            uniform_entropy = n_columns.clamp_min(2.0).log()
            teacher_top = scale_target.topk(k, dim=-1).indices
            return (
                student_entropy.mean(),
                (student_entropy / uniform_entropy).mean(),
                probs_student.max(dim=-1).values.mean(),
                scale_target.max(dim=-1).values.mean(),
                probs_student.gather(-1, teacher_top).sum(dim=-1).mean(),
                n_columns.mean(),
            )

        # Irreducible floor of the *diffusion group*: sum_r w_r KL(p_r||q) >= W *
        # JS_v(p_r), with W the group's total weight and v_r = w_r/W. The direct scale
        # is excluded: it no longer shares a column domain with the diffusion scales,
        # so a single q cannot be substituted into both, and on its own its floor is
        # zero. Folding it in was what made js_floor read 0.45 nats while the diffusion
        # scales genuinely disagreed by 0.05 -- the gap was the two halves of the
        # objective fighting, reported as if it were a property of the targets.
        diff_target = target[:, offset:, :]
        diff_weights = weights[offset:].view(1, -1)
        # Scales with no mass on this anchor's draw contribute KL 0 and must not dilute
        # the mixture either; their weight is dropped for that anchor only.
        has_mass = diff_target.sum(dim=-1) > 0
        w_eff = diff_weights * has_mass.to(diff_weights.dtype)
        w_total = w_eff.sum(dim=-1, keepdim=True)
        w_norm = w_eff / w_total.clamp_min(1e-12)

        mixture = (diff_target * w_norm.unsqueeze(-1)).sum(dim=1)
        log_mixture = torch.where(
            mixture > 0, mixture.clamp_min(1e-12).log(), torch.zeros_like(mixture)
        )
        mixture_entropy = -(mixture * log_mixture).sum(dim=-1)
        group_entropy = (target_entropy[:, offset:] * w_norm).sum(dim=-1)
        js_floor = (
            w_total.squeeze(-1) * (mixture_entropy - group_entropy).clamp_min(0.0)
        )

        # Full-stack weighted entropy: loss_diff = CE - H holds over every scale that
        # is actually in the loss, direct included.
        weighted_entropy = (target_entropy * weights.view(1, -1)).sum(dim=-1)

        scalars = [
            total_loss.detach(),
            loss_diff.detach(),
            js_floor.mean(),
            (loss_diff - js_floor.mean()).detach(),
            (loss_diff + weighted_entropy.mean()).detach(),
            weighted_entropy.mean(),
            # `target_entropy` above is the whole weighted stack, because that is what
            # loss_cross_entropy needs. It is therefore the one metric here that is not
            # scoped to the sharpest diffusion scale, and comparing it against
            # student_entropy compares two different column domains. This is the
            # teacher entropy that student_entropy is actually the counterpart of.
            target_entropy[:, offset].mean(),
            *_distribution_stats(
                log_probs_per_scale[offset], target[:, offset, :], diffusion_mask
            ),
        ]
        names = [
            "loss_total",
            "loss_diff",
            "js_floor",
            "loss_excess",
            "loss_cross_entropy",
            "target_entropy",
            "teacher_entropy_scale",
            "student_entropy",
            "student_entropy_ratio",
            "student_top1",
            "target_top1",
            f"student_mass_on_teacher_top{k}",
            "candidates_per_anchor",
        ]
        if offset:
            scalars.append(target_entropy[:, 0].mean())
            scalars.extend(
                _distribution_stats(log_probs_per_scale[0], target[:, 0, :], self_mask)
            )
            names.extend(
                [
                    "teacher_entropy_direct",
                    "student_entropy_direct",
                    "student_entropy_ratio_direct",
                    "student_top1_direct",
                    "target_top1_direct",
                    f"student_mass_on_teacher_top{k}_direct",
                    "pool_columns_direct",
                ]
            )

        per_scale = list(kl_per_scale.mean(dim=0).detach())
        # One device sync for all logged scalars instead of one sync per scalar.
        values = torch.stack(
            [value.float().reshape(()) for value in scalars + per_scale]
        ).tolist()
        metrics = dict(zip(names, values[: len(names)]))
        per_scale_values = values[len(names) :]
        if offset:
            metrics["kl_direct"] = per_scale_values[0]
        for scale_idx, value in enumerate(per_scale_values[offset:]):
            metrics[f"kl_scale{scale_idx}"] = value
        # js_floor bounds the diffusion group only, and only under a tied student
        # temperature. With distinct tau_r the true minimum is lower, so loss_excess
        # is an upper bound on what is left to learn -- it reaching 0 means the
        # objective is spent, but it can also go negative.
        metrics["excess_is_exact"] = float(
            bool(
                temps.numel() - offset <= 1
                or torch.allclose(temps[offset:], temps[offset])
            )
        )
        return metrics
