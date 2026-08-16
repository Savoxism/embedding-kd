from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
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
        use_sinkhorn: bool = False,
        sinkhorn_alpha: float = 0.1,
        sinkhorn_max_iter: int = 50,
        cosent_weight: float = 0.1,
        cosent_tau: float = 0.07,
        cosent_delta: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.student_temp = student_temp
        self.eps_norm = eps_norm
        self.diag_topk = diag_topk
        self.share_in_batch = share_in_batch

        self.use_sinkhorn = bool(use_sinkhorn)
        self.sinkhorn_alpha = float(sinkhorn_alpha)
        self.sinkhorn_max_iter = int(sinkhorn_max_iter)
        
        self.use_cosent = False # Controlled dynamically by curriculum
        self.cosent_weight = float(cosent_weight)
        self.cosent_tau = float(cosent_tau)
        self.cosent_delta = float(cosent_delta)

        # ---- Random Walk Trajectory Distillation --------------------------------
        # When enabled, the loss includes a trajectory NLL term that scores how
        # well the student's transition kernel explains step-by-step random walks
        # sampled from the teacher's transition matrix.  Unlike the marginal
        # diffusion KL (which asks "where do you end up after r steps?"), this
        # asks "can you follow the same path the teacher walks?" -- a stronger
        # constraint that captures intermediate-node ordering and local manifold
        # curvature.
        self.use_walk_loss = False  # Controlled dynamically by curriculum
        self.walk_weight = float(kwargs.get("walk_weight", 0.5))
        self.walk_temp = float(kwargs.get("walk_temp", 0.07))
        # Top-K masking: only keep the K most similar pool nodes in the walk
        # softmax denominator.  This narrows the effective vocabulary from
        # ~2700 (full shared pool) to K, focusing the gradient on local
        # neighborhood discrimination where the walk signal is informative.
        # 0 disables masking (full pool, original behavior).
        self.walk_topk = int(kwargs.get("walk_topk", 128))

        self.use_direct = teacher_embeddings is not None and direct_weight > 0.0
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

    @torch.no_grad()
    def _direct_target(
        self,
        anchor_idx: torch.Tensor,
        column_idx: torch.Tensor,
        self_mask: torch.Tensor,
        shared: bool,
    ) -> torch.Tensor:
        """Teacher similarity over every scored column, not just the graph pool."""
        bank = self.teacher_bank
        t_anchor = bank.index_select(0, anchor_idx).float()
        if shared:
            t_columns = bank.index_select(0, column_idx).float()
            logits = t_anchor @ t_columns.t()
        else:
            batch_size, candidate_size = column_idx.shape
            t_columns = bank.index_select(0, column_idx.reshape(-1)).float()
            t_columns = t_columns.view(batch_size, candidate_size, -1)
            logits = torch.einsum("bd,bcd->bc", t_anchor, t_columns)
        logits = logits / self.direct_temp
        logits = logits.masked_fill(self_mask, float("-inf"))
        return F.softmax(logits, dim=-1)
        
    def _batched_sinkhorn(self, C, a, b, eps=1e-9):
        # Force float32 for numerical stability (prevents eps=1e-9 from becoming 0.0 in float16)
        C = C.float()
        a = a.float()
        b = b.float()
        
        # C: [B, N, N] or [N, N] broadcastable
        # a: [B, N] target marginal (teacher)
        # b: [B, N] source marginal (student)
        K = torch.exp(-C / self.sinkhorn_alpha)
        # add an extra dimension for bmm
        u = torch.ones_like(a).unsqueeze(-1)
        v = torch.ones_like(b).unsqueeze(-1)
        a_ = a.unsqueeze(-1)
        b_ = b.unsqueeze(-1)
        
        # If C is [N, N], K is [N, N]. We need it batched [B, N, N]
        if K.dim() == 2:
            K = K.unsqueeze(0).expand(a.size(0), -1, -1)
            
        K_t = K.transpose(-1, -2)
        
        # Use gradient checkpointing to save memory. 
        # Instead of storing the massive computation graph for all iterations, 
        # checkpointing discards it and recomputes the loop on the fly during backward.
        def _sinkhorn_loop(K_in, K_t_in, a_in, b_in):
            u_iter = torch.ones_like(a_in)
            v_iter = torch.ones_like(b_in)
            for _ in range(self.sinkhorn_max_iter):
                v_iter = b_in / (torch.bmm(K_t_in, u_iter) + eps)
                u_iter = a_in / (torch.bmm(K_in, v_iter) + eps)
            return u_iter, v_iter
            
        u, v = checkpoint.checkpoint(_sinkhorn_loop, K, K_t, a_, b_, use_reentrant=False)
            
        # OT distance = sum(u * K * v^T * C)
        P = u * K * v.transpose(-1, -2)
        
        if C.dim() == 2:
            C_batched = C.unsqueeze(0).expand(a.size(0), -1, -1)
            dist = torch.sum(P * C_batched, dim=(-1, -2))
        else:
            dist = torch.sum(P * C, dim=(-1, -2))
            
        return dist

    def _compute_walk_loss(
        self,
        walk_paths: torch.Tensor,
        pool_norm: torch.Tensor,
        column_idx: torch.Tensor,
        shared: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Trajectory-level NLL over teacher-sampled random walks.

        For each step (j_t -> j_{t+1}) in each walk, the student's transition
        probability is:

            p^S(j_{t+1} | j_t) = softmax_{j' in pool}(cos(s_{j_t}, s_{j'}) / tau_w)

        evaluated at j_{t+1}.  The softmax denominator runs over the shared
        candidate pool, reusing the already-encoded embeddings.

        Walk steps where either j_t or j_{t+1} is not in the pool are skipped --
        this happens when a walk ventured outside the candidate set.

        Args:
            walk_paths: [B, M, L+1] node indices of teacher-sampled walks.
            pool_norm: [pool_size, D] L2-normalized shared pool embeddings.
            column_idx: [pool_size] corpus indices of pool columns.
            shared: whether in-batch sharing is active.

        Returns:
            walk_loss: scalar, mean NLL per valid step.
            walk_metrics: dict with diagnostic values.
        """
        device = pool_norm.device
        batch_size, num_walks, path_len = walk_paths.shape
        walk_length = path_len - 1

        if walk_length <= 0 or num_walks <= 0 or (walk_paths < 0).all():
            zero = torch.tensor(0.0, device=device)
            return zero, {"walk_nll": 0.0, "walk_valid_steps": 0.0}

        # Build a lookup: corpus_index -> pool_position.
        # column_idx is 1-D [pool_size] in shared mode.
        col_np = column_idx.cpu().numpy()
        idx_to_pos = {}
        for pos, ci in enumerate(col_np):
            ci = int(ci)
            if ci not in idx_to_pos:
                idx_to_pos[ci] = pos

        # Map walk paths from corpus indices to pool positions.
        # -1 marks nodes outside the pool (will be masked).
        walks_np = walk_paths.cpu().numpy()
        pool_positions = np.full_like(walks_np, -1)
        for b in range(batch_size):
            for m in range(num_walks):
                for t in range(path_len):
                    node = int(walks_np[b, m, t])
                    if node in idx_to_pos:
                        pool_positions[b, m, t] = idx_to_pos[node]

        pool_pos = torch.from_numpy(pool_positions).long().to(device)

        # For each step (t -> t+1), both j_t and j_{t+1} must be in the pool.
        src_pos = pool_pos[:, :, :-1]   # [B, M, L]
        dst_pos = pool_pos[:, :, 1:]    # [B, M, L]
        valid = (src_pos >= 0) & (dst_pos >= 0)

        n_valid = int(valid.sum().item())
        if n_valid == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, {"walk_nll": 0.0, "walk_valid_steps": 0.0}

        # Gather source embeddings for all valid steps.
        # Flatten valid steps to [n_valid], compute similarity, log_softmax, gather.
        flat_src = src_pos[valid]       # [n_valid]
        flat_dst = dst_pos[valid]       # [n_valid]

        src_emb = pool_norm[flat_src]   # [n_valid, D]
        # Similarity of each source against all pool columns.
        logits = src_emb @ pool_norm.t()  # [n_valid, pool_size]
        logits = logits / self.walk_temp

        # Mask out self-transitions (source == column).
        self_mask = flat_src.unsqueeze(1) == torch.arange(
            pool_norm.size(0), device=device
        ).unsqueeze(0)
        logits = logits.masked_fill(self_mask, float("-inf"))

        # ---- Top-K masking ----
        # Narrow the softmax denominator from the full shared pool (~2700)
        # to the K most similar nodes per source.  This focuses the gradient
        # on discriminating among nearby neighbors — the only regime where
        # the walk signal carries information the diffusion KL does not.
        #
        # The destination node must always remain unmasked so its log-prob
        # is well-defined.  We force-include it by temporarily boosting its
        # logit before top-K selection, then restoring the original value.
        pool_size = logits.size(1)
        K = self.walk_topk
        if 0 < K < pool_size:
            # Save destination logits.
            dst_logits_orig = logits.gather(
                1, flat_dst.unsqueeze(1)
            ).squeeze(1)   # [n_valid]

            # Temporarily set dst logits to +inf so they survive top-K.
            logits.scatter_(1, flat_dst.unsqueeze(1), float("inf"))

            # Keep only top-K; mask the rest.
            topk_vals, _ = logits.topk(K, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(1)  # [n_valid, 1]
            mask = logits < threshold
            logits = logits.masked_fill(mask, float("-inf"))

            # Restore original destination logits.
            logits.scatter_(1, flat_dst.unsqueeze(1),
                            dst_logits_orig.unsqueeze(1))

        log_probs = F.log_softmax(logits, dim=-1)  # [n_valid, pool_size]

        # Pick out the log probability of the actual next node.
        step_log_prob = log_probs.gather(
            1, flat_dst.unsqueeze(1)
        ).squeeze(1)  # [n_valid]

        walk_nll = -step_log_prob.mean()

        # Effective denominator size for diagnostics.
        eff_denom = float((logits > float("-inf")).float().sum(dim=-1).mean().item())

        metrics = {
            "walk_nll": float(walk_nll.detach().item()),
            "walk_valid_steps": float(n_valid),
            "walk_valid_ratio": float(n_valid / max(1, batch_size * num_walks * walk_length)),
            "walk_eff_denom": eff_denom,
        }
        return walk_nll, metrics

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        teacher_probs: torch.Tensor,
        candidate_idx: torch.Tensor | None = None,
        anchor_idx: torch.Tensor | None = None,
        walk_paths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
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
        if self.direct_active:
            direct = self._direct_target(anchor_idx, column_idx, self_mask, share)
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

        # ---- Compute Cost Matrix for Sinkhorn OT if enabled ----
        if self.use_sinkhorn:
            if self.share_in_batch:
                # pool_norm is [pool_size, D]
                # C is [pool_size, pool_size]
                ot_C = 1 - torch.matmul(pool_norm, pool_norm.t())
                ot_C = ot_C.clamp_min(0.0)
            else:
                # candidate_norm is [B, K, D]
                # C is [B, K, K]
                ot_C = 1 - torch.einsum("bid,bjd->bij", candidate_norm, candidate_norm)
                ot_C = ot_C.clamp_min(0.0)
        else:
            ot_C = None
        # --------------------------------------------------------

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
            
            if self.use_sinkhorn:
                student_probs = torch.exp(log_probs)
                # Apply Sinkhorn OT using precomputed ot_C
                ot_dist = self._batched_sinkhorn(ot_C, scale_target, student_probs)
                kl_per_scale.append(ot_dist)
            else:
                # KL Divergence
                contribution = torch.where(
                    scale_target > 0,
                    scale_target * (log_target[:, scale_idx, :] - log_probs),
                    torch.zeros_like(scale_target),
                )
                kl_per_scale.append(contribution.sum(dim=-1))
        kl_per_scale = torch.stack(kl_per_scale, dim=1)
        loss_diff = (kl_per_scale * weights.view(1, -1)).sum(dim=-1).mean()
        _assert_finite_tensors((("loss_diff", loss_diff),))
        
        # ---- CoSENT Auxiliary Loss ----
        loss_cosent = torch.tensor(0.0, device=anchor_embeddings.device)
        if getattr(self, "use_cosent", False):
            T_anchor = self.teacher_bank[anchor_idx].to(anchor_embeddings.device, non_blocking=True)
            if self.share_in_batch:
                T_pool = self.teacher_bank[column_idx].to(anchor_embeddings.device, non_blocking=True)
                T_sim = torch.matmul(T_anchor, T_pool.t())
            else:
                T_cand = self.teacher_bank[column_idx].to(anchor_embeddings.device, non_blocking=True)
                T_sim = torch.einsum("bid,bjd->bij", T_anchor.unsqueeze(1), T_cand).squeeze(1)

            cos = similarity / self.cosent_tau # [B, K]
            A = cos.unsqueeze(1) - cos.unsqueeze(2) # [B, K, K]
            
            # M[b, j, k] = True if T_sim[b, j] > T_sim[b, k] + delta
            M = T_sim.unsqueeze(2) > (T_sim.unsqueeze(1) + self.cosent_delta)
            
            # Also apply diffusion_mask so we don't penalize masked out items
            valid_mask = (~diffusion_mask).unsqueeze(1) & (~diffusion_mask).unsqueeze(2)
            M = M & valid_mask
            
            A = A.masked_fill(~M, float('-inf'))
            flat_A = A.view(A.size(0), -1)
            zeros = torch.zeros(A.size(0), 1, device=A.device)
            loss_cosent = torch.logsumexp(torch.cat([zeros, flat_A], dim=1), dim=1).mean()
        # -------------------------------
        
        # ---- Random Walk Trajectory Loss ----
        loss_walk = torch.tensor(0.0, device=anchor_embeddings.device)
        walk_metrics: dict[str, float] = {}
        if (
            getattr(self, "use_walk_loss", False)
            and walk_paths is not None
            and share
            and (walk_paths >= 0).any()
        ):
            loss_walk, walk_metrics = self._compute_walk_loss(
                walk_paths=walk_paths,
                pool_norm=pool_norm,
                column_idx=column_idx,
                shared=share,
            )
        # -----------------------------------------

        # The diffusion term is the whole objective now. `loss_total` stays a
        # separate logged metric so runs from before the anchor term was dropped
        # remain comparable on the same dashboard key.
        total_loss = loss_diff
        if getattr(self, "use_cosent", False):
            total_loss = total_loss + self.cosent_weight * loss_cosent
        if getattr(self, "use_walk_loss", False) and loss_walk.item() > 0:
            total_loss = total_loss + self.walk_weight * loss_walk

        metrics = self._diagnostics(
            total_loss=total_loss,
            loss_diff=loss_diff,
            loss_cosent=loss_cosent,
            kl_per_scale=kl_per_scale,
            log_probs_per_scale=log_probs_per_scale,
            target=target,
            target_entropy=target_entropy,
            weights=weights,
            self_mask=self_mask,
            diffusion_mask=diffusion_mask,
            temps=temps,
        )
        return total_loss, metrics

    @torch.no_grad()
    def _diagnostics(
        self,
        total_loss: torch.Tensor,
        loss_diff: torch.Tensor,
        loss_cosent: torch.Tensor,
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
        
        if getattr(self, "use_cosent", False):
            metrics["loss_cosent"] = float(loss_cosent.detach().item())

        if walk_metrics:
            metrics.update(walk_metrics)
            metrics["loss_walk"] = float(loss_walk.detach().item())

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
