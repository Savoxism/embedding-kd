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

    **The walk term, and what it actually supervises.** The diffusion scales only
    ever condition on an *anchor*: every row of the teacher operator they match is a
    row indexed by a batch anchor. The rows at every other node are never seen, even
    though those nodes are already encoded as candidates. The walk term samples such
    nodes -- by running a random walk on the teacher transition matrix P, so the
    sampling measure is the walk's occupancy around the anchor rather than uniform --
    and matches the teacher's one-step kernel there:

        L_walk = E_{j ~ nu} KL( Ptilde(.|j) || p^S(.|j) ),
        Ptilde(.|j) = P(.|j) restricted to pool ∩ supp P(.|j), renormalized.

    Three properties are deliberate. *The target is dense, not a sampled successor.*
    A first-order walk scored with teacher forcing factorizes into independent
    one-step terms, so a sampled destination is an unbiased but high-variance
    estimate of exactly this KL -- the transition row is already stored, so there is
    no reason to pay the variance. It also means the term is not a trajectory
    objective and is not described as one: nothing in it couples two consecutive
    steps.

    *The denominator is the teacher's own support, not the shared pool.* Softmaxing a
    walk step over the whole pool would recreate the false-zero gradient the column
    domains above exist to remove, and would do it on the worst possible columns: a
    hard negative is by construction top-200 by teacher cosine and excluded from the
    mutual-kNN graph, so it can never appear in supp P(.|j) and would be pushed down
    with nothing pulling it back. Restricted to pool ∩ supp P(.|j), every column in
    the denominator carries real teacher mass, and the term has a floor of 0 -- which
    also puts it on the same scale as L_diff, so walk_weight means something.

    *The walk's first node is dropped.* Step 0 is the anchor, and the lazy walk at
    r=1 is (e_i + P_i)/2, which after dropping the self-mass is P_i exactly. Scoring
    it again would distil the r=1 target twice, once densely and once by sampling.

    The three column domains are therefore disjoint in what they claim: diffusion
    ranks within the anchor's own draw, the direct scale calibrates across the batch,
    and the walk term ranks within a *non-anchor* node's teacher neighbourhood.

    **The temperature law.** No per-scale temperatures are accepted; the whole
    ladder is generated by ``graph_temp`` and one exponent:

        tau_r = graph_temp * r**temp_exponent       (diffusion scales)
        tau_0 = graph_temp * r_max**temp_exponent   (direct scale, both sides)
        tau_w = graph_temp                          (walk term)

    Provable pieces (the ties): the r=1 target after dropping self-mass IS the
    transition row -- a softmax of teacher cosines at graph_temp -- and the walk
    targets are transition rows too, so matching either at any other temperature
    makes zero loss unattainable and contradicts the direct scale on the same
    rows. Under the law, r=1 recovers tau_1 = graph_temp for *every* exponent:
    the tie is the law's base case. The direct scale uses one temperature on
    both sides (Hinton et al., 2015) and enters the ladder at its top, since it
    calibrates over the full pooled column set, the broadest comparison in the
    objective.

    Law-shaped piece: the heat kernel on the unit sphere is a softmax in cosine
    similarity at temperature 2t (via d^2 = 2 - 2 cos), and lazy-walk diffusion
    time grows linearly in r, so ``temp_exponent = 1`` is the heat-kernel value
    (exact when the student embeds the manifold isometrically and the kernel is
    in its Gaussian regime), and it is the default: the exponent is fixed by
    theory, not tuned per run. ``temp_exponent = 0.5`` reproduces the
    historical hand-tuned ladder up to a ~1% shift of tau_2 and is kept
    reachable only for comparison runs.

    The direct scale's pre-normalization weight is fixed at 1 -- it is one
    member of the ladder, not an optional term; ablate it by passing
    ``teacher_embeddings=None``. Passing any removed knob (``scale_temps``,
    ``walk_temp``, ``direct_student_temp``, ``broad_scale_temps``,
    ``direct_temp``, ``direct_weight``, ``student_temp``) raises rather than
    being silently absorbed.
    """

    _REMOVED_KNOBS = {
        "scale_temps": (
            "per-scale temperatures are generated by the law "
            "tau_r = graph_temp * r**temp_exponent"
        ),
        "broad_scale_temps": (
            "per-scale temperatures are generated by the law "
            "tau_r = graph_temp * r**temp_exponent"
        ),
        "walk_temp": (
            "tied to graph_temp: the walk target is a transition row, i.e. a "
            "softmax of teacher cosines at graph_temp"
        ),
        "direct_student_temp": (
            "the direct scale uses one temperature on both the teacher and the "
            "student side (Hinton et al., 2015)"
        ),
        "direct_temp": (
            "derived: tau_0 = graph_temp * r_max**temp_exponent, shared by both "
            "sides of the direct scale"
        ),
        "direct_weight": (
            "fixed at 1.0; the direct scale is one member of the ladder, not an "
            "optional term -- ablate it by passing teacher_embeddings=None"
        ),
        "student_temp": (
            "no longer a fallback: the temperature law covers every scale"
        ),
    }

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        scale_weights: Sequence[float],
        diffusion_scales: Sequence[int] = (1, 2, 4),
        temp_exponent: float = 1.0,
        eps_norm: float = 1e-8,
        diag_topk: int = 8,
        share_in_batch: bool = True,
        teacher_embeddings: torch.Tensor | None = None,
        transition_neighbors: torch.Tensor | None = None,
        transition_probs: torch.Tensor | None = None,
        walk_weight: float = 0.5,
        walk_topk: int | None = None,
        *,
        graph_temp: float,
        **kwargs,
    ):
        super().__init__()
        for name, why in self._REMOVED_KNOBS.items():
            if name in kwargs:
                raise ValueError(
                    f"HeatGeoDistillation no longer accepts {name!r}: {why}"
                )
        if graph_temp <= 0.0:
            raise ValueError("graph_temp must be positive")
        if temp_exponent < 0.0:
            raise ValueError(
                "temp_exponent must be >= 0: temperatures cannot sharpen as the "
                "diffusion scale broadens"
            )
        scales = tuple(sorted({int(r) for r in diffusion_scales}))
        if not scales or scales[0] < 1:
            raise ValueError(
                f"diffusion_scales must be positive integers, got {diffusion_scales!r}"
            )
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.graph_temp = float(graph_temp)
        self.temp_exponent = float(temp_exponent)
        self.diffusion_scales = scales
        self.eps_norm = eps_norm
        self.diag_topk = diag_topk
        self.share_in_batch = share_in_batch

        # ---- Random walk kernel matching ----------------------------------------
        # Enabled by the curriculum, not by construction: the walk term ranks inside
        # a neighbourhood, which is only meaningful once the student has one.
        self.use_walk_loss = False
        self.walk_weight = float(walk_weight)
        # Tied, not tuned: the walk target is a transition row = softmax of teacher
        # cosines at graph_temp (see "Temperature ties" in the class docstring).
        self.walk_temp = self.graph_temp
        self.walk_topk = None if walk_topk is None else int(walk_topk)
        if self.walk_topk is not None and self.walk_topk <= 0:
            raise ValueError("walk_topk must be positive when provided")
        if transition_neighbors is not None and transition_probs is not None:
            # The teacher's transition rows, padded with -1. Held as int32/float32
            # rather than the artifact's int64: at corpus scale this is the second
            # largest buffer the criterion owns and every use gathers a few hundred
            # rows out of it.
            self.register_buffer(
                "walk_neighbors", transition_neighbors.to(torch.int32), persistent=False
            )
            self.register_buffer(
                "walk_probs", transition_probs.to(torch.float32), persistent=False
            )
        else:
            self.walk_neighbors = None
            self.walk_probs = None
        self._warned_walk_needs_sharing = False

        # tau_0 = graph_temp * r_max^alpha: the direct scale sits at the top of
        # the ladder -- it calibrates over the full pooled column set, the
        # broadest comparison in the objective -- and uses this one temperature
        # on both the teacher and the student side (Hinton et al. 2015).
        self.direct_temp = self.graph_temp * float(max(scales)) ** self.temp_exponent
        self.use_direct = teacher_embeddings is not None
        if self.use_direct:
            # Stored normalized and in half precision: the only operation it feeds is
            # a cosine, and at corpus scale this buffer is the largest thing the
            # criterion owns (N x 2560).
            normalized = F.normalize(
                teacher_embeddings.float(), p=2, dim=-1, eps=eps_norm
            ).half()
            self.register_buffer("teacher_bank", normalized, persistent=False)
        else:
            self.teacher_bank = None

        weights = torch.tensor(list(scale_weights), dtype=torch.float32)
        if weights.numel() == 0:
            weights = torch.ones(1, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-12)
        self.register_buffer("scale_weights", weights)

        # tau_r = graph_temp * r^alpha, aligned with the artifact's sorted scale
        # order. r = 1 gives tau_1 = graph_temp exactly: the provable tie is the
        # law's base case, for every exponent. graph_temp > 0, r >= 1 and
        # alpha >= 0 make every entry positive by construction.
        temps = torch.tensor(
            [self.graph_temp * float(r) ** self.temp_exponent for r in scales],
            dtype=torch.float32,
        )
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
        scatter_index = inverse.view(batch_size, 1, candidate_size).expand(
            batch_size, n_scales, candidate_size
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

    def _compute_walk_loss(
        self,
        walk_paths: torch.Tensor,
        pool_norm: torch.Tensor,
        column_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Teacher one-step kernel matched at the nodes a walk visits.

        Args:
            walk_paths: [B, M, L+1] corpus indices; column 0 is the anchor and is
                dropped, -1 marks a disabled or truncated walk.
            pool_norm: [pool_size, D] L2-normalized shared pool embeddings.
            column_idx: [pool_size] corpus indices of the pool columns, sorted.

        Returns:
            walk_kl: scalar, occupancy-weighted mean KL per supervised node.
            metrics: diagnostics, including the ones that say whether the term has
                anything to work with at all.
        """
        device = pool_norm.device
        empty = torch.zeros((), device=device)
        if self.walk_neighbors is None:
            return empty, {}

        pool_size = pool_norm.size(0)

        # Sources: every node visited after the start. Step 0 is the anchor, whose
        # teacher row is already the r=1 diffusion target.
        visited = walk_paths[..., 1:].reshape(-1)
        total_visits = max(int(visited.numel()), 1)
        visited = visited[visited >= 0]
        if visited.numel() == 0:
            return empty, {}

        # A node visited k times is supervised with weight k: that is what makes the
        # sampling measure the walk's occupancy rather than its support.
        src_nodes, src_counts = torch.unique(visited, return_counts=True)

        # column_idx comes out of torch.unique and is therefore sorted, so the
        # corpus -> pool lookup is a binary search on device. The previous version
        # rebuilt a Python dict over the whole pool and walked B*M*(L+1) indices in
        # interpreter code, with two host syncs, on every step.
        src_pos = torch.searchsorted(column_idx, src_nodes).clamp_max(pool_size - 1)
        in_pool = column_idx[src_pos] == src_nodes
        src_nodes = src_nodes[in_pool]
        src_pos = src_pos[in_pool]
        src_counts = src_counts[in_pool]
        if src_nodes.numel() == 0:
            return empty, {"walk_valid_ratio": 0.0, "walk_rows": 0.0}
        visits_in_pool = src_counts.sum()

        # Teacher transition row per source, restricted to columns the pool holds.
        neighbors = self.walk_neighbors.index_select(0, src_nodes).long()
        probs = self.walk_probs.index_select(0, src_nodes).float()
        if self.walk_topk is not None and self.walk_topk < probs.size(1):
            probs, top_positions = torch.topk(
                probs, k=self.walk_topk, dim=1, largest=True, sorted=True
            )
            neighbors = neighbors.gather(1, top_positions)
        flat = neighbors.reshape(-1)
        nb_pos = torch.searchsorted(column_idx, flat.clamp_min(0)).clamp_max(
            pool_size - 1
        )
        present = (flat >= 0) & (column_idx[nb_pos] == flat)
        nb_pos = nb_pos.view_as(neighbors)
        keep = present.view_as(neighbors) & (probs > 0)
        # P carries no self-loops, but nothing downstream would survive one.
        keep = keep & (nb_pos != src_pos.unsqueeze(1))

        target = torch.zeros(src_nodes.numel(), pool_size, device=device)
        target.scatter_add_(
            1, nb_pos, torch.where(keep, probs, torch.zeros_like(probs))
        )
        allowed = target > 0

        # Two live columns is the smallest denominator on which a softmax has any
        # freedom; a one-column row contributes exactly 0 and would only dilute the
        # mean. Dropping it here is what keeps walk_eff_denom honest.
        live = allowed.sum(dim=1)
        usable = live >= 2
        if not bool(usable.any()):
            return empty, {
                "walk_valid_ratio": 0.0,
                "walk_rows": 0.0,
                "walk_node_hit_ratio": float(visits_in_pool.item()) / total_visits,
            }
        target = target[usable]
        allowed = allowed[usable]
        src_pos = src_pos[usable]
        weight = src_counts[usable].float()
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)

        logits = (pool_norm.index_select(0, src_pos) @ pool_norm.t()) / self.walk_temp
        logits = logits.masked_fill(~allowed, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)

        zeros = torch.zeros_like(target)
        log_target = torch.where(target > 0, target.clamp_min(1e-12).log(), zeros)
        kl_rows = torch.where(target > 0, target * (log_target - log_probs), zeros).sum(
            dim=1
        )
        weight = weight / weight.sum().clamp_min(1e-12)
        walk_kl = (kl_rows * weight).sum()

        row_entropy = -(target * log_target).sum(dim=1)
        stats = torch.stack(
            [
                walk_kl.detach(),
                (row_entropy * weight).sum(),
                live[usable].float().mean(),
                usable.sum().float(),
                src_counts[usable].sum().float() / total_visits,
                visits_in_pool.float() / total_visits,
            ]
        ).tolist()
        metrics = dict(
            zip(
                [
                    "walk_kl",
                    "walk_teacher_entropy",
                    "walk_eff_denom",
                    "walk_rows",
                    "walk_valid_ratio",
                    "walk_node_hit_ratio",
                ],
                stats,
            )
        )
        return walk_kl, metrics

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

        share = (
            self.share_in_batch and candidate_idx is not None and anchor_idx is not None
        )
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
        self.direct_active = (
            self.use_direct and anchor_idx is not None and column_idx is not None
        )
        if self.direct_active:
            direct = self._direct_target(anchor_idx, column_idx, self_mask, share)
            target = torch.cat([direct.unsqueeze(1).to(target.dtype), target], dim=1)
            # Pre-normalization weight 1: the direct scale is one member of the
            # ladder, weighted like the sharpest diffusion scale.
            weights = torch.cat([weights.new_ones(1), weights])
            # Same temperature as the teacher side of the direct target: the tie
            # that makes the target attainable rather than a rescaling exercise.
            temps = torch.cat([temps.new_full((1,), self.direct_temp), temps])
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
            logits = logits.masked_fill(
                self_mask if is_direct else diffusion_mask, float("-inf")
            )
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_per_scale.append(log_probs)
            scale_target = target[:, scale_idx, :]
            contribution = torch.where(
                scale_target > 0,
                scale_target * (log_target[:, scale_idx, :] - log_probs),
                torch.zeros_like(scale_target),
            )
            kl_per_scale.append(contribution.sum(dim=-1))
        kl_per_scale = torch.stack(kl_per_scale, dim=1)
        loss_diff = (kl_per_scale * weights.view(1, -1)).sum(dim=-1).mean()
        _assert_finite_tensors((("loss_diff", loss_diff),))

        # ---- Walk term: teacher kernel at non-anchor nodes ----------------------
        # Needs in-batch sharing: without it there is no cross-anchor pool to hold a
        # walk node's neighbours, so every row would fall below the two-column
        # minimum and the term would be silently zero.
        loss_walk = torch.zeros((), device=anchor_embeddings.device)
        walk_metrics: dict[str, float] = {}
        if self.use_walk_loss and walk_paths is not None:
            if share:
                loss_walk, walk_metrics = self._compute_walk_loss(
                    walk_paths=walk_paths,
                    pool_norm=pool_norm,
                    column_idx=column_idx,
                )
            elif not self._warned_walk_needs_sharing:
                self._warned_walk_needs_sharing = True
                print("HeatGeo: walk loss requires share_in_batch=True; term disabled.")

        total_loss = loss_diff + self.walk_weight * loss_walk

        metrics = self._diagnostics(
            total_loss=total_loss,
            loss_diff=loss_diff,
            walk_metrics=walk_metrics,
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
        walk_metrics: dict[str, float],
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
        js_floor = w_total.squeeze(-1) * (mixture_entropy - group_entropy).clamp_min(
            0.0
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

        if walk_metrics:
            metrics.update(walk_metrics)
            metrics["loss_walk"] = self.walk_weight * walk_metrics["walk_kl"]

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
