from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.heatgeo.policy import (
    DIAG_TOPK,
    EPS_NORM,
    FIXED_BANDWIDTH_TEMP,
    diffusion_weights,
)


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

    **Support-mass calibration.** Conditional KL on a selected support C_i does not
    determine how much of the student's full-corpus probability belongs to C_i. For
    every diffusion scale the sampler therefore retains

        alpha_T = sum_{j in C_i} p_T(j)

    before normalizing the conditional target. Hard and uniform ambient samples
    carry inverse inclusion probabilities, giving a Horvitz--Thompson estimate of
    the complement partition. With Z_C scored exactly and Z_bar estimated from the
    ambient sample, alpha_S = Z_C / (Z_C + Z_bar) and

        L_mass = sum_r omega_r KL(Bern(alpha_T,r) || Bern(alpha_S,r)).

    This is the first term in the exact KL chain rule for the partition {C_i,
    complement}; the conditional relational loss controls the within-support term,
    while candidate selection makes the unresolved complement coefficient small.

    **Temperature ties.** The temperatures are not free parameters and are
    therefore not constructor arguments:

    * ``tau_1 = graph_temp``: the r=1 target after dropping self-mass IS the
      transition row, a softmax of teacher cosines at graph_temp. Matching it at
      the same temperature makes zero loss attainable exactly on the shift family
      cos_S = cos_T + a_i; any other temperature forces the affine family
      cos_S = (tau_1/graph_temp) cos_T + a_i, which contradicts the direct scale
      on the same row (it pins unrescaled gaps), so the joint zero set is empty
      unless the teacher cosines are constant.

      With ``row_temps`` supplied -- the entropic-affinity graph, where each row
      is solved for a fixed perplexity instead of sharing one temperature -- the
      tie is read row by row: row i's target was built at tau_i, so row i is
      matched at tau_i. Nothing in the argument above depends on the *value* of
      the temperature, only on the two sides sharing it, and the attainable set is
      the same shift family; so the proposition holds per row.
    * ``direct student temp = direct_temp``: the classic same-temperature
      convention of distillation (Hinton et al., 2015). Unequal temperatures
      make the direct target attainable only as a rescaling of teacher cosines,
      which is the calibration distortion the scale exists to prevent.

    * ``tau_r = sqrt(r) * tau_1`` for the broader diffusion scales: the spread of
      a diffusion grows as sqrt of its time, so scale r is matched at the
      resolution its own target already has. This is a stated rule, not a
      derivation -- the lazy walk takes r/2 real steps in expectation, so a
      strict derivation would carry a different constant -- but it removes the
      last free student temperatures, and under ``row_temps`` it inherits the
      per-row tie automatically: tau_r(i) = sqrt(r) * tau_i.

    **Scale weights.** ``omega_r = 1/r`` with ``omega_0 = omega_1``: a scale's
    influence falls off with the diffusion time it describes, and the ambient scale
    sits at the sharpest scale's weight. The full vector is derived from
    ``diffusion_scales`` and normalized together once in ``forward``.

    Passing removed or derived knobs raises rather than being silently absorbed.
    """

    _TIED_KNOBS = {
        "scale_weights": "derived from diffusion_scales by omega_r = 1/r",
        "direct_weight": "derived from omega_0 = omega_1",
        "share_in_batch": "in-batch sharing is always used when corpus indices exist",
        "graph_temp": (
            "canonical rows use their stored bandwidths; the scalar fixed-bandwidth "
            "baseline is an internal constant"
        ),
        "scale_temps": (
            "the whole ladder is derived: tau_1 is tied to graph_temp (or to the "
            "per-row bandwidth) and tau_r = sqrt(r) * tau_1"
        ),
        "walk_temp": "row supervision was removed in favor of support-mass calibration",
        "walk_weight": "use mass_weight; row supervision no longer exists",
        "transition_neighbors": "transition rows are not consumed by the loss",
        "transition_probs": "transition rows are not consumed by the loss",
        "direct_student_temp": (
            "tied to direct_temp: the direct scale uses one temperature on both "
            "the teacher and the student side (Hinton et al., 2015)"
        ),
        "broad_scale_temps": (
            "tied to the sharpest scale by tau_r = sqrt(r) * tau_1; pass "
            "diffusion_scales instead and the ladder is derived from it"
        ),
    }

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        diffusion_scales: Sequence[int] | None = None,
        teacher_embeddings: torch.Tensor | None = None,
        direct_temp: float = 0.10,
        mass_weight: float = 0.5,
        geo_weight: float = 0.0,
        sym_weight: float = 0.0,
        row_temps: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__()
        for name, why in self._TIED_KNOBS.items():
            if name in kwargs:
                raise ValueError(
                    f"HeatGeoDistillation no longer accepts {name!r}: {why}"
                )
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.graph_temp = FIXED_BANDWIDTH_TEMP

        # Per-row temperatures (entropic affinities). The tie tau_1 =
        # "the temperature this row's target was built at" is unchanged; it is
        # simply no longer the same number for every row. Zero loss is still
        # attainable exactly on the shift family cos_S(i,.) = cos_T(i,.) + a_i,
        # which is independent of tau_i, so the attainability result carries over
        # row by row.
        if row_temps is not None:
            temps_row = row_temps.detach().to(torch.float32).reshape(-1)
            if not bool(torch.isfinite(temps_row).all()) or bool(
                (temps_row <= 0).any()
            ):
                raise ValueError("row_temps must be finite and positive")
            self.register_buffer("row_temps", temps_row, persistent=False)
        else:
            self.row_temps = None
        self.eps_norm = EPS_NORM
        self.diag_topk = DIAG_TOPK
        if mass_weight < 0.0:
            raise ValueError("mass_weight must be non-negative")
        if geo_weight < 0.0:
            raise ValueError("geo_weight must be non-negative")
        if sym_weight < 0.0:
            raise ValueError("sym_weight must be non-negative")
        self.mass_weight = float(mass_weight)
        self.geo_weight = float(geo_weight)
        self.sym_weight = float(sym_weight)

        self.use_direct = teacher_embeddings is not None
        if self.use_direct:
            if direct_temp <= 0.0:
                raise ValueError("direct_temp must be positive")
            # Stored normalized and in half precision: the only operation it feeds is
            # a cosine, and at corpus scale this buffer is the largest thing the
            # criterion owns (N x 2560).
            normalized = F.normalize(
                teacher_embeddings.float(), p=2, dim=-1, eps=self.eps_norm
            ).half()
            self.register_buffer("teacher_bank", normalized, persistent=False)
            # One temperature for both sides of the direct scale (same-temperature
            # distillation, Hinton et al. 2015): the student softmax at scale 0 in
            # forward() reuses this exact value.
            self.direct_temp = float(direct_temp)
        else:
            self.teacher_bank = None
            self.direct_temp = float(direct_temp)

        # Weights stay unnormalized until the ambient and diffusion entries have
        # been concatenated. This preserves omega_0 = omega_1 exactly.
        # The whole diffusion ladder is derived from the sharpest scale by
        # tau_r = sqrt(r) * tau_1. The sharpest scale is itself tied (its target IS
        # the transition row), so no student temperature on this ladder is free.
        # `scale_sqrt` holds sqrt(r / r_sharpest) so it is 1.0 at the sharpest scale
        # and multiplies whichever tau_1 applies -- graph_temp, or the per-row tau_i
        # of the entropic-affinity graph. `_resolved` pads by repeating the last
        # entry when the artifact carries more scales than were declared.
        if diffusion_scales is None or len(tuple(diffusion_scales)) == 0:
            scales = (1,)
        else:
            scales = tuple(int(r) for r in diffusion_scales)
        if min(scales) < 1:
            raise ValueError(f"diffusion_scales must be >= 1, got {scales}")
        # Two conditions the rest of the class assumes and nothing used to enforce.
        #
        # *Sorted and unique.* The artifact stores its scales through `_as_tuple`,
        # which sorts and deduplicates, so `pool_probs[r]` is indexed by rank in the
        # sorted order. This class indexes `scale_sqrt` and `scale_weights` by
        # position in the tuple as given. Pass (4, 1, 2) and the artifact holds
        # (1, 2, 4) while the criterion matches them at tau = (tau_1, tau_1/2,
        # tau_1/sqrt(2)) and weights them by 1/4, 1, 1/2 -- every scale supervised
        # at another scale's temperature, silently.
        #
        # *Sharpest scale is r = 1.* The whole temperature ladder hangs off
        # tau_1 = "the bandwidth this row's target was built at", which holds
        # because the r=1 lazy-walk target after dropping self-mass IS the
        # transition row. At r >= 2 it is not, so the tie is simply false and
        # tau_r = sqrt(r/r_0) tau_1 is anchored to nothing.
        if scales != tuple(sorted(set(scales))):
            raise ValueError(
                f"diffusion_scales must be sorted and unique, got {scales}: the "
                f"artifact stores them sorted, so any other order matches each "
                f"scale at another scale's temperature and weight"
            )
        if scales[0] != 1:
            raise ValueError(
                f"diffusion_scales must start at 1, got {scales}: the temperature "
                f"tie tau_1 = graph bandwidth holds because the r=1 target is the "
                f"transition row, and the whole ladder is derived from it"
            )
        self.diffusion_scales = scales
        weights = torch.tensor(diffusion_weights(scales), dtype=torch.float32)
        self.register_buffer("scale_weights", weights)
        # The ambient profile has the same unnormalized weight as r=1.
        self.register_buffer("direct_weight", weights[:1].clone())
        sqrt_r = torch.tensor(
            [(r / scales[0]) ** 0.5 for r in scales], dtype=torch.float32
        )
        self.register_buffer("scale_sqrt", sqrt_r)
        temps = self.graph_temp * sqrt_r
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

    def _support_mass_loss(
        self,
        similarity: torch.Tensor,
        teacher_mass: torch.Tensor,
        support_mask: torch.Tensor,
        ambient_importance: torch.Tensor,
        diffusion_temps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Bernoulli KL for mass on C, using a stratified complement estimate."""
        logits = similarity.unsqueeze(1) / diffusion_temps.unsqueeze(-1)
        support = support_mask.unsqueeze(1)
        ambient = ambient_importance > 0

        log_z_support = torch.logsumexp(
            logits.masked_fill(~support, float("-inf")), dim=-1
        )
        log_weights = torch.where(
            ambient,
            ambient_importance.clamp_min(1e-12).log(),
            torch.full_like(ambient_importance, float("-inf")),
        )
        log_z_complement = torch.logsumexp(
            logits + log_weights.unsqueeze(1), dim=-1
        )

        valid = support_mask.any(dim=-1) & ambient.any(dim=-1)
        mass_logit = torch.where(
            valid.unsqueeze(1),
            log_z_support - log_z_complement,
            torch.zeros_like(log_z_support),
        )
        alpha_teacher = teacher_mass.clamp(0.0, 1.0)
        zeros = torch.zeros_like(alpha_teacher)
        teacher_entropy = -(
            torch.where(
                alpha_teacher > 0,
                alpha_teacher * alpha_teacher.clamp_min(1e-12).log(),
                zeros,
            )
            + torch.where(
                alpha_teacher < 1,
                (1 - alpha_teacher)
                * (1 - alpha_teacher).clamp_min(1e-12).log(),
                zeros,
            )
        )
        mass_kl = F.binary_cross_entropy_with_logits(
            mass_logit, alpha_teacher, reduction="none"
        ) - teacher_entropy
        mass_kl = mass_kl.clamp_min(0.0) * valid.unsqueeze(1)

        weights = self._resolved(self.scale_weights, teacher_mass.size(1))
        weights = weights / weights.sum().clamp_min(1e-12)
        loss = (mass_kl * weights.view(1, -1)).sum(dim=-1).mean()
        alpha_student = mass_logit.sigmoid()
        valid_scale = valid.unsqueeze(1).expand_as(alpha_student)
        valid_count = valid_scale.sum().clamp_min(1)
        metrics = {
            "teacher_support_mass": alpha_teacher.mean(),
            "student_support_mass": (alpha_student * valid_scale).sum()
            / valid_count,
            "support_mass_abs_error": (
                (alpha_teacher - alpha_student).abs() * valid_scale
            ).sum()
            / valid_count,
            "mass_valid_ratio": valid.float().mean(),
        }
        return loss, metrics

    def _selected_geometry_losses(
        self,
        similarity: torch.Tensor,
        teacher_probs: torch.Tensor,
        candidate_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute directed and unique-edge cosine geometry objectives.

        Diffusion scales are first merged with the canonical ``1/r`` weights.
        ``L_geo`` is the theorem-aligned conditional expectation of bounded cosine
        distortion on each anchor's selected support. ``L_sym`` groups the same
        selected directed relations by unordered corpus edge, adds the two available
        directional masses, and scores every unique edge once.
        """
        zero = similarity.new_zeros(())
        if self.teacher_bank is None:
            return zero, zero, {
                "geo_relations": zero,
                "sym_edges": zero,
            }

        scale_weights = self._resolved(
            self.scale_weights, teacher_probs.size(1)
        ).to(teacher_probs.dtype)
        scale_weights = scale_weights / scale_weights.sum().clamp_min(1e-12)
        selected_mass = (
            teacher_probs * scale_weights.view(1, -1, 1)
        ).sum(dim=1)
        self_mask = candidate_idx == anchor_idx.view(-1, 1)
        selected_mass = selected_mass.masked_fill(self_mask, 0.0)
        selected_distribution = selected_mass / selected_mass.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        selected = selected_mass > 0

        teacher_anchor = self.teacher_bank.index_select(0, anchor_idx).float()
        teacher_candidate = self.teacher_bank.index_select(
            0, candidate_idx.reshape(-1)
        ).float()
        teacher_candidate = teacher_candidate.view(*candidate_idx.shape, -1)
        teacher_similarity = torch.einsum(
            "bd,bcd->bc", teacher_anchor, teacher_candidate
        )
        squared_error = (similarity - teacher_similarity).square()

        # ell_ij = (cos_S(i,j) - cos_T(i,j))^2 / 4 lies in [0, 1].
        loss_geo = zero
        if self.geo_weight > 0.0:
            loss_geo = (
                selected_distribution * squared_error / 4.0
            ).sum(dim=-1).mean()

        # A geo-only run must not pay for unique-edge construction (or execute
        # scatter kernels that cannot affect its objective).
        if self.sym_weight <= 0.0:
            return loss_geo, zero, {
                "geo_relations": selected.float().sum(),
                "sym_edges": zero,
            }

        flat_selected = selected.reshape(-1)
        directed_anchor = anchor_idx.view(-1, 1).expand_as(candidate_idx).reshape(-1)
        directed_candidate = candidate_idx.reshape(-1)
        edge_left = torch.minimum(directed_anchor, directed_candidate)[flat_selected]
        edge_right = torch.maximum(directed_anchor, directed_candidate)[flat_selected]
        n_items = int(self.teacher_bank.size(0))
        edge_key = edge_left * n_items + edge_right
        unique_key, inverse = torch.unique(edge_key, sorted=False, return_inverse=True)
        n_edges = unique_key.numel()

        directed_weight = selected_distribution.reshape(-1)[flat_selected]
        directed_similarity = similarity.reshape(-1)[flat_selected]
        edge_weight = similarity.new_zeros(n_edges)
        edge_similarity = similarity.new_zeros(n_edges)
        edge_count = similarity.new_zeros(n_edges)
        edge_weight.scatter_add_(0, inverse, directed_weight)
        edge_similarity.scatter_add_(0, inverse, directed_similarity)
        edge_count.scatter_add_(0, inverse, torch.ones_like(directed_similarity))
        edge_similarity = edge_similarity / edge_count.clamp_min(1.0)

        unique_left = torch.div(unique_key, n_items, rounding_mode="floor")
        unique_right = unique_key.remainder(n_items)
        teacher_edge_similarity = (
            self.teacher_bank.index_select(0, unique_left).float()
            * self.teacher_bank.index_select(0, unique_right).float()
        ).sum(dim=-1)
        # w_ij = 1/2 [p_i(j|C_i) + p_j(i|C_j)]. A direction absent from the
        # current selected union contributes zero; reciprocal edges are merged.
        edge_weight = 0.5 * edge_weight
        loss_sym = (
            edge_weight * (edge_similarity - teacher_edge_similarity).square()
        ).sum() / max(1, n_edges)

        return loss_geo, loss_sym, {
            "geo_relations": selected.float().sum(),
            "sym_edges": similarity.new_tensor(float(n_edges)),
        }

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        teacher_probs: torch.Tensor,
        ambient_importance: torch.Tensor | None = None,
        candidate_idx: torch.Tensor | None = None,
        anchor_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        named_tensors = [
            ("anchor_embeddings", anchor_embeddings),
            ("candidate_embeddings", candidate_embeddings),
            ("teacher_probs", teacher_probs),
        ]
        if ambient_importance is not None:
            named_tensors.append(("ambient_importance", ambient_importance))
        _assert_finite_tensors(named_tensors)

        batch_size = anchor_embeddings.size(0)
        candidate_size = teacher_probs.size(-1)
        n_scales = teacher_probs.size(1)

        teacher_probs = teacher_probs.clamp_min(0.0)
        absolute_teacher_probs = teacher_probs
        teacher_support_mass = teacher_probs.sum(dim=-1)
        support_mask = teacher_probs.sum(dim=1) > 0
        teacher_probs = teacher_probs / teacher_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1, eps=self.eps_norm)
        candidate_embeddings = candidate_embeddings.reshape(
            batch_size, candidate_size, -1
        )
        candidate_norm_own = F.normalize(
            candidate_embeddings, p=2, dim=-1, eps=self.eps_norm
        )
        similarity_own = torch.einsum("bd,bcd->bc", anchor_norm, candidate_norm_own)
        candidate_embeddings = candidate_embeddings.reshape(
            batch_size * candidate_size, -1
        )

        share = candidate_idx is not None and anchor_idx is not None
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
            similarity = similarity_own
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
            weights = torch.cat([self.direct_weight.to(weights.dtype), weights])
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

        # The sharpest diffusion scale is the one whose target IS the anchor's
        # transition row, so it is the scale the temperature tie binds. With
        # entropic affinities that temperature is per anchor, not global.
        offset = 1 if self.direct_active else 0
        row_tau = None
        if self.row_temps is not None and anchor_idx is not None:
            row_tau = self.row_temps.index_select(0, anchor_idx).view(-1, 1)
        # tau_r(i) = sqrt(r) * tau_i: the tie at the sharpest scale propagates up the
        # ladder, so with per-row bandwidths every diffusion scale is per-row too.
        sqrt_r = self._resolved(self.scale_sqrt, n_scales - offset)

        loss_mass = torch.zeros((), device=anchor_embeddings.device)
        mass_metrics = {
            "teacher_support_mass": loss_mass,
            "student_support_mass": loss_mass,
            "support_mass_abs_error": loss_mass,
            "mass_valid_ratio": loss_mass,
        }
        if ambient_importance is not None:
            if row_tau is not None:
                diffusion_temps = row_tau * sqrt_r.view(1, -1)
            else:
                diffusion_temps = self._resolved(
                    self.scale_temps, teacher_support_mass.size(1)
                ).view(1, -1)
            loss_mass, mass_metrics = self._support_mass_loss(
                similarity=similarity_own,
                teacher_mass=teacher_support_mass,
                support_mask=support_mask,
                ambient_importance=ambient_importance,
                diffusion_temps=diffusion_temps,
            )

        loss_geo = torch.zeros((), device=anchor_embeddings.device)
        loss_sym = torch.zeros((), device=anchor_embeddings.device)
        geometry_metrics = {
            "geo_relations": loss_geo,
            "sym_edges": loss_sym,
        }
        if (
            (self.geo_weight > 0.0 or self.sym_weight > 0.0)
            and candidate_idx is not None
            and anchor_idx is not None
        ):
            loss_geo, loss_sym, geometry_metrics = self._selected_geometry_losses(
                similarity=similarity_own,
                teacher_probs=absolute_teacher_probs,
                candidate_idx=candidate_idx,
                anchor_idx=anchor_idx,
            )

        kl_per_scale = []
        log_probs_per_scale = []
        for scale_idx in range(n_scales):
            is_direct = self.direct_active and scale_idx == 0
            if row_tau is not None and not is_direct:
                logits = similarity / (row_tau * sqrt_r[scale_idx - offset])
            else:
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
        total_loss = (
            loss_diff
            + self.mass_weight * loss_mass
            + self.geo_weight * loss_geo
            + self.sym_weight * loss_sym
        )
        _assert_finite_tensors(
            (
                ("loss_diff", loss_diff),
                ("loss_mass", loss_mass),
                ("loss_geo", loss_geo),
                ("loss_sym", loss_sym),
                ("total_loss", total_loss),
            )
        )

        metrics = self._diagnostics(
            total_loss=total_loss,
            loss_diff=loss_diff,
            loss_mass=loss_mass,
            loss_geo=loss_geo,
            loss_sym=loss_sym,
            mass_metrics=mass_metrics,
            geometry_metrics=geometry_metrics,
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
        loss_mass: torch.Tensor,
        loss_geo: torch.Tensor,
        loss_sym: torch.Tensor,
        mass_metrics: dict[str, torch.Tensor],
        geometry_metrics: dict[str, torch.Tensor],
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
            loss_mass.detach(),
            (self.mass_weight * loss_mass).detach(),
            loss_geo.detach(),
            (self.geo_weight * loss_geo).detach(),
            loss_sym.detach(),
            (self.sym_weight * loss_sym).detach(),
            geometry_metrics["geo_relations"].detach(),
            geometry_metrics["sym_edges"].detach(),
            mass_metrics["teacher_support_mass"].detach(),
            mass_metrics["student_support_mass"].detach(),
            mass_metrics["support_mass_abs_error"].detach(),
            mass_metrics["mass_valid_ratio"].detach(),
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
            "loss_mass",
            "loss_mass_weighted",
            "loss_geo",
            "loss_geo_weighted",
            "loss_sym",
            "loss_sym_weighted",
            "geo_relations",
            "sym_edges",
            "teacher_support_mass",
            "student_support_mass",
            "support_mass_abs_error",
            "mass_valid_ratio",
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
                self.row_temps is None
                and (
                    temps.numel() - offset <= 1
                    or torch.allclose(temps[offset:], temps[offset])
                )
            )
        )
        return metrics
