import numpy as np
import torch

from .policy import (
    DETERMINISTIC_TOPM,
    SUPPORT_POLICIES,
    candidate_budget,
    normalized_diffusion_weights,
)


def _gumbel_topk(probs: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling without replacement (Gumbel top-k)."""
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    support = np.flatnonzero(probs > 0)
    if support.size == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, support.size)
    keys = np.log(probs[support]) + rng.gumbel(size=support.size)
    chosen = np.argpartition(-keys, k - 1)[:k]
    chosen = chosen[np.argsort(-keys[chosen])]
    return support[chosen]


class GGPKDCandidateSampler:
    """Draw the relational candidate set for one anchor.

    The candidate stream is seeded by ``(seed, epoch, idx)``. The rows ``L_row``
    supervises are derived from this draw inside the criterion -- the sampler has
    no row-selection role and reads no transition arrays.
    """

    def __init__(
        self,
        artifact: dict,
        diffusion_quota: int,
        hard_neg_k: int,
        random_neg_k: int,
        seed: int,
        deterministic_topm: int = DETERMINISTIC_TOPM,
        unbiased_geometry: bool = False,
        support_policy: str = "hybrid",
    ):
        self.pool_indices = artifact["pool_indices"].numpy()
        self.pool_probs = artifact["pool_probs"].numpy()
        self.hard_neg_indices = artifact["hard_neg_indices"].numpy()
        self.n_items = int(self.pool_indices.shape[0])
        self.n_scales = int(self.pool_probs.shape[0])

        self.candidate_size = candidate_budget(
            diffusion_quota, hard_neg_k, random_neg_k
        )
        self.diffusion_quota = int(diffusion_quota)
        self.hard_neg_k = int(hard_neg_k)
        self.random_neg_k = int(random_neg_k)
        self.deterministic_topm = int(deterministic_topm)
        self.unbiased_geometry = bool(unbiased_geometry)
        if support_policy not in SUPPORT_POLICIES:
            raise ValueError(
                f"support_policy must be one of {SUPPORT_POLICIES}, "
                f"got {support_policy!r}"
            )
        # The head--tail geometry estimator is defined against the hybrid draw: it
        # needs a deterministic stratum evaluated exactly plus one exact
        # teacher-proportional tail draw. No other policy supplies both, and a
        # silently biased E_hat is worse than none.
        if self.unbiased_geometry and support_policy != "hybrid":
            raise ValueError(
                "the unbiased geometry estimator is only defined for the hybrid "
                f"support policy, got support_policy={support_policy!r}"
            )
        self.support_policy = support_policy
        self.seed = int(seed)

        scales = tuple(artifact.get("metadata", {}).get("diffusion_scales", ()))
        if len(scales) != self.n_scales:
            if self.n_scales == 1:
                scales = (1,)
            else:
                raise ValueError(
                    "GGPKD artifact metadata must provide one diffusion scale "
                    f"per target tensor; got scales={scales}, n_scales={self.n_scales}"
                )
        self.weights = normalized_diffusion_weights(scales)
        self.epoch = 0

    def _mixture_row(self, idx: int) -> np.ndarray:
        """Scale-mixture of anchor ``idx``'s diffusion pool, in float64.

        This used to be a precomputed ``(n_items, width)`` array. ``pool_probs`` is
        float32 and the weights are float64, so the product promoted the whole
        thing: 224 MB at the production shape, duplicated into every DataLoader
        worker, to serve one rarely-taken branch of ``_select_support``. Computing
        the single row on demand is the same arithmetic in the same order over the
        scale axis, so the values are identical, not merely close.
        """
        return (self.pool_probs[:, idx, :] * self.weights.reshape(-1, 1)).sum(axis=0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    _STREAM_CANDIDATES = 0

    def _rng(self, idx: int, stream: int = 0) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, idx, stream])
        )

    def _scale_quotas(self, total: int) -> list[int]:
        """Split the support quota across scales, favoring sharper remainders."""
        base = total // self.n_scales
        quotas = [base] * self.n_scales
        for position in range(total - base * self.n_scales):
            quotas[position] += 1
        return quotas

    def _select_support_uniform(
        self,
        idx: int,
        pool: np.ndarray,
        valid: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform draw over the anchor's own pool, ignoring teacher mass.

        The control arm for the support ablation. It holds the graph, the quota
        and the column population fixed and removes exactly one thing -- the
        teacher's ordering of which of those columns matter -- so a gap against
        the hybrid draw is attributable to relevance rather than to budget,
        encoder cost, or which nodes are reachable at all.

        The per-scale split is deliberately not applied: without teacher mass
        there is nothing to split by, and one uniform draw over the pool is the
        only unambiguous reading of "uniform support".
        """
        available = np.flatnonzero(valid)
        take = min(self.diffusion_quota, available.size)
        support_positions = (
            rng.choice(available, size=take, replace=False).astype(np.int64)
            if take > 0
            else np.empty(0, dtype=np.int64)
        )
        support = pool[support_positions].astype(np.int64)
        return (
            support,
            support_positions,
            np.zeros((self.n_scales, support_positions.size), dtype=np.float32),
            np.full(self.n_scales, -1, dtype=np.int64),
            np.zeros(self.n_scales, dtype=np.float32),
        )

    def _select_support_impl(
        self,
        idx: int,
        rng: np.random.Generator,
        estimate_geometry: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pool = self.pool_indices[idx]
        valid = pool >= 0
        if self.support_policy == "uniform":
            return self._select_support_uniform(idx, pool, valid, rng)
        quotas = self._scale_quotas(self.diffusion_quota)
        taken = np.zeros(pool.size, dtype=bool)
        positions: list[int] = []
        deficit = 0
        geometry_head_pool = np.zeros(
            (self.n_scales, pool.size), dtype=np.float64
        )
        geometry_tail_pool = np.full(self.n_scales, -1, dtype=np.int64)
        geometry_tail_mass = np.zeros(self.n_scales, dtype=np.float64)

        for scale_idx in range(self.n_scales):
            need = quotas[scale_idx] + deficit
            row_probs = np.where(
                valid, self.pool_probs[scale_idx, idx], 0.0
            ).astype(np.float64)
            row_total = float(row_probs.sum())
            normalized = (
                row_probs / row_total
                if row_total > 0.0
                else np.zeros_like(row_probs)
            )
            # `topk` spends the whole per-scale quota on the deterministic head
            # and never reaches the Gumbel draw; `proportional` has no head at all.
            # Both are expressed here so the rest of the draw -- quotas, deficit
            # spill, target extraction -- is byte-for-byte the shared path.
            if self.support_policy == "topk":
                head_per_scale = max(need, 0)
            elif self.support_policy == "proportional":
                head_per_scale = 0
            else:
                head_per_scale = max(1, self.deterministic_topm)
            taken_before = taken.copy()
            probs = row_probs.copy()
            probs[taken_before] = 0.0
            if need <= 0 or not (probs > 0).any():
                if estimate_geometry:
                    geometry_head_pool[scale_idx, taken_before] = normalized[
                        taken_before
                    ]
                    geometry_tail_mass[scale_idx] = float(
                        normalized[~taken_before].sum()
                    )
                    if geometry_tail_mass[scale_idx] > 1e-12:
                        raise ValueError(
                            "unbiased geometry sampling needs at least one available "
                            f"tail slot per diffusion scale; scale={scale_idx}, "
                            f"diffusion_quota={self.diffusion_quota}"
                        )
                deficit = need
                continue

            order = np.argsort(-probs)
            head = order[: max(0, min(head_per_scale, need))]
            head = head[probs[head] > 0]
            rest = probs.copy()
            rest[head] = 0.0
            if (
                estimate_geometry
                and head.size == need
                and head.size > 0
                and (rest > 0).any()
            ):
                # Preserve the fixed support budget while reserving one slot for
                # an exact Gumbel-max draw from the remaining teacher mass.
                head = head[:-1]
                rest = probs.copy()
                rest[head] = 0.0

            estimator_head = taken_before.copy()
            estimator_head[head] = True
            tail = _gumbel_topk(rest, need - head.size, rng)
            if estimate_geometry:
                geometry_head_pool[scale_idx, estimator_head] = normalized[
                    estimator_head
                ]
                tail_mass = float(normalized[~estimator_head].sum())
                geometry_tail_mass[scale_idx] = tail_mass
                if tail_mass > 1e-12:
                    if tail.size == 0:
                        raise ValueError(
                            "unbiased geometry sampling could not draw from a "
                            f"positive-mass tail at scale={scale_idx}"
                        )
                    # The first ordered Gumbel top-k item is an exact categorical
                    # draw. Later items have non-trivial inclusion probabilities
                    # and remain support for KL only.
                    geometry_tail_pool[scale_idx] = int(tail[0])

            chosen = np.concatenate([head, tail]).astype(np.int64)
            taken[chosen] = True
            positions.extend(int(position) for position in chosen)
            deficit = need - chosen.size

        if deficit > 0:
            # Only this branch ever needed the mixture, and it is the rare one: the
            # per-scale quotas normally fill from the scales themselves.
            spill = np.where(valid, self._mixture_row(idx), 0.0)
            spill[taken] = 0.0
            extra = np.argsort(-spill)[:deficit]
            extra = extra[spill[extra] > 0]
            positions.extend(int(position) for position in extra)

        support_positions = np.asarray(positions, dtype=np.int64)
        support = pool[support_positions].astype(np.int64)
        geometry_head_probs = np.zeros(
            (self.n_scales, support_positions.size), dtype=np.float32
        )
        geometry_tail_positions = np.full(self.n_scales, -1, dtype=np.int64)
        if estimate_geometry and support_positions.size:
            candidate_position = np.full(pool.size, -1, dtype=np.int64)
            candidate_position[support_positions] = np.arange(support_positions.size)
            for scale_idx in range(self.n_scales):
                head_pool_positions = np.flatnonzero(
                    geometry_head_pool[scale_idx] > 0
                )
                if head_pool_positions.size:
                    mapped = candidate_position[head_pool_positions]
                    if (mapped < 0).any():
                        raise RuntimeError(
                            "geometry head contains a node outside the candidate support"
                        )
                    geometry_head_probs[scale_idx, mapped] = geometry_head_pool[
                        scale_idx, head_pool_positions
                    ].astype(np.float32)
                tail_pool_position = int(geometry_tail_pool[scale_idx])
                if tail_pool_position >= 0:
                    mapped_tail = int(candidate_position[tail_pool_position])
                    if mapped_tail < 0:
                        raise RuntimeError(
                            "geometry tail draw is outside the candidate support"
                        )
                    geometry_tail_positions[scale_idx] = mapped_tail

        return (
            support,
            support_positions,
            geometry_head_probs,
            geometry_tail_positions,
            geometry_tail_mass.astype(np.float32),
        )

    def _select_support(
        self, idx: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        support, positions, _, _, _ = self._select_support_impl(
            idx, rng, estimate_geometry=False
        )
        return support, positions

    def _sample_impl(self, idx: int, estimate_geometry: bool):
        rng = self._rng(idx, self._STREAM_CANDIDATES)
        (
            support,
            support_positions,
            geometry_head_probs,
            geometry_tail_positions,
            geometry_tail_mass,
        ) = self._select_support_impl(idx, rng, estimate_geometry=estimate_geometry)
        support_set = set(int(node) for node in support)

        # Partition the complement into a hard stratum and everything else. Their
        # disjointness makes pi_h=k_h/H and pi_u=k_u/U exact marginal inclusion
        # probabilities for the without-replacement draws.
        hard_pool = {
            int(node)
            for node in self.hard_neg_indices[idx]
            if int(node) >= 0 and int(node) != idx and int(node) not in support_set
        }
        uniform_excluded = {int(idx), *support_set, *hard_pool}
        uniform_population = self.n_items - len(uniform_excluded)
        remaining = self.candidate_size - support.size

        n_hard = min(self.hard_neg_k, len(hard_pool), remaining)
        n_uniform = min(self.random_neg_k, uniform_population, remaining - n_hard)
        deficit = remaining - n_hard - n_uniform
        add_uniform = min(deficit, uniform_population - n_uniform)
        n_uniform += add_uniform
        deficit -= add_uniform
        n_hard += min(deficit, len(hard_pool) - n_hard)

        if support.size + n_hard + n_uniform != self.candidate_size:
            raise ValueError(
                "GGPKD candidate budget exceeds the available non-anchor corpus: "
                f"budget={self.candidate_size}, n_items={self.n_items}"
            )

        hard_nodes = np.asarray(sorted(hard_pool), dtype=np.int64)
        if n_hard:
            hard_nodes = rng.choice(hard_nodes, size=n_hard, replace=False)
        else:
            hard_nodes = np.empty(0, dtype=np.int64)
        uniform_nodes = np.asarray(
            self._draw_random(rng, set(uniform_excluded), n_uniform), dtype=np.int64
        )
        candidate_arr = np.concatenate([support, hard_nodes, uniform_nodes])

        teacher_probs = np.zeros((self.n_scales, candidate_arr.size), dtype=np.float32)
        if support.size:
            teacher_probs[:, : support.size] = self.pool_probs[
                :, idx, support_positions
            ]
        if not estimate_geometry:
            return candidate_arr, teacher_probs

        geometry_head_full = np.zeros_like(teacher_probs)
        geometry_head_full[:, : support.size] = geometry_head_probs
        return (
            candidate_arr,
            teacher_probs,
            geometry_head_full,
            geometry_tail_positions,
            geometry_tail_mass,
        )

    def sample(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return candidates and diffusion targets restricted to that draw."""
        return self._sample_impl(idx, estimate_geometry=False)

    def sample_with_geometry(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return candidates plus exact head--tail estimator metadata."""
        return self._sample_impl(idx, estimate_geometry=True)

    def _draw_random(
        self, rng: np.random.Generator, excluded: set[int], count: int
    ) -> list[int]:
        if count <= 0:
            return []
        drawn: list[int] = []
        block = rng.integers(0, self.n_items, size=max(4 * count, 16))
        for node in block:
            if len(drawn) >= count:
                break
            node = int(node)
            if node in excluded:
                continue
            drawn.append(node)
            excluded.add(node)
        while len(drawn) < count:
            node = int(rng.integers(0, self.n_items))
            if node in excluded:
                continue
            drawn.append(node)
            excluded.add(node)
        return drawn

    def sample_torch(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_arr, teacher_probs = self.sample(idx)
        return (
            torch.from_numpy(candidate_arr).long(),
            torch.from_numpy(teacher_probs).float(),
        )

    def sample_geometry_torch(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_arr, teacher_probs, head_probs, tail_positions, tail_mass = (
            self.sample_with_geometry(idx)
        )
        return (
            torch.from_numpy(candidate_arr).long(),
            torch.from_numpy(teacher_probs).float(),
            torch.from_numpy(head_probs).float(),
            torch.from_numpy(tail_positions).long(),
            torch.from_numpy(tail_mass).float(),
        )
