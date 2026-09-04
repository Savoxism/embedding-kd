import numpy as np
import torch

from .policy import (
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
        support_policy: str = "topk",
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
        if support_policy not in SUPPORT_POLICIES:
            raise ValueError(
                f"support_policy must be one of {SUPPORT_POLICIES}, "
                f"got {support_policy!r}"
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform draw over the anchor's own pool, ignoring teacher mass.

        The control arm for the support ablation. It holds the graph, the quota
        and the column population fixed and removes exactly one thing -- the
        teacher's ordering of which of those columns matter -- so a gap against
        the Top-k draw is attributable to relevance rather than to budget,
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
        return support, support_positions

    def _select_support_impl(
        self,
        idx: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        pool = self.pool_indices[idx]
        valid = pool >= 0
        if self.support_policy == "uniform":
            return self._select_support_uniform(idx, pool, valid, rng)
        if self.support_policy == "local_topk":
            # Clean no-diffusion control. Select from the one-step transition
            # row only, but read it from the full multi-scale artifact. This is
            # intentionally distinct from rebuilding an R={1} artifact: keeping
            # all configured scales lets relation_target="direct" collapse the
            # graph group with the exact same total graph weight as the method.
            probs = np.where(valid, self.pool_probs[0, idx], 0.0).astype(np.float64)
            order = np.argsort(-probs)
            positions = order[: min(self.diffusion_quota, int((probs > 0).sum()))]
            positions = positions[probs[positions] > 0].astype(np.int64)
            return pool[positions].astype(np.int64), positions
        quotas = self._scale_quotas(self.diffusion_quota)
        taken = np.zeros(pool.size, dtype=bool)
        positions: list[int] = []
        deficit = 0

        for scale_idx in range(self.n_scales):
            need = quotas[scale_idx] + deficit
            row_probs = np.where(
                valid, self.pool_probs[scale_idx, idx], 0.0
            ).astype(np.float64)
            probs = row_probs.copy()
            probs[taken] = 0.0
            if need <= 0 or not (probs > 0).any():
                deficit = need
                continue

            if self.support_policy == "topk":
                order = np.argsort(-probs)
                chosen = order[: min(need, int((probs > 0).sum()))]
                chosen = chosen[probs[chosen] > 0].astype(np.int64)
            elif self.support_policy == "proportional":
                chosen = _gumbel_topk(probs, need, rng)
            else:  # guarded by SUPPORT_POLICIES in __init__
                raise RuntimeError(f"unsupported support policy {self.support_policy!r}")
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
        return support, support_positions

    def _select_support(
        self, idx: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._select_support_impl(idx, rng)

    def _sample_impl(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rng = self._rng(idx, self._STREAM_CANDIDATES)
        support, support_positions = self._select_support_impl(idx, rng)
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
        return candidate_arr, teacher_probs

    def sample(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return candidates and diffusion targets restricted to that draw."""
        return self._sample_impl(idx)

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
