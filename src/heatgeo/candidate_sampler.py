from collections.abc import Sequence

import numpy as np
import torch


def _normalized_weights(scale_weights: Sequence[float], n_scales: int) -> np.ndarray:
    weights = np.asarray(list(scale_weights), dtype=np.float64)
    if weights.size == 0:
        weights = np.ones(n_scales, dtype=np.float64)
    if weights.size < n_scales:
        weights = np.concatenate(
            [weights, np.repeat(weights[-1], n_scales - weights.size)]
        )
    weights = weights[:n_scales]
    return weights / max(float(weights.sum()), 1e-12)


def _gumbel_topk(
    probs: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Weighted sampling without replacement (Gumbel top-k / Efraimidis-Spirakis).

    Returns positions into `probs`. O(n log k) with no rejection loop, which matters
    because this runs once per training example per epoch inside the dataloader.
    """
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


class HeatGeoCandidateSampler:
    """Builds a fresh candidate set per (anchor, epoch) from the precomputed pools.

    The previous design froze one candidate set per anchor for the whole run, so
    every epoch after the first re-presented comparisons the student had already
    fit. Here only the diffusion *support* is precomputed; which members of it are
    scored, and against which negatives, is redrawn each epoch.

    Sampling is seeded by (seed, epoch, idx), so it is reproducible and safe under
    multi-worker dataloading -- no shared RNG state is touched.
    """

    def __init__(
        self,
        artifact: dict,
        candidate_size: int,
        diffusion_quota: int,
        hard_neg_k: int,
        random_neg_k: int,
        scale_weights: Sequence[float],
        seed: int,
        deterministic_topm: int = 4,
        stochastic: bool = True,
    ):
        self.pool_indices = artifact["pool_indices"].numpy()
        self.pool_probs = artifact["pool_probs"].numpy()
        self.hard_neg_indices = artifact["hard_neg_indices"].numpy()
        self.n_items = int(self.pool_indices.shape[0])
        self.n_scales = int(self.pool_probs.shape[0])

        if diffusion_quota + hard_neg_k + random_neg_k > candidate_size:
            raise ValueError(
                f"quotas exceed candidate_size: diffusion={diffusion_quota} + "
                f"hard={hard_neg_k} + random={random_neg_k} > {candidate_size}"
            )

        self.candidate_size = int(candidate_size)
        self.diffusion_quota = int(diffusion_quota)
        self.hard_neg_k = int(hard_neg_k)
        self.random_neg_k = int(random_neg_k)
        self.deterministic_topm = int(deterministic_topm)
        self.stochastic = bool(stochastic)
        self.seed = int(seed)
        self.weights = _normalized_weights(scale_weights, self.n_scales)
        self.mixture = (self.pool_probs * self.weights.reshape(-1, 1, 1)).sum(axis=0)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, idx])
        )

    def _scale_quotas(self, total: int) -> list[int]:
        """Split the diffusion quota evenly across scales, remainder to sharper ones.

        Drawing the whole quota from the mixture looks natural but starves the
        sharpest scale: the mixture's support is dominated by the broadest walk, so
        r=1 -- whose support is only the mutual-kNN neighbourhood -- ends up with
        one or two scored candidates and its KL term stops carrying any ranking
        information at all.
        """
        base = total // self.n_scales
        quotas = [base] * self.n_scales
        for position in range(total - base * self.n_scales):
            quotas[position] += 1
        return quotas

    def sample(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rng = self._rng(idx)
        pool = self.pool_indices[idx]
        valid = pool >= 0
        mixture = np.where(valid, self.mixture[idx], 0.0).astype(np.float64)

        # --- diffusion positives, round-robin over scales ------------------------
        quotas = self._scale_quotas(self.diffusion_quota)
        head_per_scale = max(1, self.deterministic_topm)
        taken = np.zeros(pool.size, dtype=bool)
        positions: list[int] = []
        deficit = 0

        for scale_idx in range(self.n_scales):
            need = quotas[scale_idx] + deficit
            probs = np.where(valid, self.pool_probs[scale_idx, idx], 0.0).astype(
                np.float64
            )
            probs[taken] = 0.0
            if need <= 0 or not (probs > 0).any():
                deficit = need
                continue

            if self.stochastic:
                # Always keep this scale's strongest neighbours, sample the rest in
                # proportion to diffusion mass so the tail of the support is seen too.
                order = np.argsort(-probs)
                head = order[: min(head_per_scale, need)]
                head = head[probs[head] > 0]
                rest = probs.copy()
                rest[head] = 0.0
                tail = _gumbel_topk(rest, need - head.size, rng)
                chosen = np.concatenate([head, tail]).astype(np.int64)
            else:
                order = np.argsort(-probs)
                chosen = order[:need]
                chosen = chosen[probs[chosen] > 0]

            taken[chosen] = True
            positions.extend(int(position) for position in chosen)
            deficit = need - chosen.size

        if deficit > 0:
            spill = mixture.copy()
            spill[taken] = 0.0
            extra = np.argsort(-spill)[:deficit]
            extra = extra[spill[extra] > 0]
            taken[extra] = True
            positions.extend(int(position) for position in extra)

        candidates = [int(node) for node in pool[np.asarray(positions, dtype=np.int64)]]
        seen = {int(idx), *candidates}

        # --- hard negatives ------------------------------------------------------
        hard_pool = self.hard_neg_indices[idx]
        hard_pool = hard_pool[hard_pool >= 0]
        if hard_pool.size and self.hard_neg_k:
            take = min(self.hard_neg_k, hard_pool.size)
            picked = rng.choice(hard_pool, size=take, replace=False)
            for node in picked:
                node = int(node)
                if node not in seen:
                    candidates.append(node)
                    seen.add(node)

        # --- random negatives ----------------------------------------------------
        target = min(self.candidate_size, len(candidates) + self.random_neg_k)
        candidates.extend(self._draw_random(rng, seen, target - len(candidates)))

        # --- padding -------------------------------------------------------------
        # Every appended index is checked against `seen`. The previous padding loop
        # skipped that check, so the same node could occupy several slots: its
        # teacher mass was counted once per slot and the student softmax scored the
        # same text several times.
        if len(candidates) < self.candidate_size:
            for position in np.argsort(-mixture):
                if len(candidates) >= self.candidate_size:
                    break
                node = int(pool[position])
                if node < 0 or node in seen:
                    continue
                candidates.append(node)
                seen.add(node)
        if len(candidates) < self.candidate_size:
            for node in hard_pool:
                if len(candidates) >= self.candidate_size:
                    break
                node = int(node)
                if node in seen:
                    continue
                candidates.append(node)
                seen.add(node)
        if len(candidates) < self.candidate_size:
            candidates.extend(
                self._draw_random(rng, seen, self.candidate_size - len(candidates))
            )

        candidate_arr = np.asarray(candidates[: self.candidate_size], dtype=np.int64)

        # --- teacher targets restricted to the drawn set -------------------------
        position_of = {int(node): pos for pos, node in enumerate(pool) if node >= 0}
        gather = np.asarray(
            [position_of.get(int(node), -1) for node in candidate_arr], dtype=np.int64
        )
        present = gather >= 0
        teacher_probs = np.zeros(
            (self.n_scales, candidate_arr.size), dtype=np.float32
        )
        if present.any():
            teacher_probs[:, present] = self.pool_probs[:, idx, gather[present]]
        # A scale with no mass on the drawn set is left as all-zeros, not filled with
        # a uniform distribution. Uniform is not "no opinion": it is the assertion
        # that the anchor is equally close to all candidate_size candidates, random
        # cross-domain negatives included, and its KL term is a ~log(candidate_size)
        # constant that dominates the anchor's loss. The criterion skips zero-target
        # scales, which is the correct "this scale has nothing to say here".
        totals = teacher_probs.sum(axis=-1, keepdims=True)
        teacher_probs = np.divide(
            teacher_probs,
            np.maximum(totals, 1e-12),
            out=teacher_probs,
        )

        return candidate_arr, teacher_probs

    def _draw_random(
        self, rng: np.random.Generator, seen: set[int], count: int
    ) -> list[int]:
        if count <= 0 or len(seen) >= self.n_items:
            return []
        drawn: list[int] = []
        # Oversample once and filter: cheaper than a rejection loop, and the
        # collision rate is negligible for corpora of any realistic size.
        block = rng.integers(0, self.n_items, size=max(4 * count, 16))
        for node in block:
            if len(drawn) >= count:
                break
            node = int(node)
            if node in seen:
                continue
            drawn.append(node)
            seen.add(node)
        while len(drawn) < count and len(seen) < self.n_items:
            node = int(rng.integers(0, self.n_items))
            if node in seen:
                continue
            drawn.append(node)
            seen.add(node)
        return drawn

    def sample_torch(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_arr, teacher_probs = self.sample(idx)
        return (
            torch.from_numpy(candidate_arr).long(),
            torch.from_numpy(teacher_probs).float(),
        )
