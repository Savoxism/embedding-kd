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
        num_walks: int = 0,
        walk_length: int = 4,
        walk_topk: int | None = None,
        walk_non_backtracking: bool = True,
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

        # ---- Random walk trajectory sampling ------------------------------------
        self.num_walks = int(num_walks)
        self.walk_length = int(walk_length)
        self.walk_non_backtracking = bool(walk_non_backtracking)
        self.walk_topk = None if walk_topk is None else int(walk_topk)
        if self.walk_topk is not None and self.walk_topk <= 0:
            raise ValueError("walk_topk must be positive when provided")
        if self.num_walks > 0:
            if "transition_neighbors" not in artifact or "transition_probs" not in artifact:
                raise ValueError(
                    "Walk sampling requires transition_neighbors and transition_probs "
                    "in the artifact. Rebuild the graph cache with the updated graph_builder."
                )
            self.trans_neighbors = artifact["transition_neighbors"].numpy()
            self.trans_probs = artifact["transition_probs"].numpy()
            if (
                self.walk_topk is not None
                and self.walk_topk < self.trans_probs.shape[1]
            ):
                order = np.argsort(-self.trans_probs, axis=1)[:, : self.walk_topk]
                self.trans_neighbors = np.take_along_axis(
                    self.trans_neighbors, order, axis=1
                )
                self.trans_probs = np.take_along_axis(
                    self.trans_probs, order, axis=1
                )
                totals = self.trans_probs.sum(axis=1, keepdims=True)
                self.trans_probs = np.divide(
                    self.trans_probs,
                    np.maximum(totals, 1e-12),
                    out=np.zeros_like(self.trans_probs),
                )
        else:
            self.trans_neighbors = None
            self.trans_probs = None

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

    def _sample_walks(
        self, idx: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, set[int]]:
        """Sample ``num_walks`` trajectories of length ``walk_length`` from anchor *idx*.

        Each walk starts at *idx* and follows the teacher's transition matrix.

        The anchor is *not* in the returned node set: it is excluded from its own
        candidate draw by construction, the loss drops step 0 as redundant with the
        r=1 diffusion target, and injecting it would burn a candidate slot that every
        scale then masks out again.

        **Non-backtracking.** With ``walk_non_backtracking`` the step at time t+1 may
        not return to the node occupied at t-1. This changes only the sampling measure
        nu -- *which* rows the walk term supervises -- and never a target: the teacher
        row matched at a visited node is still the full transition row P(.|j) at
        graph_temp, read from the artifact by the criterion. The temperature tie and
        the walk term's infimum of 0 are therefore untouched.

        Two things the plain walk wastes and this recovers. Rows are supervised with
        weight equal to visit count, so an A->B->A->B excursion spends the whole walk
        budget re-supervising two rows; a non-backtracking walk of the same length
        touches more distinct rows for the same number of student forward columns.
        And the gauge argument (Proposition ``prop:walkgauge``) identifies similarity
        offsets by *traversed edges*, so a repeated edge adds no constraint -- the
        reduction propagates through the component faster when consecutive steps
        cannot undo each other.

        Non-backtracking walks are also the ones with the mixing and cover-time
        guarantees (Alon, Benjamini, Lubetzky and Sodin, 2007); on a mutual-kNN graph
        with bounded degree that is the whole of the structural argument, which is why
        no degree-dependent reweighting is applied on top -- it would tilt nu without
        a matching correction and buy a knob with no evidence behind it.

        Returns:
            walks: int array [num_walks, walk_length + 1] of node indices.
            walk_nodes: set of unique node indices visited after the start.
        """
        walks = np.empty(
            (self.num_walks, self.walk_length + 1), dtype=np.int64
        )
        walk_node_set: set[int] = set()
        for m in range(self.num_walks):
            current = idx
            previous = -1
            walks[m, 0] = current
            for t in range(self.walk_length):
                neighbors = self.trans_neighbors[current]
                probs = self.trans_probs[current]
                # A padded slot carries neighbour -1 *and* probability 0, but walk_topk
                # truncation can also leave a real neighbour at probability 0; both are
                # dropped here so the renormalization below can never divide by zero.
                valid = (neighbors >= 0) & (probs > 0)
                if self.walk_non_backtracking and previous >= 0:
                    forward = valid & (neighbors != previous)
                    # Only when every forward edge is gone -- a degree-1 node, where
                    # the edge we arrived on is the sole way out -- is the backtrack
                    # allowed. Alon et al. define non-backtracking walks on graphs of
                    # minimum degree 2; retreating beats stalling on a self-loop, which
                    # would supervise the same row twice and teach nothing.
                    if forward.any():
                        valid = forward
                if not valid.any():
                    # Dead-end node: stay in place (self-loop).
                    walks[m, t + 1] = current
                    continue
                nb = neighbors[valid]
                pr = probs[valid].astype(np.float64)
                pr = pr / pr.sum()
                next_node = int(rng.choice(nb, p=pr))
                walks[m, t + 1] = next_node
                walk_node_set.add(next_node)
                previous, current = current, next_node
        return walks, walk_node_set

    def sample_with_walks(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like ``sample`` but also returns walk trajectories.

        Walk-visited nodes that are not already in the candidate set are
        appended (replacing random negatives if room is needed), so the
        student encoder produces embeddings for them and the walk loss can
        score their teacher rows without extra forward passes. Their presence
        also raises the odds that a *neighbour* of a walk node is in the pool,
        which is what the walk softmax denominator is drawn from.

        Returns:
            candidate_arr: int array [candidate_size]
            teacher_probs: float array [n_scales, candidate_size]
            walks: int array [num_walks, walk_length + 1]   (-1 padded when disabled)
        """
        rng = self._rng(idx)
        # Sample walks first so their nodes can be injected into the candidate set.
        if self.num_walks > 0:
            walks, walk_node_set = self._sample_walks(idx, rng)
        else:
            walks = np.full((1, 2), -1, dtype=np.int64)
            walk_node_set = set()

        # --- Run the standard candidate sampling --------------------------------
        candidate_arr, teacher_probs = self.sample(idx)

        if self.num_walks <= 0:
            return candidate_arr, teacher_probs, walks

        # --- Inject walk nodes that are not in the candidate set ----------------
        existing = set(int(c) for c in candidate_arr)
        missing = [n for n in walk_node_set if n not in existing]
        if missing:
            # Replace trailing random negatives (least informative slots) with
            # walk nodes.  This keeps candidate_size constant and avoids any
            # change to the shared pool / collate logic.
            pool_set = set(int(p) for p in self.pool_indices[idx] if p >= 0)
            hard_set = set(int(h) for h in self.hard_neg_indices[idx] if h >= 0)
            # Identify replaceable positions: random negatives are those that
            # are neither in the diffusion pool nor in the hard-negative pool.
            replaceable = [
                pos for pos, c in enumerate(candidate_arr)
                if int(c) not in pool_set and int(c) not in hard_set and int(c) != idx
            ]
            for node, pos in zip(missing, replaceable):
                candidate_arr[pos] = node
            # If more walk nodes remain than replaceable slots, we just accept
            # that some walk steps will reference nodes outside the candidate set
            # and will be skipped during walk-loss computation.

        # Recompute teacher_probs for any swapped-in walk nodes.
        position_of = {
            int(node): pos
            for pos, node in enumerate(self.pool_indices[idx])
            if node >= 0
        }
        for pos in range(candidate_arr.size):
            pool_pos = position_of.get(int(candidate_arr[pos]), -1)
            if pool_pos >= 0:
                teacher_probs[:, pos] = self.pool_probs[:, idx, pool_pos]
            else:
                teacher_probs[:, pos] = 0.0
        totals = teacher_probs.sum(axis=-1, keepdims=True)
        teacher_probs = np.divide(
            teacher_probs,
            np.maximum(totals, 1e-12),
            out=teacher_probs,
        )

        return candidate_arr, teacher_probs, walks

    def sample_torch(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_arr, teacher_probs, walks = self.sample_with_walks(idx)
        return (
            torch.from_numpy(candidate_arr).long(),
            torch.from_numpy(teacher_probs).float(),
            torch.from_numpy(walks).long(),
        )
