import numpy as np
import torch

from .policy import (
    SUPPORT_POLICIES,
    candidate_budget,
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
        diffusion_quota: int | None,
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

        # `None` is the method: the candidate set is the anchor's whole transition
        # row, so there is no budget and no selection. Resolving it to the
        # pool width keeps one fixed row width for collation -- the columns an anchor
        # does not have are padded inertly below, exactly as a short draw already was.
        self.pool_width = int(self.pool_indices.shape[1])
        self.full_pool = diffusion_quota is None
        resolved_quota = self.pool_width if self.full_pool else int(diffusion_quota)
        self.candidate_size = candidate_budget(
            resolved_quota, hard_neg_k, random_neg_k
        )
        self.diffusion_quota = resolved_quota
        self.hard_neg_k = int(hard_neg_k)
        self.random_neg_k = int(random_neg_k)
        if support_policy not in SUPPORT_POLICIES:
            raise ValueError(
                f"support_policy must be one of {SUPPORT_POLICIES}, "
                f"got {support_policy!r}"
            )
        self.support_policy = support_policy
        self.seed = int(seed)
        # The method draws no negatives at all. Kept as a derived flag rather than
        # re-tested per draw, because it changes what a short support draw means.
        self.no_negatives = self.hard_neg_k == 0 and self.random_neg_k == 0
        # Anchors padded to the fixed row width because their own row was shorter.
        # Under `full_pool` this is inert for the method: a directed graph with no
        # truncation gives every anchor exactly graph_k columns, so nothing is
        # padded. It is the arms that produce ragged rows -- a mutual filter, or a
        # positive truncation tolerance -- where fill runs below pool_width and
        # every anchor short of the widest row is padded. With an explicit quota it means the quota outran the pool, which
        # is what silently turned a requested width of 500 into a real width of 67 --
        # so it is counted either way and reported by the distiller.
        self.short_support_rows = 0

        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    _STREAM_CANDIDATES = 0

    def _rng(self, idx: int, stream: int = 0) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, idx, stream])
        )

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

    def _select_support_off_graph(
        self,
        idx: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Columns drawn from the corpus rather than from the anchor's row.

        The control arms of the support study. Both draw uniformly without
        replacement from every node but the anchor; they differ only in how many:

            corpus_uniform  the configured quota, the same number for every
                            anchor and the same number the teacher arm is given.
            rewired         this anchor's own degree, so the ragged degree
                            profile of the teacher graph survives the rewiring
                            and only the endpoints move.

        The returned positions are empty by construction: a position indexes the
        anchor's diffusion pool, and none of these columns are in it. Every
        caller downstream reads teacher mass through those positions, so the mass
        these arms carry is exactly zero -- which is why the config requires
        relation_target="direct", where the target is read off the teacher bank
        instead.
        """
        if self.support_policy == "rewired":
            degree = int((self.pool_indices[idx] >= 0).sum())
            # Clamped to the budget, not to the raw degree. The teacher arm this
            # is a control for is itself given `diffusion_quota` columns, so an
            # anchor whose row is wider than the budget contributes exactly
            # `diffusion_quota` columns there too -- matching the raw degree would
            # hand the rewired arm more supervision than the arm it is meant to
            # isolate, and would overrun the fixed candidate width besides.
            # Anchors below the budget keep their own smaller degree, which is
            # what preserves the ragged profile where it actually differs.
            take = min(max(degree, 1), self.diffusion_quota, self.n_items - 1)
        else:
            take = min(self.diffusion_quota, self.n_items - 1)
        support = np.asarray(
            self._draw_random(rng, {int(idx)}, take), dtype=np.int64
        )
        return support, np.empty(0, dtype=np.int64)

    def _select_support(
        self,
        idx: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.support_policy in ("corpus_uniform", "rewired"):
            # Before the full_pool branch: these arms define their own width and
            # never read the pool, so "the whole transition row" is not a width
            # they could take.
            return self._select_support_off_graph(idx, rng)
        pool = self.pool_indices[idx]
        valid = pool >= 0
        if self.full_pool:
            # Every column the teacher put mass on. Taken directly rather than
            # through the budgeted branch below, which exists only for the arms
            # that are given a quota smaller than the row.
            mass = np.where(valid, self.pool_probs[0, idx], 0.0)
            positions = np.flatnonzero(mass > 0).astype(np.int64)
            return pool[positions].astype(np.int64), positions
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
        # One hop, so there is one target row and the budget is spent on it in a
        # single pass. This used to split the quota across diffusion scales and
        # spill the remainder onto their mixture.
        probs = np.where(valid, self.pool_probs[0, idx], 0.0).astype(np.float64)
        need = self.diffusion_quota
        if not (probs > 0).any():
            positions: list[int] = []
        elif self.support_policy == "topk":
            order = np.argsort(-probs)
            chosen = order[: min(need, int((probs > 0).sum()))]
            positions = [int(c) for c in chosen[probs[chosen] > 0]]
        elif self.support_policy == "proportional":
            positions = [int(c) for c in _gumbel_topk(probs, need, rng)]
        else:  # guarded by SUPPORT_POLICIES in __init__
            raise RuntimeError(f"unsupported support policy {self.support_policy!r}")

        support_positions = np.asarray(positions, dtype=np.int64)
        support = pool[support_positions].astype(np.int64)
        return support, support_positions

    def sample(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return candidates and diffusion targets restricted to that draw."""
        rng = self._rng(idx, self._STREAM_CANDIDATES)
        support, support_positions = self._select_support(idx, rng)
        support_set = set(int(node) for node in support)

        # Partition the complement into a hard stratum and everything else. Their
        # disjointness makes pi_h=k_h/H and pi_u=k_u/U exact marginal inclusion
        # probabilities for the without-replacement draws.
        hard_pool = {
            int(node)
            for node in self.hard_neg_indices[idx]
            if int(node) >= 0 and int(node) != idx and int(node) not in support_set
        }
        # When the hard stratum is active, keep the two draws disjoint so their
        # marginal inclusion probabilities remain exact.  With hard_neg_k=0,
        # however, there is no hard draw: excluding the hard pool would turn the
        # advertised uniform-corpus control into "uniform except for the nearest
        # neighbours".  Fold that pool back into the uniform population.
        uniform_excluded = {int(idx), *support_set}
        if self.hard_neg_k > 0:
            uniform_excluded.update(hard_pool)
        uniform_population = self.n_items - len(uniform_excluded)
        remaining = self.candidate_size - support.size

        if self.no_negatives:
            # Both quotas are zero, so a short support draw has nothing to be
            # filled from -- and the deficit path below would have filled it with
            # uniform corpus draws anyway, silently reintroducing the negatives
            # this configuration exists to remove. The row is padded to the fixed
            # width with the anchor's own index instead, which is inert: it
            # carries teacher mass 0, and `self_mask` removes it from the anchor's
            # ambient and diffusion softmaxes alike, so the anchor is simply
            # supervised on a narrower support.
            #
            # `support.size == 0` is not padded this way: a row of nothing but
            # self is fully masked, its diffusion log-softmax is all -inf, and
            # while the loss stays 0 there the per-scale diagnostics go NaN. It
            # cannot happen with diffusion_quota >= 1 -- the graph build gives
            # every node at least its top-k fallback neighbours -- so it is
            # asserted rather than handled.
            if remaining > 0:
                if support.size == 0:
                    raise ValueError(
                        f"GGPKD anchor {idx} has no diffusion support and no "
                        "negative quota to fall back on; rebuild the graph "
                        "artifact or give the arm a non-zero negative quota"
                    )
                self.short_support_rows += 1
                padding = np.full(remaining, int(idx), dtype=np.int64)
                candidate_arr = np.concatenate([support, padding])
                teacher_probs = np.zeros(
                    (self.n_scales, candidate_arr.size), dtype=np.float32
                )
                # Guarded on the *positions*, not on the support. An off-graph arm
                # returns columns with no pool position at all, and its mass is
                # zero for every one of them; indexing the pool with the empty
                # position array would try to write a width-0 block into a
                # width-|S| slot.
                if support_positions.size:
                    teacher_probs[:, : support_positions.size] = self.pool_probs[
                        :, idx, support_positions
                    ]
                return candidate_arr, teacher_probs
            n_hard = n_uniform = 0
        else:
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

        # Sorted so the draw is a function of the seed and not of set iteration
        # order.
        hard_nodes = np.fromiter(hard_pool, dtype=np.int64, count=len(hard_pool))
        hard_nodes.sort()
        if n_hard:
            hard_nodes = rng.choice(hard_nodes, size=n_hard, replace=False)
        else:
            hard_nodes = np.empty(0, dtype=np.int64)
        uniform_nodes = np.asarray(
            self._draw_random(rng, set(uniform_excluded), n_uniform), dtype=np.int64
        )
        candidate_arr = np.concatenate([support, hard_nodes, uniform_nodes])

        teacher_probs = np.zeros((self.n_scales, candidate_arr.size), dtype=np.float32)
        if support_positions.size:
            teacher_probs[:, : support_positions.size] = self.pool_probs[
                :, idx, support_positions
            ]
        return candidate_arr, teacher_probs

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
