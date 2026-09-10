"""Teacher-informed batch composition, for the batch-intervention study.

A mini-batch is supposed to be a computational device: which examples travel
together should change the gradient's variance and nothing else. For a loss whose
relational support *is* the batch, it changes the objective instead -- the set of
relations that ever get supervised is decided by co-occurrence.

These samplers are the intervention that makes that difference measurable. They
compose batches from teacher geometry alone, with no labels, and they are the
only thing that differs between the arms of the study:

    random            i.i.d. reshuffle. The method's setting, and the control.
    teacher_neighbor  each batch is drawn from one teacher neighbourhood, so
                      in-batch relations are unusually teacher-relevant.
    teacher_diverse   each batch spreads across distant neighbourhoods, so
                      in-batch relations are unusually irrelevant.

The prediction the study tests: a pointwise objective and a teacher-support
relational objective are unmoved by all three (their supervision does not read
batch membership), while an in-batch relational objective moves with them. If
that holds, batch composition is not an optimization detail for in-batch
relational KD; it is part of the objective.

Both teacher-informed samplers are *partitions*: every corpus index appears
exactly once per epoch, in exactly one batch. That is what keeps the arms
comparable -- the anchor set, the number of steps and the number of gradient
updates are identical to the random arm, and only the grouping differs. A sampler
that resampled instead would also change how often each anchor is seen, and the
comparison would no longer be one-factor.
"""

from collections.abc import Iterator, Sequence

import numpy as np

BATCH_SAMPLERS = ("random", "teacher_neighbor", "teacher_diverse")


def _shuffled_batches(
    groups: list[np.ndarray], rng: np.random.Generator
) -> list[list[int]]:
    """Shuffle within each batch and shuffle the batch order.

    The grouping is the intervention; the order inside and between batches is
    not, and leaving it in the order the construction happened to produce would
    correlate batch index with position in the corpus.
    """
    batches = []
    for group in groups:
        members = group.copy()
        rng.shuffle(members)
        batches.append([int(index) for index in members])
    rng.shuffle(batches)
    return batches


def _neighbor_batches(
    neighbors: np.ndarray, batch_size: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Greedy neighbourhood partition.

    Repeatedly take an unused seed at random, then fill its batch with that
    seed's nearest unused teacher neighbours in teacher-cosine order. When the
    seed's neighbour list is exhausted before the batch is full -- the graph is
    ragged and neighbours get consumed by earlier batches -- the remainder is
    filled from the unused pool at random rather than left short: a short batch
    would change the number of relations per step, which is not the variable
    under test.

    Greedy rather than a balanced graph partition on purpose. The claim being
    tested is comparative ("more teacher-relevant than random"), and the study
    measures the resulting relevance directly rather than assuming it, so a
    cheap construction whose achieved relevance is reported is worth more than an
    expensive one whose optimality is asserted.
    """
    n_items = int(neighbors.shape[0])
    used = np.zeros(n_items, dtype=bool)
    order = rng.permutation(n_items)
    batches: list[np.ndarray] = []
    cursor = 0

    while True:
        while cursor < n_items and used[order[cursor]]:
            cursor += 1
        if cursor >= n_items:
            break
        seed = int(order[cursor])
        used[seed] = True
        members = [seed]

        for candidate in neighbors[seed]:
            if len(members) >= batch_size:
                break
            candidate = int(candidate)
            if candidate < 0 or candidate == seed or used[candidate]:
                continue
            used[candidate] = True
            members.append(candidate)

        if len(members) < batch_size:
            remaining = np.flatnonzero(~used)
            take = min(batch_size - len(members), remaining.size)
            if take > 0:
                filler = rng.choice(remaining, size=take, replace=False)
                used[filler] = True
                members.extend(int(index) for index in filler)
        batches.append(np.asarray(members, dtype=np.int64))

    return batches


def _diverse_batches(
    neighbors: np.ndarray, batch_size: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Round-robin over neighbourhood blocks, so batch-mates come from far apart.

    Built from the same greedy neighbourhood partition, then transposed: taking
    one member from each of `batch_size` different neighbourhood blocks gives a
    batch whose members are, by construction, not each other's teacher
    neighbours. This reuses the neighbourhood construction rather than adding a
    second notion of distance, so `teacher_neighbor` and `teacher_diverse` are
    two readings of one grouping and their difference is not an artefact of two
    different algorithms.

    Exactness is not claimed and is not needed: two members can still be
    neighbours if the greedy blocks overlapped in the corpus. The achieved
    in-batch teacher relevance is measured and reported per arm, which is what
    the study actually reads.
    """
    blocks = _neighbor_batches(neighbors, batch_size, rng)
    # Draining the blocks column-wise makes each output batch take at most one
    # member per block. Blocks are consumed from a list of cursors so a block
    # that runs out early does not shift the alignment of the rest.
    cursors = [0] * len(blocks)
    remaining = sum(block.size for block in blocks)
    batches: list[np.ndarray] = []

    while remaining > 0:
        members: list[int] = []
        for block_index, block in enumerate(blocks):
            if len(members) >= batch_size:
                break
            position = cursors[block_index]
            if position >= block.size:
                continue
            members.append(int(block[position]))
            cursors[block_index] = position + 1
            remaining -= 1
        if not members:
            break
        batches.append(np.asarray(members, dtype=np.int64))

    return batches


class TeacherBatchSampler:
    """A `batch_sampler` for `DataLoader`, reseeded every epoch.

    Yields lists of dataset positions. `set_epoch` must be called before each
    epoch: the grouping is randomized per epoch (a fixed partition would let the
    student memorize which texts are scored together), and the epoch number is
    part of the seed so the sequence is reproducible from `(seed, epoch)` alone.
    """

    def __init__(
        self,
        mode: str,
        neighbors: np.ndarray,
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ):
        if mode not in BATCH_SAMPLERS or mode == "random":
            raise ValueError(
                f"TeacherBatchSampler handles the teacher-informed modes "
                f"{BATCH_SAMPLERS[1:]}, got {mode!r}; 'random' is the loader's "
                "own shuffle and needs no sampler"
            )
        if batch_size < 2:
            raise ValueError(
                f"batch composition is only defined for batch_size >= 2, got {batch_size}"
            )
        self.mode = mode
        # int32 is enough for any corpus this runs on and halves the copy each
        # DataLoader worker forks.
        self.neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._batches: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._batches = None

    def _build(self) -> list[list[int]]:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, 0xB47C])
        )
        builder = (
            _neighbor_batches if self.mode == "teacher_neighbor" else _diverse_batches
        )
        groups = builder(self.neighbors, self.batch_size, rng)
        batches = _shuffled_batches(groups, rng)
        if self.drop_last:
            # Same contract as the loader's own drop_last, and for the same
            # reason: a short final batch optimizes a structurally different
            # objective for a batch-relational loss.
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self._batches is None:
            self._batches = self._build()
        return iter(self._batches)

    def __len__(self) -> int:
        if self._batches is None:
            self._batches = self._build()
        return len(self._batches)


def batch_relevance_stats(
    batches: Sequence[Sequence[int]],
    teacher_topk: np.ndarray,
) -> dict[str, float]:
    """How teacher-relevant the in-batch relations actually came out.

    This is the number that makes the intervention an observation rather than an
    assumption: for each anchor, the share of its batch-mates that are among its
    teacher top-K. Reported per arm so the paper can state the achieved contrast
    instead of the intended one.
    """
    if not len(batches):
        return {}
    k = int(teacher_topk.shape[1])
    membership = [set(int(j) for j in row) for row in teacher_topk]
    hit_rates = []
    recalls = []
    for batch in batches:
        members = [int(index) for index in batch]
        if len(members) < 2:
            continue
        member_set = set(members)
        for anchor in members:
            others = member_set - {anchor}
            if not others:
                continue
            hits = len(others & membership[anchor])
            hit_rates.append(hits / len(others))
            recalls.append(hits / k)
    if not hit_rates:
        return {}
    return {
        # Of the relations this batch exposes for an anchor, what fraction the
        # teacher considers top-K neighbours.
        "in_batch_precision": float(np.mean(hit_rates)),
        # Of the anchor's teacher top-K, what fraction this batch exposed.
        "in_batch_recall": float(np.mean(recalls)),
        "n_batches": float(len(batches)),
    }
