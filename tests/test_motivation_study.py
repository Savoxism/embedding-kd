"""The motivation study's moving parts, pinned where they can fail silently.

Each experiment rests on one claim about its instrument, and each claim below is
the one that would still produce a clean-looking number if it were false:

* a random-support arm must draw *off* the graph and carry no teacher mass,
  otherwise it is a second teacher arm wearing a different label;
* the edge holdout must be symmetric and seed-independent, otherwise the
  "never supervised" relations were supervised from the other endpoint;
* a teacher-informed batch sampler must partition the corpus, otherwise the arms
  differ in how often each anchor is seen as well as in who it is seen with;
* the held-out metrics must actually read the mask they are given, otherwise
  every arm is scored on its own training edges.
"""

import numpy as np
import pytest
import torch

from config import GGPKDConfig
from src.data_utils.batch_samplers import (
    TeacherBatchSampler,
    batch_relevance_stats,
)
from src.distill.geometry import knn_recall, pair_order_accuracy
from src.ggpkd.candidate_sampler import GGPKDCandidateSampler
from src.ggpkd.graph_builder import heldout_edge_mask


# --------------------------------------------------------------------------- #
# Support policies: the random-support control arms
# --------------------------------------------------------------------------- #


def _artifact(n_items=200, width=12, n_scales=1, seed=0):
    """A pool where every anchor has a ragged, strictly positive-mass row."""
    rng = np.random.default_rng(seed)
    pool_indices = np.full((n_items, width), -1, dtype=np.int64)
    pool_probs = np.zeros((n_scales, n_items, width), dtype=np.float32)
    for i in range(n_items):
        degree = int(rng.integers(3, width + 1))
        choices = rng.choice(
            [j for j in range(n_items) if j != i], size=degree, replace=False
        )
        pool_indices[i, :degree] = choices
        mass = rng.random(degree) + 0.1
        pool_probs[:, i, :degree] = (mass / mass.sum()).astype(np.float32)
    return {
        "pool_indices": torch.from_numpy(pool_indices),
        "pool_probs": torch.from_numpy(pool_probs),
        "hard_neg_indices": torch.from_numpy(
            np.full((n_items, 4), -1, dtype=np.int64)
        ),
        "metadata": {"diffusion_scales": (1,)},
    }


def _sampler(policy, quota, artifact=None):
    return GGPKDCandidateSampler(
        artifact=artifact if artifact is not None else _artifact(),
        diffusion_quota=quota,
        hard_neg_k=0,
        random_neg_k=0,
        seed=7,
        support_policy=policy,
    )


def test_corpus_uniform_draws_off_graph_and_carries_no_teacher_mass():
    artifact = _artifact()
    quota = 10
    sampler = _sampler("corpus_uniform", quota, artifact)
    pool = artifact["pool_indices"].numpy()

    off_graph = 0
    for idx in range(0, 200, 17):
        candidates, probs = sampler.sample(idx)
        assert candidates.size == quota
        assert idx not in set(candidates.tolist())
        assert len(set(candidates.tolist())) == candidates.size
        # The whole point of the arm: the columns are not the teacher's choice,
        # and the diffusion target on them is exactly zero, which is why the arm
        # is only defined against relation_target="direct".
        assert float(probs.sum()) == 0.0
        own_row = set(int(j) for j in pool[idx] if j >= 0)
        off_graph += sum(1 for j in candidates.tolist() if j not in own_row)
    # A uniform draw from 200 nodes hitting a <=12-column row is rare; requiring
    # "mostly off-graph" rather than "never" keeps the test about the policy
    # rather than about a lucky seed.
    assert off_graph > 0.8 * quota * len(range(0, 200, 17))


@pytest.mark.parametrize("quota", [12, 6])
def test_rewired_matches_each_anchors_own_degree_up_to_the_budget(quota):
    """Degree-matched, but capped at the budget the teacher arm also gets.

    Matching the raw degree would give this arm more columns than the teacher arm
    it controls for whenever a row is wider than the budget -- and would overrun
    the fixed candidate width, which is how the overrun was found.
    """
    artifact = _artifact()
    pool = artifact["pool_indices"].numpy()
    sampler = _sampler("rewired", quota, artifact)
    saw_capped = False
    for idx in range(0, 200, 13):
        candidates, probs = sampler.sample(idx)
        degree = int((pool[idx] >= 0).sum())
        expected = min(degree, quota)
        drawn = [j for j in candidates.tolist() if j != idx]
        # Short rows are padded with the anchor's own index, which every softmax
        # masks out, so the real support width is the expected one.
        assert len(drawn) == expected
        assert float(probs.sum()) == 0.0
        saw_capped |= degree > quota
    if quota == 6:
        assert saw_capped, "fixture should exercise the capped branch"


def test_corpus_uniform_support_size_matches_the_teacher_arm():
    """The arms are only a control for each other at equal support width."""
    artifact = _artifact()
    quota = 8
    teacher = _sampler("topk", quota, artifact)
    random_support = _sampler("corpus_uniform", quota, artifact)
    for idx in (3, 44, 111):
        teacher_columns, _ = teacher.sample(idx)
        random_columns, _ = random_support.sample(idx)
        assert teacher_columns.size == random_columns.size == quota


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"relation_target": "diffusion"}, "relation_target='direct'"),
        ({"relation_target": "direct", "diffusion_quota": None}, "diffusion_quota"),
        (
            {"relation_target": "direct", "diffusion_quota": 8, "row_weight": 1.0},
            "row_weight=0",
        ),
    ],
)
def test_off_graph_policies_refuse_their_broken_configurations(overrides, message):
    settings = {"support_policy": "corpus_uniform", "diffusion_quota": 8, "row_weight": 0.0}
    settings.update(overrides)
    with pytest.raises(ValueError, match=message):
        GGPKDConfig(**settings)


# --------------------------------------------------------------------------- #
# Edge holdout
# --------------------------------------------------------------------------- #


def test_holdout_is_symmetric_and_reproducible():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 5000, size=4000)
    cols = rng.integers(0, 5000, size=4000)
    forward = heldout_edge_mask(rows, cols, seed=99, frac=0.2)
    backward = heldout_edge_mask(cols, rows, seed=99, frac=0.2)
    # Asymmetry is the failure that keeps the numbers looking fine: the edge is
    # then still supervised from the other endpoint by L_row.
    assert np.array_equal(forward, backward)
    assert np.array_equal(forward, heldout_edge_mask(rows, cols, seed=99, frac=0.2))


def test_holdout_takes_about_the_requested_fraction():
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 20000, size=200000)
    cols = rng.integers(0, 20000, size=200000)
    rate = float(heldout_edge_mask(rows, cols, seed=5, frac=0.2).mean())
    assert 0.19 < rate < 0.21


def test_holdout_of_zero_withholds_nothing():
    rows = np.arange(100)
    cols = np.arange(100)[::-1]
    assert not heldout_edge_mask(rows, cols, seed=5, frac=0.0).any()


def test_different_seeds_choose_different_edges():
    rng = np.random.default_rng(2)
    rows = rng.integers(0, 5000, size=20000)
    cols = rng.integers(0, 5000, size=20000)
    first = heldout_edge_mask(rows, cols, seed=1, frac=0.2)
    second = heldout_edge_mask(rows, cols, seed=2, frac=0.2)
    agreement = float((first == second).mean())
    # Two independent 20% masks agree on ~68% of pairs by chance; anything near
    # 1.0 would mean the seed is not reaching the hash.
    assert 0.6 < agreement < 0.75


# --------------------------------------------------------------------------- #
# Batch composition
# --------------------------------------------------------------------------- #


def _teacher_topk(n_items=240, k=8, block=24):
    """A corpus of disjoint semantic blocks, so "neighbour" is unambiguous."""
    neighbors = np.zeros((n_items, k), dtype=np.int64)
    for i in range(n_items):
        start = (i // block) * block
        members = [j for j in range(start, start + block) if j != i]
        neighbors[i] = members[:k]
    return neighbors


@pytest.mark.parametrize("mode", ["teacher_neighbor", "teacher_diverse"])
def test_teacher_batch_samplers_partition_the_corpus(mode):
    neighbors = _teacher_topk()
    sampler = TeacherBatchSampler(
        mode=mode, neighbors=neighbors, batch_size=8, seed=3, drop_last=False
    )
    batches = list(sampler)
    flat = [index for batch in batches for index in batch]
    # Every anchor exactly once per epoch: the arms must differ in grouping only,
    # not in how many gradient updates each example contributes to.
    assert sorted(flat) == list(range(neighbors.shape[0]))
    assert all(len(batch) <= 8 for batch in batches)


def test_teacher_batch_samplers_are_reseeded_per_epoch():
    neighbors = _teacher_topk()
    sampler = TeacherBatchSampler(
        mode="teacher_neighbor", neighbors=neighbors, batch_size=8, seed=3
    )
    first = [list(batch) for batch in sampler]
    sampler.set_epoch(1)
    second = [list(batch) for batch in sampler]
    assert first != second

    same = TeacherBatchSampler(
        mode="teacher_neighbor", neighbors=neighbors, batch_size=8, seed=3
    )
    assert [list(batch) for batch in same] == first


def test_neighbor_batching_is_more_teacher_relevant_than_diverse():
    """The contrast the intervention exists to create, measured not assumed."""
    neighbors = _teacher_topk()
    stats = {}
    for mode in ("teacher_neighbor", "teacher_diverse"):
        sampler = TeacherBatchSampler(
            mode=mode, neighbors=neighbors, batch_size=8, seed=11
        )
        stats[mode] = batch_relevance_stats(list(sampler), neighbors)
    assert (
        stats["teacher_neighbor"]["in_batch_precision"]
        > stats["teacher_diverse"]["in_batch_precision"]
    )


def test_random_mode_has_no_sampler():
    with pytest.raises(ValueError, match="random"):
        TeacherBatchSampler(
            mode="random", neighbors=_teacher_topk(), batch_size=8, seed=0
        )


# --------------------------------------------------------------------------- #
# Held-out geometry metrics
# --------------------------------------------------------------------------- #


def _cosines(embeddings):
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return normalized @ normalized.t()


def test_a_perfect_student_scores_one_on_both_metrics():
    teacher = torch.randn(60, 16)
    cosines = _cosines(teacher)
    off_diagonal = ~torch.eye(60, dtype=torch.bool)
    assert knn_recall(cosines, cosines, k=5, restrict=off_diagonal) == 1.0
    assert (
        pair_order_accuracy(cosines, cosines, restrict=off_diagonal, n_triplets=5000)
        == 1.0
    )


def test_an_unrelated_student_scores_near_chance_on_ordering():
    torch.manual_seed(0)
    teacher = _cosines(torch.randn(80, 16))
    student = _cosines(torch.randn(80, 16))
    off_diagonal = ~torch.eye(80, dtype=torch.bool)
    accuracy = pair_order_accuracy(
        teacher, student, restrict=off_diagonal, n_triplets=20000
    )
    assert 0.4 < accuracy < 0.6


def test_the_restrict_mask_actually_removes_pairs():
    """A mask that were ignored would score every arm on its own training edges."""
    torch.manual_seed(1)
    teacher = _cosines(torch.randn(50, 16))
    student = teacher.clone()
    # Corrupt the student everywhere *except* one held-out block, then score only
    # that block: a metric that reads the mask sees a perfect student.
    held_out = torch.zeros(50, 50, dtype=torch.bool)
    held_out[:10, 10:20] = True
    noise = torch.randn(50, 50) * 5.0
    student = torch.where(held_out, teacher, teacher + noise)
    assert pair_order_accuracy(
        teacher, student, restrict=held_out, n_triplets=20000
    ) == 1.0


# --------------------------------------------------------------------------- #
# Held-out relation sets (exp3)
# --------------------------------------------------------------------------- #


def test_relation_sets_exclude_everything_any_arm_was_trained_on():
    """The masks are the experiment's claim; a leak here fakes the whole result.

    `supervised` is the union of the anchor's diffusion pool and its transition
    row. If either were missing from the exclusion, an arm would be scored partly
    on its own training targets and would appear to generalize.
    """
    from scripts.exp.heldout_geometry import build_masks

    n_items, width, graph_k = 400, 10, 12
    rng = np.random.default_rng(3)
    pool = np.full((n_items, width), -1, dtype=np.int64)
    transition = np.full((n_items, width), -1, dtype=np.int64)
    for i in range(n_items):
        choices = rng.choice(
            [j for j in range(n_items) if j != i], size=width, replace=False
        )
        pool[i] = choices
        transition[i] = choices
    artifact = {
        "pool_indices": torch.from_numpy(pool),
        "transition_neighbors": torch.from_numpy(transition),
        "metadata": {"graph_k": graph_k},
    }
    teacher = torch.nn.functional.normalize(torch.randn(n_items, 24), p=2, dim=-1)
    anchors = np.arange(0, n_items, 7)

    masks = build_masks(
        artifact,
        anchors,
        teacher,
        graph_k=graph_k,
        nonlocal_mult=5,
        holdout_frac=0.2,
        holdout_seed=12345,
        device="cpu",
    )
    assert set(masks) == {"unsupervised", "nonlocal", "heldout_edges"}
    for position, anchor in enumerate(anchors):
        trained = set(int(j) for j in pool[anchor]) | {int(anchor)}
        for name in ("unsupervised", "nonlocal"):
            scored = set(torch.nonzero(masks[name][position]).flatten().tolist())
            assert not (scored & trained), f"{name} leaked a supervised pair"
    # The holdout set has to be non-empty, or exp3's headline metric silently
    # reports on nothing.
    assert int(masks["heldout_edges"].sum()) > 0


def test_no_holdout_artifact_yields_no_holdout_set():
    from scripts.exp.heldout_geometry import build_masks

    n_items = 200
    pool = np.full((n_items, 5), -1, dtype=np.int64)
    for i in range(n_items):
        pool[i] = [(i + offset) % n_items for offset in (1, 2, 3, 4, 5)]
    artifact = {
        "pool_indices": torch.from_numpy(pool),
        "transition_neighbors": torch.from_numpy(pool),
        "metadata": {"graph_k": 5},
    }
    teacher = torch.nn.functional.normalize(torch.randn(n_items, 8), p=2, dim=-1)
    masks = build_masks(
        artifact,
        np.arange(0, n_items, 11),
        teacher,
        graph_k=5,
        nonlocal_mult=4,
        holdout_frac=0.0,
        holdout_seed=0,
        device="cpu",
    )
    assert "heldout_edges" not in masks


# --------------------------------------------------------------------------- #
# End to end through the real dataset, collate and criterion
# --------------------------------------------------------------------------- #


class _CharTokenizer:
    """Enough tokenizer for the collate; no model download in a unit test."""

    pad_token_id = 0

    def __call__(self, texts, truncation=True, max_length=128, **kwargs):
        return {
            "input_ids": [
                [(ord(c) % 60) + 2 for c in text[:max_length]] or [2] for text in texts
            ]
        }


def _pipeline_batch(policy, quota, corpus_size=120, batch_size=8):
    """One collated batch, produced the way training produces it."""
    from src.data_utils.ggpkd_dataset import GGPKDCollate, TextPairWithTeacherAndGGPKD

    artifact = _artifact(n_items=corpus_size, width=10)
    sampler = GGPKDCandidateSampler(
        artifact=artifact,
        diffusion_quota=quota,
        hard_neg_k=0,
        random_neg_k=0,
        seed=5,
        support_policy=policy,
    )
    texts = [f"text {i} on subject {i % 9}" for i in range(corpus_size)]
    dataset = TextPairWithTeacherAndGGPKD(
        anchor_texts=texts,
        teacher_cls=torch.randn(corpus_size, 12),
        sampler=sampler,
        batch_local=False,
    )
    collate = GGPKDCollate(
        _CharTokenizer(),
        "single_cls",
        32,
        corpus_texts=texts,
        batch_local=False,
        n_scales=1,
    )
    return collate([dataset[i] for i in range(batch_size)]), artifact


@pytest.mark.parametrize("policy", ["topk", "corpus_uniform", "rewired"])
def test_every_support_policy_survives_the_collate_and_the_criterion(policy):
    """The random-support arms have to produce a finite gradient, not just a draw.

    Their teacher mass is zero everywhere, which is a shape the diffusion path
    never sees in a normal run: the target row sums to zero, the renormalization
    divides by a clamped zero, and only `relation_target='direct'` puts a real
    distribution back. This runs that path.
    """
    from src.criterions.ggpkd_distillation import GGPKDDistillation

    batch, artifact = _pipeline_batch(policy, quota=6)
    n_items = artifact["pool_indices"].shape[0]
    batch_size = batch["idx"].numel()
    candidate_size = batch["candidate_idx"].shape[1]

    criterion = GGPKDDistillation(
        diffusion_scales=(1,),
        teacher_embeddings=torch.randn(n_items, 12),
        calibration_mode="none",
        relation_target="direct",
        row_weight=0.0,
        row_temps=torch.full((n_items,), 0.05),
        transition_neighbors=artifact["pool_indices"],
        transition_probs=artifact["pool_probs"][0],
    )
    anchors = torch.randn(batch_size, 12, requires_grad=True)
    candidates = torch.randn(batch_size * candidate_size, 12, requires_grad=True)
    loss, metrics = criterion(
        anchor_embeddings=anchors,
        candidate_embeddings=candidates,
        teacher_probs=batch["teacher_probs"],
        candidate_idx=batch["candidate_idx"],
        anchor_idx=batch["idx"],
    )
    assert torch.isfinite(loss) and loss.item() > 0
    # The ambient scale is out, so the whole objective is the graph group.
    assert metrics["loss_amb"] == 0.0
    loss.backward()
    assert torch.isfinite(anchors.grad).all()
    assert torch.isfinite(candidates.grad).all()


def test_in_batch_arm_is_temperature_matched_to_the_teacher_arm():
    """Both score at the anchor's own tau_i, which is what makes E1 one-factor.

    The older form of the in-batch baseline (`ambient_only`) scores at a single
    global temperature, so an arm comparison against a teacher-support arm would
    differ in temperature as well as in support. This pins that `batch_local` can
    now take the per-row target instead.
    """
    from config import GGPKDConfig

    config = GGPKDConfig(
        batch_local=True, relation_target="direct", calibration_mode="none"
    )
    assert config.relation_target == "direct"

    with pytest.raises(ValueError, match="ambient_only"):
        GGPKDConfig(batch_local=True, relation_target="diffusion")
