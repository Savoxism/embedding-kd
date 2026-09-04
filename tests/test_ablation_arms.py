"""The ablation switches: each arm must be exactly one thing away from the method.

An ablation is only evidence if the arm differs from the full model in the way it
claims to and in no other way. These tests pin the "no other way" half, which is
the half that fails silently: a policy that quietly shrinks the candidate budget,
or a deletion arm that also changes the row set, produces a clean-looking number
that means something else.
"""

import numpy as np
import pytest
import torch

from config import GGPKDConfig
from src.criterions.ggpkd_distillation import GGPKDDistillation
from src.ggpkd.candidate_sampler import GGPKDCandidateSampler
from src.ggpkd.graph_builder import _build_transition, _hubness_stats
from src.ggpkd.policy import SUPPORT_POLICIES

N_ITEMS = 160
POOL_WIDTH = 40
N_SCALES = 3
QUOTA, HARD_K, RANDOM_K = 9, 5, 4


@pytest.fixture
def artifact():
    rng = np.random.default_rng(0)
    pool_indices = np.stack(
        [rng.choice(N_ITEMS, size=POOL_WIDTH, replace=False) for _ in range(N_ITEMS)]
    )
    # Peaked rows, so the deterministic and the proportional draws can differ.
    shape = 1.0 / np.arange(1, POOL_WIDTH + 1) ** 1.5
    probs = np.stack(
        [
            np.stack([rng.permutation(shape) for _ in range(N_ITEMS)])
            for _ in range(N_SCALES)
        ]
    )
    probs /= probs.sum(axis=-1, keepdims=True)
    return {
        "pool_indices": torch.from_numpy(pool_indices),
        "pool_probs": torch.from_numpy(probs.astype(np.float32)),
        "hard_neg_indices": torch.from_numpy(
            np.stack([rng.choice(N_ITEMS, size=20, replace=False) for _ in range(N_ITEMS)])
        ),
        "metadata": {"diffusion_scales": (1, 2, 4)},
    }


def _sampler(artifact, policy):
    return GGPKDCandidateSampler(
        artifact=artifact,
        diffusion_quota=QUOTA,
        hard_neg_k=HARD_K,
        random_neg_k=RANDOM_K,
        seed=42,
        support_policy=policy,
    )


@pytest.mark.parametrize("policy", SUPPORT_POLICIES)
def test_every_policy_spends_the_same_budget(artifact, policy):
    """The comparison is only fixed-budget if the budget is literally identical."""
    sampler = _sampler(artifact, policy)
    for idx in (0, 7, 63):
        candidates, teacher_probs = sampler.sample(idx)
        assert candidates.shape == (QUOTA + HARD_K + RANDOM_K,)
        assert teacher_probs.shape == (N_SCALES, QUOTA + HARD_K + RANDOM_K)
        assert len(set(candidates.tolist())) == candidates.size
        assert idx not in candidates.tolist()


def test_ranked_policies_are_deterministic_and_sampling_controls_explore(artifact):
    """Ranked policies repeat; the two stochastic controls keep exploring."""
    supports = {}
    for policy in SUPPORT_POLICIES:
        sampler = _sampler(artifact, policy)
        per_epoch = []
        for epoch in range(3):
            sampler.set_epoch(epoch)
            per_epoch.append(set(sampler.sample(11)[0][:QUOTA].tolist()))
        supports[policy] = per_epoch

    for policy in ("topk", "local_topk"):
        assert supports[policy][0] == supports[policy][1] == supports[policy][2]
    for policy in ("proportional", "uniform"):
        assert supports[policy][0] != supports[policy][1], policy


def test_local_topk_uses_only_the_one_step_row(artifact):
    """The clean no-diffusion control must never spend quota on r>1 mass."""
    sampler = _sampler(artifact, "local_topk")
    candidates, targets = sampler.sample(11)
    support = candidates[:QUOTA]

    pool = artifact["pool_indices"][11].numpy()
    p1 = artifact["pool_probs"][0, 11].numpy()
    expected_positions = np.argsort(-p1)[:QUOTA]
    expected = pool[expected_positions]

    assert np.array_equal(support, expected)
    assert np.all(targets[0, :QUOTA] > 0)


def test_topk_covers_more_teacher_mass_per_epoch_than_uniform(artifact):
    """The policies must actually order by teacher relevance, not just differ."""
    mass = {}
    for policy in ("topk", "proportional", "uniform"):
        sampler = _sampler(artifact, policy)
        sampler.set_epoch(0)
        mass[policy] = float(sampler.sample(11)[1][0, :QUOTA].sum())
    assert mass["topk"] > mass["proportional"]
    assert mass["topk"] > mass["uniform"]


# --------------------------------------------------------------------------- #
# Criterion arms
# --------------------------------------------------------------------------- #


def _criterion_inputs(batch=4, candidates=12, dim=16, n_items=60, scales=3):
    torch.manual_seed(0)
    teacher = torch.randn(n_items, dim)
    probs = torch.rand(batch, scales, candidates)
    probs[:, :, candidates // 2 :] = 0.0
    probs /= probs.sum(-1, keepdim=True)
    candidate_idx = torch.stack([torch.randperm(n_items)[:candidates] for _ in range(batch)])
    anchor_idx = torch.arange(batch)
    for row in range(batch):
        candidate_idx[row][candidate_idx[row] == row] = (row + 31) % n_items
    return {
        "teacher": teacher,
        "anchor": torch.randn(batch, dim, requires_grad=True),
        "candidates": torch.randn(batch * candidates, dim, requires_grad=True),
        "probs": probs,
        "candidate_idx": candidate_idx,
        "anchor_idx": anchor_idx,
        "graph": {
            "transition_neighbors": torch.randint(0, n_items, (n_items, 5)).int(),
            "transition_probs": torch.rand(n_items, 5).softmax(-1),
            "row_temps": torch.full((n_items,), 0.05),
        },
    }


@pytest.mark.parametrize(
    "arm,kwargs,scales",
    [
        ("full", {}, (1, 2, 4)),
        ("no_ambient", {"no_teacher": True}, (1, 2, 4)),
        ("direct_target", {"relation_target": "direct"}, (1, 2, 4)),
        ("local_only", {}, (1,)),
    ],
)
def test_every_criterion_arm_produces_a_finite_gradient(arm, kwargs, scales):
    data = _criterion_inputs()
    no_teacher = kwargs.pop("no_teacher", False)
    criterion = GGPKDDistillation(
        diffusion_scales=scales,
        teacher_embeddings=None if no_teacher else data["teacher"],
        row_weight=1.0,
        **data["graph"],
        **kwargs,
    )
    probs = data["probs"][:, : len(scales)]
    probs = probs / probs.sum(-1, keepdim=True)
    loss, metrics = criterion(
        data["anchor"],
        data["candidates"],
        probs,
        candidate_idx=data["candidate_idx"],
        anchor_idx=data["anchor_idx"],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(data["anchor"].grad).all()
    # The ambient scale is present exactly when the arm says it is.
    assert (metrics["loss_amb"] > 0.0) is (not no_teacher)


def test_direct_relation_target_needs_the_teacher_bank():
    data = _criterion_inputs()
    with pytest.raises(ValueError, match="teacher_embeddings"):
        GGPKDDistillation(
            diffusion_scales=(1, 2, 4),
            teacher_embeddings=None,
            relation_target="direct",
            row_weight=1.0,
            **data["graph"],
        )


def test_direct_relation_target_changes_the_target_not_the_column_set():
    """S3's contract: same columns, same weight, different target.

    Checked through the diagnostics rather than by inspecting internals --
    `candidates_per_anchor` is the number of unmasked columns the diffusion
    softmax runs over, so an equal count is the evidence that only the target
    moved.
    """
    data = _criterion_inputs()
    metrics = {}
    for name, target in (("diffusion", "diffusion"), ("direct", "direct")):
        criterion = GGPKDDistillation(
            diffusion_scales=(1, 2, 4),
            teacher_embeddings=data["teacher"],
            relation_target=target,
            row_weight=1.0,
            **data["graph"],
        )
        _, metrics[name] = criterion(
            data["anchor"],
            data["candidates"],
            data["probs"],
            candidate_idx=data["candidate_idx"],
            anchor_idx=data["anchor_idx"],
        )
    assert metrics["direct"]["candidates_per_anchor"] == pytest.approx(
        metrics["diffusion"]["candidates_per_anchor"]
    )
    assert metrics["direct"]["loss_amb"] == pytest.approx(metrics["diffusion"]["loss_amb"])
    assert metrics["direct"]["loss_nbr"] != pytest.approx(metrics["diffusion"]["loss_nbr"])


# --------------------------------------------------------------------------- #
# Graph arms
# --------------------------------------------------------------------------- #


def _knn_inputs(n=80, k=8):
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(n, 12))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosine = embeddings @ embeddings.T
    np.fill_diagonal(cosine, -np.inf)
    order = np.argsort(-cosine, axis=1)[:, :k]
    scores = np.take_along_axis(cosine, order, axis=1)
    return order.astype(np.int64), scores.astype(np.float32)


def test_knn_modes_nest_by_construction():
    """mutual subset-of directed subset-of symmetrized, edge for edge."""
    top_indices, top_scores = _knn_inputs()
    edges = {}
    for mode in ("mutual", "directed", "symmetrized"):
        neighbors, _, _, _, _, _ = _build_transition(
            top_indices, top_scores, graph_k=8, graph_temp=0.05,
            perplexity=None, knn_mode=mode,
        )
        edges[mode] = {(i, int(j)) for i, row in enumerate(neighbors) for j in row}
    assert edges["mutual"] <= edges["directed"] <= edges["symmetrized"]
    assert len(edges["mutual"]) < len(edges["symmetrized"])


def test_hubness_is_reported_and_orders_the_modes():
    """G1's evidence is the indegree tail, so it must actually be recorded."""
    top_indices, top_scores = _knn_inputs()
    stats = {}
    for mode in ("mutual", "symmetrized"):
        neighbors, _, _, _, _, _ = _build_transition(
            top_indices, top_scores, graph_k=8, graph_temp=0.05,
            perplexity=None, knn_mode=mode,
        )
        stats[mode] = _hubness_stats(neighbors)
    for mode in stats:
        assert {"indegree_max", "indegree_p99", "indegree_gini",
                "hub_edge_share_top1pct"} <= stats[mode].keys()
    # Mutuality removes the edges into nodes that retrieve nothing back, so it
    # cannot leave a *larger* indegree tail than the union does.
    assert stats["mutual"]["indegree_max"] <= stats["symmetrized"]["indegree_max"]


def test_unknown_knn_mode_is_rejected():
    top_indices, top_scores = _knn_inputs()
    with pytest.raises(ValueError, match="knn_mode"):
        _build_transition(
            top_indices, top_scores, graph_k=8, graph_temp=0.05,
            perplexity=None, knn_mode="reciprocal",
        )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_config_defaults_are_the_unablated_method():
    config = GGPKDConfig()
    assert config.support_policy == "topk"
    assert config.relation_target == "diffusion"
    assert config.use_ambient is True
    assert config.knn_mode == "mutual"


def test_config_rejects_the_one_impossible_combination():
    with pytest.raises(ValueError, match="teacher bank"):
        GGPKDConfig(relation_target="direct", use_ambient=False)


# --------------------------------------------------------------------------- #
# Batch-local baseline (S1)
# --------------------------------------------------------------------------- #


class _CharTokenizer:
    pad_token_id = 0

    def __call__(self, texts, truncation=True, max_length=128, **kwargs):
        return {
            "input_ids": [
                [(ord(c) % 60) + 2 for c in text[:max_length]] or [2] for text in texts
            ]
        }


def _batch_local_batch(corpus_size=40, batch_size=6, n_scales=3):
    from src.data_utils.dataset_cache import GGPKDCollate, TextPairWithTeacherAndGGPKD

    texts = [f"sentence {i} about topic {i % 7}" for i in range(corpus_size)]
    dataset = TextPairWithTeacherAndGGPKD(
        texts, torch.randn(corpus_size, 8), sampler=None, batch_local=True
    )
    collate = GGPKDCollate(
        _CharTokenizer(), "single_cls", 32, corpus_texts=texts,
        batch_local=True, n_scales=n_scales,
    )
    return collate([dataset[i] for i in range(batch_size)])


def test_batch_local_needs_no_sampler_at_all():
    """The arm forms no graph relations, so it must not draw candidates either.

    `sampler=None` is the assertion: if the dataset still reached for a draw it
    would raise here rather than quietly paying for candidates it discards.
    """
    batch = _batch_local_batch()
    assert batch["candidate_idx"].shape == (6, 6)


def test_batch_local_candidates_are_exactly_the_batch():
    batch = _batch_local_batch(batch_size=6)
    for row in range(6):
        assert torch.equal(batch["candidate_idx"][row], batch["idx"])
    encoded = sum(chunk["input_ids"].size(0) for chunk in batch["candidate_chunks"])
    # The candidate encode is the batch and nothing else -- that cheapness is the
    # baseline's defining property and what the encoder-budget columns report.
    assert encoded == 6


def test_batch_local_carries_no_diffusion_mass():
    batch = _batch_local_batch(n_scales=3)
    assert batch["teacher_probs"].shape == (6, 3, 6)
    assert float(batch["teacher_probs"].abs().sum()) == 0.0


def test_batch_local_rejects_a_batch_too_small_to_have_a_relation():
    from src.data_utils.dataset_cache import GGPKDCollate, TextPairWithTeacherAndGGPKD

    texts = ["only one"]
    dataset = TextPairWithTeacherAndGGPKD(
        texts, torch.randn(1, 8), sampler=None, batch_local=True
    )
    collate = GGPKDCollate(
        _CharTokenizer(), "single_cls", 32, corpus_texts=texts,
        batch_local=True, n_scales=1,
    )
    with pytest.raises(ValueError, match="at least two"):
        collate([dataset[0]])


def test_ambient_only_gives_the_ambient_scale_the_whole_weight():
    """The reason the graph group is dropped rather than zeroed.

    A scale with a zero target still holds its weight in the normalization. Left
    in, the baseline's loss would be scaled by the ambient share alone -- 0.36 at
    R={1,2,4} -- which is a different effective learning rate, not a different
    objective. Under `ambient_only` the loss must equal the ambient KL exactly.
    """
    data = _criterion_inputs()
    zero_targets = torch.zeros_like(data["probs"])
    criterion = GGPKDDistillation(
        diffusion_scales=(1, 2, 4),
        teacher_embeddings=data["teacher"],
        relation_target="ambient_only",
        row_weight=0.0,
        **data["graph"],
    )
    loss, metrics = criterion(
        data["anchor"], data["candidates"], zero_targets,
        candidate_idx=data["candidate_idx"], anchor_idx=data["anchor_idx"],
    )
    assert loss.item() == pytest.approx(metrics["loss_amb"], rel=1e-5)
    assert metrics["loss_nbr"] == 0.0
    assert metrics["loss_row"] == 0.0

    # The same targets under the normal objective are scaled down by the dead
    # group's weight -- which is exactly the failure mode being avoided.
    scaled = GGPKDDistillation(
        diffusion_scales=(1, 2, 4),
        teacher_embeddings=data["teacher"],
        row_weight=0.0,
        **data["graph"],
    )
    scaled_loss, scaled_metrics = scaled(
        data["anchor"], data["candidates"], zero_targets,
        candidate_idx=data["candidate_idx"], anchor_idx=data["anchor_idx"],
    )
    assert scaled_loss.item() < 0.5 * loss.item()
    assert scaled_metrics["loss_amb"] == pytest.approx(metrics["loss_amb"], rel=1e-5)


def test_ambient_only_is_finite_and_differentiable():
    data = _criterion_inputs()
    criterion = GGPKDDistillation(
        diffusion_scales=(1, 2, 4),
        teacher_embeddings=data["teacher"],
        relation_target="ambient_only",
        row_weight=1.0,
        **data["graph"],
    )
    criterion.use_row_loss = True
    loss, _ = criterion(
        data["anchor"], data["candidates"], torch.zeros_like(data["probs"]),
        candidate_idx=data["candidate_idx"], anchor_idx=data["anchor_idx"],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(data["anchor"].grad).all()


def test_ambient_only_needs_the_teacher_bank():
    data = _criterion_inputs()
    with pytest.raises(ValueError, match="teacher bank|teacher_embeddings"):
        GGPKDDistillation(
            diffusion_scales=(1, 2, 4),
            teacher_embeddings=None,
            relation_target="ambient_only",
            row_weight=0.0,
            **data["graph"],
        )


def test_config_forces_batch_local_onto_the_ambient_only_objective():
    with pytest.raises(ValueError, match="ambient_only"):
        GGPKDConfig(batch_local=True)
    with pytest.raises(ValueError, match="at least two"):
        GGPKDConfig(batch_local=True, relation_target="ambient_only", batch_size=1)
    config = GGPKDConfig(batch_local=True, relation_target="ambient_only")
    assert config.batch_local is True
