import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import main
from distiller import KnowledgeDistiller, add_domain_averages
from src.distill.checkpointing import save_student_weights
from src.criterions.heatgeo_distillation import HeatGeoDistillation
from src.heatgeo.candidate_sampler import HeatGeoCandidateSampler
from src.heatgeo.graph_builder import _entropic_affinity, _mass_prefix, _softmax_at
from src.heatgeo.policy import (
    FIXED_BANDWIDTH_TEMP,
    candidate_budget,
    hard_negative_pool_size,
    normalized_diffusion_weights,
)


def test_heatgeo_cli_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--method",
            "heatgeo",
            "--row_weight",
            "0.8",
            "--row_start_epoch",
            "2",
            "--perplexity",
            "45",
            "--diffusion_quota",
            "20",
            "--hard_neg_k",
            "30",
            "--random_neg_k",
            "46",
            "--cache_path",
            "cache/teacher.pt",
            "--heatgeo_cache_path",
            "cache/graph.pt",
            "--pooling_method",
            "last_token",
            "--weights_dir",
            "weights",
            "--final_weights_only",
        ],
    )

    args = main.parse_args()
    config = main.get_config(args.method, args)

    assert config.row_weight == 0.8
    assert config.row_start_epoch == 2
    assert config.eval_every == 0
    assert not hasattr(config, "walk_temp")
    assert config.perplexity == 45
    assert (
        candidate_budget(config.diffusion_quota, config.hard_neg_k, config.random_neg_k)
        == 96
    )
    for removed in (
        "graph_temp",
        "scale_weights",
        "direct_weight",
        "candidate_size",
        "hard_neg_pool",
        "share_in_batch",
        "resample_candidates_per_epoch",
        "stochastic_candidates",
        "dedup_corpus",
        "walk_non_backtracking",
        "row_mode",
        "row_ambient",
        "num_walks",
        "walk_length",
        "diag_topk",
        "eps_norm",
        "encode_chunk_size",
    ):
        assert not hasattr(config, removed)
    assert config.cache_path == "cache/teacher.pt"
    assert config.heatgeo_cache_path == "cache/graph.pt"
    assert config.pooling_method == "last_token"
    assert config.weights_dir == "weights"
    assert config.final_weights_only is True


def test_evaluation_table_renders_every_family(capsys):
    """Nothing called this before, so a broken signature stayed invisible."""
    from src.distill.benchmarks import print_evaluation_table

    results = {
        "classification": {"data/test_set/emotion_test.csv": {"accuracy": 0.7,
                                                              "f1": 0.62}},
        "pair": {"data/test_set/wic_test.csv": {"accuracy": 0.62,
                                                "average_precision": 0.65}},
        "sts": {"data/test_set/stsb_test.csv": 0.766},
    }
    print_evaluation_table(current_epoch=3, split="validation", results=results)

    out = capsys.readouterr().out
    assert "VALIDATION - EPOCH 4" in out
    for token in ("emotion", "wic", "stsb", "MEAN", "F1", "AP", "Spearman"):
        assert token in out, f"{token} missing from the table"

    print_evaluation_table(current_epoch=0, split="test", results=results)
    assert "FINAL TEST" in capsys.readouterr().out


def test_domain_averages_follow_the_paper_metric_protocol():
    results = {
        "classification": {
            "data/test_set/banking77_test.csv": {"accuracy": 0.01, "f1": 0.9},
            "data/test_set/emotion_test.csv": {"accuracy": 0.02, "f1": 0.6},
            "data/test_set/tweet_test.csv": {"accuracy": 0.03, "f1": 0.7},
        },
        "pair": {
            "data/test_set/mrpc_test.csv": {
                "accuracy": 0.04,
                "average_precision": 0.8,
            },
            "data/test_set/scitail_test.csv": {
                "accuracy": 0.05,
                "average_precision": 0.75,
            },
            "data/test_set/wic_test.csv": {
                "accuracy": 0.06,
                "average_precision": 0.65,
            },
        },
        "sts": {
            "data/test_set/sick_test.csv": 0.72,
            "data/test_set/sts12_test.csv": 0.68,
            "data/test_set/stsb_test.csv": 0.74,
        },
    }

    enriched = add_domain_averages(results)

    assert enriched["avg_in"] == 66.33
    assert enriched["avg_out"] == 75.83
    assert enriched["avg"] == 72.67
    assert enriched["classification"] is results["classification"]


def test_rkd_cli_uses_paper_defaults_and_accepts_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--method",
            "rkd",
            "--rkd_distance_weight",
            "0.5",
            "--rkd_angle_weight",
            "3.0",
            "--cache_path",
            "cache/rkd/teacher.pt",
            "--pooling_method",
            "mean",
        ],
    )

    args = main.parse_args()
    config = main.get_config(args.method, args)

    assert config.distill_method == "rkd"
    assert config.rkd_distance_weight == 0.5
    assert config.rkd_angle_weight == 3.0
    assert config.w_task == 0.0
    assert config.batch_size == 128
    assert config.epochs == 80
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 1e-5
    assert config.rkd_lr_decay_epochs == (40, 60)
    assert config.cache_path == "cache/rkd/teacher.pt"
    assert config.pooling_method == "mean"


def test_heatgeo_temperature_ties():
    criterion = HeatGeoDistillation(
        diffusion_scales=(1, 2, 4),
        row_weight=0.0,  # L_rel only: no graph passed, so L_row must be off
    )
    # The scalar baseline is fixed; omega_r and tau_r are derived from the scales.
    assert criterion.scale_temps.tolist() == pytest.approx(
        [
            FIXED_BANDWIDTH_TEMP,
            FIXED_BANDWIDTH_TEMP * 2**0.5,
            FIXED_BANDWIDTH_TEMP * 2.0,
        ]
    )
    assert criterion.scale_weights.tolist() == pytest.approx([1.0, 0.5, 0.25])
    assert criterion.direct_weight.item() == pytest.approx(1.0)

    for removed in (
        "scale_weights",
        "direct_weight",
        "share_in_batch",
        "graph_temp",
        "scale_temps",
        "broad_scale_temps",
        "walk_temp",
        "walk_weight",
        "row_temp",
        "mass_weight",
        "geo_weight",
        "sym_weight",
        "direct_student_temp",
    ):
        with pytest.raises(ValueError, match=removed):
            HeatGeoDistillation(
                row_weight=0.0,
                **{removed: 0.07},
            )


def test_canonical_policy_preserves_the_previous_resolved_values():
    assert candidate_budget(14, 26, 26) == 66
    assert hard_negative_pool_size(200) == 200
    np.testing.assert_allclose(
        normalized_diffusion_weights((1, 2, 4)),
        np.asarray([1.0, 0.5, 0.25]) / 1.75,
    )


def test_heatgeo_relational_loss_reports_semantic_decomposition():
    """Renaming the scale groups must not change the optimized objective."""
    torch.manual_seed(7)
    teacher_bank = F.normalize(torch.randn(6, 4), dim=-1)
    anchor_embeddings = F.normalize(torch.randn(2, 4), dim=-1)
    candidate_embeddings = F.normalize(torch.randn(2, 3, 4), dim=-1)
    candidate_idx = torch.tensor([[1, 2, 3], [0, 4, 5]], dtype=torch.long)
    anchor_idx = torch.tensor([0, 1], dtype=torch.long)
    teacher_probs = torch.rand(2, 3, 3)

    criterion = HeatGeoDistillation(
        diffusion_scales=(1, 2, 4),
        teacher_embeddings=teacher_bank,
        row_weight=0.0,
    )
    loss, metrics = criterion(
        anchor_embeddings=anchor_embeddings,
        candidate_embeddings=candidate_embeddings,
        teacher_probs=teacher_probs,
        candidate_idx=candidate_idx,
        anchor_idx=anchor_idx,
    )

    # Raw semantic terms: ambient r=0, neighbor r=1, and the normalized
    # multi-hop group (2/3 at r=2 and 1/3 at r=4).
    assert metrics["loss_amb"] == pytest.approx(metrics["kl_amb"], rel=1e-6)
    assert metrics["loss_nbr"] == pytest.approx(metrics["kl_nbr"], rel=1e-6)
    expected_diff = (2.0 / 3.0) * metrics["kl_diff_r2"] + (1.0 / 3.0) * metrics[
        "kl_diff_r4"
    ]
    assert metrics["loss_diff"] == pytest.approx(expected_diff, rel=1e-6)

    # The original raw weights [1, 1, 1/2, 1/4] normalize to the grouped
    # coefficients [4/11, 4/11, 3/11].
    expected_rel = (
        (4.0 / 11.0) * metrics["loss_amb"]
        + (4.0 / 11.0) * metrics["loss_nbr"]
        + (3.0 / 11.0) * metrics["loss_diff"]
    )
    assert metrics["loss_rel"] == pytest.approx(expected_rel, rel=1e-6)
    assert loss.item() == pytest.approx(metrics["loss_rel"], rel=1e-6)
    assert metrics["loss_total"] == pytest.approx(metrics["loss_rel"], rel=1e-6)


def test_heatgeo_tied_temperature_makes_teacher_row_attainable():
    """With tau_1 = graph_temp, a student that reproduces the teacher's cosines
    reaches loss 0 exactly -- the attainability statement behind the tie."""
    graph_temp = FIXED_BANDWIDTH_TEMP
    torch.manual_seed(0)
    anchor = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
    candidates = torch.nn.functional.normalize(torch.randn(1, 3, 4), dim=-1)
    cosines = torch.einsum("bd,bcd->bc", anchor, candidates)
    teacher_probs = torch.softmax(cosines / graph_temp, dim=-1).unsqueeze(1)

    criterion = HeatGeoDistillation(
        row_weight=0.0,
    )
    loss, _ = criterion(
        anchor_embeddings=anchor,
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def _row_artifact() -> dict:
    return {
        "pool_indices": torch.tensor(
            [[1, 2], [0, 2], [1, 3], [2, 1]], dtype=torch.long
        ),
        "pool_probs": torch.tensor(
            [[[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]]],
            dtype=torch.float32,
        ),
        "hard_neg_indices": torch.full((4, 1), -1, dtype=torch.long),
        "transition_neighbors": torch.tensor(
            [[1, 2], [0, 2], [1, 3], [2, 1]], dtype=torch.long
        ),
        "transition_probs": torch.tensor(
            [[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]],
            dtype=torch.float32,
        ),
        "metadata": {"diffusion_scales": (1,)},
    }


def test_sampler_returns_candidates_and_targets_only():
    """The sampler has no row-selection role: no walks, no row_paths."""
    sampler = HeatGeoCandidateSampler(
        artifact=_row_artifact(),
        diffusion_quota=1,
        hard_neg_k=0,
        random_neg_k=2,
        deterministic_topm=1,
        seed=42,
    )

    candidates, teacher_probs = sampler.sample_torch(0)

    assert candidates.shape == (3,)
    assert teacher_probs.shape == (1, 3)
    # Only the diffusion support carries teacher mass; the negatives carry none,
    # which is what lets the criterion recover the row set from the targets alone.
    assert bool((teacher_probs[0, 1:] == 0).all())
    assert not hasattr(sampler, "sample_with_rows")
    assert not hasattr(sampler, "transition_neighbors")


def test_sampler_mixture_row_matches_the_weighted_pool_and_feeds_the_spill():
    """The scale mixture is computed per anchor instead of being precomputed.

    It used to be an (n_items, width) float64 array -- 224 MB at the production
    shape, in every DataLoader worker -- serving only the rare spill branch of
    `_select_support`. This pins both the value and that the branch still works.
    """
    n_items, width = 12, 6
    pool_probs = torch.zeros(2, n_items, width)
    pool_probs[0, :, 0] = 0.4
    pool_probs[0, :, 1] = 0.3
    pool_probs[0, :, 2] = 0.2
    pool_probs[0, :, 3] = 0.1
    # Scale 1 only covers columns the sharper scale takes first, so its quota
    # cannot be filled and the spill has to make up the deficit.
    pool_probs[1, :, 0] = 0.5
    pool_probs[1, :, 1] = 0.5
    artifact = {
        "pool_indices": torch.stack([torch.arange(width) for _ in range(n_items)]),
        "pool_probs": pool_probs,
        "hard_neg_indices": torch.full((n_items, 1), -1, dtype=torch.long),
        "metadata": {"diffusion_scales": (1, 2)},
    }
    sampler = HeatGeoCandidateSampler(
        artifact=artifact, diffusion_quota=5, hard_neg_k=0, random_neg_k=1,
        deterministic_topm=1, seed=0,
    )

    weights = np.array([1.0, 0.5])
    weights = weights / weights.sum()
    expected = (pool_probs[:, 0, :].numpy() * weights.reshape(-1, 1)).sum(axis=0)
    # Bit-exact, not approximate: same operands, same order over the scale axis.
    assert np.array_equal(sampler._mixture_row(0), expected)
    assert not hasattr(sampler, "mixture"), "the precomputed array must be gone"

    _, positions = sampler._select_support(0, sampler._rng(0, 0))
    # Both scales together can supply at most 4 distinct columns; reaching 4 means
    # the spill branch ran and used the mixture.
    assert len(positions) == 4
    assert sorted(positions.tolist()) == [0, 1, 2, 3]


def _closure_graph() -> dict:
    """Five nodes, three teacher neighbours each; node 4 is kept out of the pool.

    Node 2 is the only row whose neighbourhood is partly unexposed (its 0.25 on
    node 4 is unreachable), so it is the row that shows the restricted target
    renormalizing over an incomplete support.
    """
    return {
        "transition_neighbors": torch.tensor(
            [[1, 2, 3], [0, 2, 3], [0, 1, 4], [0, 1, 2], [2, 3, 0]],
            dtype=torch.long,
        ),
        "transition_probs": torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.6, 0.3, 0.1],
                [0.5, 0.25, 0.25],
                [0.4, 0.4, 0.2],
                [0.7, 0.2, 0.1],
            ],
            dtype=torch.float32,
        ),
    }


def _closure_criterion(**kwargs) -> HeatGeoDistillation:
    graph = _closure_graph()
    return HeatGeoDistillation(
        transition_neighbors=graph["transition_neighbors"],
        transition_probs=graph["transition_probs"],
        row_temps=torch.full((5,), 0.2),
        row_weight=1.0,
        **kwargs,
    )


def _expected_closure_rows(pool: torch.Tensor) -> dict[int, torch.Tensor]:
    """KL of each promoted row, computed directly from the graph above."""
    graph = _closure_graph()
    expected = {}
    for node, support in ((1, [0, 2, 3]), (2, [0, 1]), (3, [0, 1, 2])):
        neighbors = graph["transition_neighbors"][node].tolist()
        probs = graph["transition_probs"][node]
        mass = torch.tensor([probs[neighbors.index(v)] for v in support])
        target = mass / mass.sum()
        logits = torch.stack([pool[node] @ pool[v] for v in support]) / 0.2
        expected[node] = (target * (target.log() - logits.log_softmax(dim=0))).sum()
    return expected


def test_row_loss_promotes_teacher_selected_columns_and_skips_anchors():
    torch.manual_seed(0)
    pool = F.normalize(torch.randn(4, 3), dim=-1).requires_grad_()
    # Node 0 is the batch anchor: L_rel already matches its transition row as the
    # r=1 target, so promoting it again would duplicate that term.
    loss, metrics = _closure_criterion()._compute_row_loss(
        pool_norm=pool,
        column_idx=torch.arange(4),
        selected_columns=torch.ones(4, dtype=torch.bool),
        anchor_columns=torch.tensor([True, False, False, False]),
    )

    rows = _expected_closure_rows(pool)
    # nu is uniform over the three promoted rows.
    assert loss.item() == pytest.approx((sum(rows.values()) / 3).item(), rel=1e-5)
    assert metrics["row_count"].item() == 3.0
    assert metrics["row_valid_ratio"].item() == pytest.approx(1.0)
    # Row 2 loses the 0.25 sitting on the out-of-pool node 4; the other two are
    # fully exposed. This is the truncation the restricted target accepts.
    assert metrics["row_exposed_mass"].item() == pytest.approx(
        (1.0 + 0.75 + 1.0) / 3, rel=1e-5
    )
    loss.backward()
    assert pool.grad is not None and torch.isfinite(pool.grad).all()


def test_row_loss_ignores_columns_without_teacher_mass():
    """Hard and uniform negatives are columns, not teacher-selected relations."""
    torch.manual_seed(0)
    pool = F.normalize(torch.randn(4, 3), dim=-1)
    # Node 3 is in the pool as a negative: it carries no diffusion mass for anyone.
    loss, metrics = _closure_criterion()._compute_row_loss(
        pool_norm=pool,
        column_idx=torch.arange(4),
        selected_columns=torch.tensor([True, True, True, False]),
        anchor_columns=torch.tensor([True, False, False, False]),
    )

    rows = _expected_closure_rows(pool)
    assert metrics["row_count"].item() == 2.0
    assert loss.item() == pytest.approx(((rows[1] + rows[2]) / 2).item(), rel=1e-5)


def test_row_loss_forward_takes_no_walk_input_and_backpropagates():
    torch.manual_seed(0)
    criterion = _closure_criterion(diffusion_scales=(1,))
    criterion.use_row_loss = True

    anchors = torch.randn(2, 3, requires_grad=True)
    candidates = torch.randn(2, 3, 3, requires_grad=True)
    loss, metrics = criterion(
        anchor_embeddings=anchors,
        candidate_embeddings=candidates,
        teacher_probs=torch.tensor(
            [[[0.5, 0.3, 0.2]], [[0.7, 0.2, 0.1]]], dtype=torch.float32
        ),
        candidate_idx=torch.tensor([[1, 2, 3], [2, 3, 1]]),
        anchor_idx=torch.tensor([0, 4]),
    )

    assert metrics["loss_row"] > 0.0
    assert metrics["row_count"] > 0.0
    loss.backward()
    assert torch.isfinite(anchors.grad).all()
    assert torch.isfinite(candidates.grad).all()


def test_row_loss_without_a_graph_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError, match="transition arrays"):
        HeatGeoDistillation(row_weight=1.0)
    # row_weight=0 is the honest way to run without L_row.
    HeatGeoDistillation(row_weight=0.0)


def test_row_selection_knobs_are_gone():
    """The removed arms must raise rather than be silently absorbed."""
    from config import HeatGeoConfig

    config = HeatGeoConfig()
    for removed in ("row_mode", "row_ambient", "num_walks", "walk_length"):
        assert not hasattr(config, removed)
        with pytest.raises(AttributeError):
            HeatGeoConfig(**{removed: 1})
        with pytest.raises(ValueError, match=removed):
            HeatGeoDistillation(**{removed: 1})

    assert config.row_weight == 1.0
    assert config.row_start_epoch == 1


def test_entropic_affinity_hits_the_requested_perplexity():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=64) * 0.1 + 0.7
    for perplexity in (2.0, 5.0, 30.0):
        probs, tau, clamped = _entropic_affinity(scores, float(np.log(perplexity)))
        entropy = -(probs * np.log(probs)).sum()
        assert not clamped
        assert tau > 0.0
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.exp(entropy), perplexity, rtol=1e-5)


def test_row_entropy_is_strictly_increasing_in_temperature():
    # The uniqueness of the solve rests on dH/dbeta = -beta * Var_p(s) < 0, i.e.
    # entropy strictly increasing in tau. If that fails the bisection is solving
    # for a root that need not be unique.
    rng = np.random.default_rng(1)
    scores = rng.normal(size=32) * 0.1 + 0.7
    taus = np.geomspace(1e-3, 1e1, 40)
    entropies = [_softmax_at(scores, 1.0 / tau)[1] for tau in taus]
    assert all(b > a for a, b in zip(entropies, entropies[1:]))
    # The range it sweeps is (0, log d), which is what makes any perplexity below
    # the degree reachable and anything at or above it not.
    assert entropies[0] < 0.05
    assert entropies[-1] > np.log(scores.size) - 0.05


def test_entropic_affinity_is_invariant_to_affine_rescaling():
    # The claim that removes per-teacher retuning: a teacher whose cosines are
    # spread differently produces the *same* graph, with the temperature absorbing
    # the rescaling. A fixed temperature does not have this property.
    rng = np.random.default_rng(2)
    scores = rng.normal(size=48) * 0.1 + 0.7
    target = float(np.log(30.0))

    base_probs, base_tau, _ = _entropic_affinity(scores, target)
    for a, b in ((3.0, 0.0), (0.25, 0.0), (2.0, -1.5), (0.5, 4.0)):
        probs, tau, _ = _entropic_affinity(a * scores + b, target)
        np.testing.assert_allclose(probs, base_probs, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(tau, a * base_tau, rtol=1e-4)

    # The fixed-temperature row, by contrast, sharpens when the scale is stretched.
    fixed = np.exp((scores - scores.max()) / 0.05)
    fixed /= fixed.sum()
    stretched = np.exp((3.0 * scores - (3.0 * scores).max()) / 0.05)
    stretched /= stretched.sum()
    assert not np.allclose(fixed, stretched, atol=1e-6)


def test_entropic_affinity_clamps_when_perplexity_exceeds_degree():
    # log d is the supremum, reached only as tau -> infinity, so a row with fewer
    # neighbours than the requested perplexity is solved under its own ceiling and
    # says so rather than running the bracket off to infinity.
    scores = np.array([0.9, 0.8, 0.4])
    probs, tau, clamped = _entropic_affinity(scores, float(np.log(30.0)))
    assert clamped
    assert np.isfinite(tau) and tau > 0.0
    entropy = -(probs * np.log(probs)).sum()
    assert entropy <= np.log(scores.size) + 1e-12


def test_mass_prefix_meets_the_stated_tv_and_kl_bounds():
    # The identity the tolerance rests on: truncating to a set carrying 1 - delta
    # and renormalizing perturbs the row by exactly delta in total variation and by
    # -log(1 - delta) nats in KL. If this fails, the tolerance is not a guarantee.
    rng = np.random.default_rng(3)
    for tolerance in (0.001, 0.01, 0.05):
        probs = rng.dirichlet(np.full(500, 0.3))
        keep = _mass_prefix(probs, float(probs.sum()), tolerance)
        kept = probs[keep]
        delta = 1.0 - kept.sum()

        assert 0.0 <= delta <= tolerance
        truncated = kept / kept.sum()
        total_variation = 0.5 * (np.abs(truncated - kept / probs.sum()).sum() + delta)
        np.testing.assert_allclose(total_variation, delta, atol=1e-12)
        kl = float((truncated * np.log(truncated / kept)).sum())
        np.testing.assert_allclose(kl, -np.log1p(-delta), atol=1e-12)

        # Smallest such set: dropping one more entry would break the tolerance.
        assert delta + kept.min() > tolerance


def test_criterion_uses_the_row_temperature_of_each_anchor():
    # Two anchors with different row temperatures must produce different student
    # distributions from identical similarities; a scalar temperature cannot.
    torch.manual_seed(0)
    # Anchor 0 is matched at tau = 0.02, anchor 1 at tau = 0.50.
    criterion = HeatGeoDistillation(
        row_temps=torch.tensor([0.02, 0.50, 0.10, 0.10, 0.10, 0.10]),
        row_weight=0.0,
    )
    # Identical anchor and identical candidates for both rows, and no anchor index
    # among the candidates, so the only thing that differs is the temperature.
    anchor = F.normalize(torch.randn(1, 4), dim=-1).repeat(2, 1)
    candidates = F.normalize(torch.randn(1, 4, 4), dim=-1).repeat(2, 1, 1)
    teacher_probs = torch.tensor(
        [[[0.5, 0.25, 0.15, 0.10]], [[0.5, 0.25, 0.15, 0.10]]], dtype=torch.float32
    )
    candidate_idx = torch.tensor([[2, 3, 4, 5], [2, 3, 4, 5]], dtype=torch.long)

    entropies = []
    for idx in (0, 1):
        _, metrics = criterion(
            anchor_embeddings=anchor[:1],
            candidate_embeddings=candidates[:1],
            teacher_probs=teacher_probs[:1],
            candidate_idx=candidate_idx[:1],
            anchor_idx=torch.tensor([idx], dtype=torch.long),
        )
        entropies.append(metrics["student_entropy"])
        # A per-row temperature is not the tied-temperature case the excess bound
        # assumes, and the flag has to say so.
        assert metrics["excess_is_exact"] == 0.0

    # tau = 0.02 is 25x sharper than tau = 0.50, so its softmax is lower entropy.
    assert entropies[0] < entropies[1]


@pytest.mark.parametrize(
    "removed_flag",
    (
        "--graph_temp",
        "--direct_weight",
        "--candidate_size",
        "--walk_non_backtracking",
        "--walk_weight",
        "--walk_start_epoch",
        "--mass_weight",
        "--geo_weight",
        "--sym_weight",
        # Removed with walk-based row selection and its weighting variants.
        "--num_walks",
        "--walk_length",
        "--row_mode",
        "--row_ambient",
        # Removed with the W&B surface, which was disabled at import.
        "--no_wandb",
        "--wandb_project",
        "--wandb_run_name",
        "--wandb_mode",
        # Removed: never read.
        "--eval_data",
    ),
)
def test_removed_heatgeo_cli_knobs_are_rejected(monkeypatch, removed_flag):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--method", "heatgeo", removed_flag, "1"]
    )
    with pytest.raises(SystemExit):
        main.parse_args()


class _TinyHeatGeoStudent(torch.nn.Module):
    """Minimal stand-in for the student encoder used by the heatgeo train_step."""

    def __init__(self, vocab: int = 64, dim: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, dim)

    def forward(
        self, input_ids, attention_mask, return_dict=True, output_hidden_states=False
    ):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_heatgeo_train_step_updates_the_student(monkeypatch):
    """End-to-end cover for the active path: the candidate_chunks/candidate_inverse
    gather, the criterion call, and the optimizer/scheduler step."""
    from torch.amp import GradScaler

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)

    corpus, dim, batch_size, n_candidates = 40, 8, 4, 5
    graph_k = 4
    neighbors = torch.stack([torch.randperm(corpus)[:graph_k] for _ in range(corpus)])
    probs = F.normalize(torch.rand(corpus, graph_k), p=1, dim=1)

    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = SimpleNamespace(distill_method="heatgeo", w_task=0.0)
    distiller.device_s = torch.device("cpu")
    distiller.model_student = _TinyHeatGeoStudent(vocab=corpus, dim=dim)
    distiller.criterion = HeatGeoDistillation(
        diffusion_scales=(1,),
        teacher_embeddings=F.normalize(torch.randn(corpus, dim), dim=-1),
        transition_neighbors=neighbors,
        transition_probs=probs,
        row_temps=torch.full((corpus,), 0.15),
        row_weight=1.0,
    )
    distiller.criterion.use_row_loss = True
    distiller.optimizer = torch.optim.Adam(
        distiller.model_student.parameters(), lr=1e-2
    )
    distiller.scheduler = torch.optim.lr_scheduler.LambdaLR(
        distiller.optimizer, lambda _: 1.0
    )
    distiller.scaler = GradScaler("cuda", enabled=False)
    distiller.current_epoch = 0
    distiller.current_step = 0

    anchor_idx = torch.arange(batch_size)
    candidate_idx = torch.stack(
        [
            torch.randperm(corpus - batch_size)[:n_candidates] + batch_size
            for _ in range(batch_size)
        ]
    )
    teacher_probs = torch.rand(batch_size, 1, n_candidates)
    # Only the first half of each draw is diffusion support; the rest are negatives,
    # which is what lets the criterion recover the promoted row set.
    teacher_probs[:, :, n_candidates // 2 :] = 0.0

    # The collate hands the student one chunk per length bucket plus the inverse
    # index that scatters encoded rows back to [batch * candidates].
    unique_idx, inverse = torch.unique(candidate_idx.reshape(-1), return_inverse=True)
    batch = {
        "idx": anchor_idx,
        "input_ids1_stu": anchor_idx.view(-1, 1),
        "attention_mask1_stu": torch.ones(batch_size, 1, dtype=torch.long),
        "candidate_idx": candidate_idx,
        "candidate_inverse": inverse,
        "teacher_probs": teacher_probs,
        "candidate_chunks": [
            {
                "input_ids": unique_idx.view(-1, 1),
                "attention_mask": torch.ones(unique_idx.numel(), 1, dtype=torch.long),
            }
        ],
    }
    before = distiller.model_student.embedding.weight.detach().clone()

    loss, metrics = distiller.train_step(batch)

    assert torch.isfinite(loss)
    assert metrics["loss_rel"] > 0
    # L_row must actually be active, not silently zero.
    assert metrics["row_count"] > 0
    assert metrics["loss_row"] > 0
    assert not torch.equal(before, distiller.model_student.embedding.weight.detach())


def test_final_student_weights_are_idempotent(tmp_path):
    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = SimpleNamespace(
        weights_dir=str(tmp_path / "weights"),
        save_dir=str(tmp_path / "run"),
        student_model_name="student",
        teacher_model_name="teacher",
    )
    distiller.model_student = torch.nn.Linear(2, 2)
    distiller._saved_student_weight_epochs = set()

    save_student_weights(distiller, 4)
    save_student_weights(distiller, 4)

    files = list((tmp_path / "weights").glob("*.pt"))
    assert [path.name for path in files] == ["student_epoch_5.pt"]
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    assert payload["epoch"] == 5
