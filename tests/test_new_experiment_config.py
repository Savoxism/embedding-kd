import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import main
from distiller import KnowledgeDistiller, add_domain_averages
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
            "--mass_weight",
            "0.8",
            "--geo_weight",
            "0.3",
            "--sym_weight",
            "0.4",
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

    assert config.mass_weight == 0.8
    assert config.geo_weight == 0.3
    assert config.sym_weight == 0.4
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
        student_dim=4,
        teacher_dim=4,
        diffusion_scales=(1, 2, 4),
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
        "transition_neighbors",
        "transition_probs",
        "direct_student_temp",
    ):
        with pytest.raises(ValueError, match=removed):
            HeatGeoDistillation(
                student_dim=4,
                teacher_dim=4,
                **{removed: 0.07},
            )


def test_canonical_policy_preserves_the_previous_resolved_values():
    assert candidate_budget(14, 26, 26) == 66
    assert hard_negative_pool_size(200) == 200
    np.testing.assert_allclose(
        normalized_diffusion_weights((1, 2, 4)),
        np.asarray([1.0, 0.5, 0.25]) / 1.75,
    )


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
        student_dim=4,
        teacher_dim=4,
    )
    loss, _ = criterion(
        anchor_embeddings=anchor,
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_sampler_preserves_support_mass_and_returns_exact_stratum_weights():
    n_items = 8
    artifact = {
        "pool_indices": torch.tensor(
            [
                [(i + 1) % n_items, (i + 2) % n_items, (i + 3) % n_items]
                for i in range(n_items)
            ],
            dtype=torch.long,
        ),
        "pool_probs": torch.tensor(
            [[[0.6, 0.2, 0.2]] * n_items], dtype=torch.float32
        ),
        "hard_neg_indices": torch.tensor([[4, 5]] * n_items, dtype=torch.long),
        "metadata": {"diffusion_scales": (1,)},
    }
    sampler = HeatGeoCandidateSampler(
        artifact=artifact,
        diffusion_quota=2,
        hard_neg_k=1,
        random_neg_k=1,
        deterministic_topm=2,
        seed=42,
    )

    candidates, teacher_probs, importance = sampler.sample(0)

    assert candidates.size == 4
    assert teacher_probs.sum() == pytest.approx(0.8)
    assert np.count_nonzero(teacher_probs) == 2
    # H=2 hard nodes sampled once; U=8-{anchor, two support, two hard}=3.
    np.testing.assert_allclose(np.sort(importance[importance > 0]), [2.0, 3.0])


def test_support_mass_loss_matches_bernoulli_kl():
    anchor = F.normalize(torch.tensor([[1.0, 0.2, -0.1]]), dim=-1)
    candidates = F.normalize(
        torch.tensor(
            [
                [
                    [0.9, 0.1, 0.0],
                    [0.7, 0.4, -0.2],
                    [0.1, 0.8, 0.3],
                    [-0.2, 0.4, 0.9],
                ]
            ]
        ),
        dim=-1,
    )
    teacher_probs = torch.tensor([[[0.6, 0.2, 0.0, 0.0]]])
    importance = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
    criterion = HeatGeoDistillation(
        student_dim=3, teacher_dim=3, mass_weight=0.7
    )

    total, metrics = criterion(
        anchor_embeddings=anchor,
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
        ambient_importance=importance,
    )

    logits = torch.einsum("bd,bcd->bc", anchor, candidates) / FIXED_BANDWIDTH_TEMP
    log_z_support = torch.logsumexp(logits[:, :2], dim=-1)
    log_z_complement = torch.logsumexp(
        torch.stack([logits[:, 2], logits[:, 3] + torch.tensor(2.0).log()], dim=-1),
        dim=-1,
    )
    mass_logit = log_z_support - log_z_complement
    alpha_s = mass_logit.sigmoid()
    alpha_t = torch.tensor(0.8)
    entropy_t = -(alpha_t * alpha_t.log() + (1 - alpha_t) * (1 - alpha_t).log())
    expected = (
        F.binary_cross_entropy_with_logits(
            mass_logit, alpha_t.expand_as(mass_logit)
        )
        - entropy_t
    )

    assert metrics["loss_mass"] == pytest.approx(expected.item(), rel=1e-5)
    assert metrics["teacher_support_mass"] == pytest.approx(0.8)
    assert metrics["student_support_mass"] == pytest.approx(alpha_s.item(), rel=1e-5)
    assert total.item() == pytest.approx(
        metrics["loss_diff"] + 0.7 * expected.item(), rel=1e-5
    )


def _geometry_loss_inputs():
    teacher_bank = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        dim=-1,
    )
    anchor = F.normalize(
        torch.tensor([[1.0, 0.2, 0.0], [0.7, 0.7, 0.0]]), dim=-1
    )
    candidates = F.normalize(
        torch.tensor(
            [
                [[0.9, 0.4, 0.0], [0.1, 0.9, 0.0]],
                [[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]],
            ]
        ),
        dim=-1,
    )
    teacher_probs = torch.tensor([[[0.75, 0.25]], [[0.20, 0.80]]])
    candidate_idx = torch.tensor([[1, 2], [0, 2]], dtype=torch.long)
    anchor_idx = torch.tensor([0, 1], dtype=torch.long)
    return teacher_bank, anchor, candidates, teacher_probs, candidate_idx, anchor_idx


def test_selected_geometry_loss_matches_mass_weighted_cosine_distortion():
    teacher_bank, anchor, candidates, teacher_probs, candidate_idx, anchor_idx = (
        _geometry_loss_inputs()
    )
    criterion = HeatGeoDistillation(
        student_dim=3,
        teacher_dim=3,
        teacher_embeddings=teacher_bank,
        mass_weight=0.0,
        geo_weight=1.0,
        sym_weight=0.0,
    )

    total, metrics = criterion(
        anchor_embeddings=anchor,
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
        candidate_idx=candidate_idx,
        anchor_idx=anchor_idx,
    )

    student_similarity = torch.einsum("bd,bcd->bc", anchor, candidates)
    stored_teacher_bank = criterion.teacher_bank.float()
    teacher_similarity = torch.einsum(
        "bd,bcd->bc",
        stored_teacher_bank.index_select(0, anchor_idx),
        stored_teacher_bank.index_select(0, candidate_idx.reshape(-1)).view(2, 2, -1),
    )
    expected = (
        teacher_probs.squeeze(1)
        * (student_similarity - teacher_similarity).square()
        / 4.0
    ).sum(dim=-1).mean()

    assert metrics["loss_geo"] == pytest.approx(expected.item(), rel=1e-5)
    assert metrics["geo_relations"] == 4.0
    assert total.item() == pytest.approx(
        metrics["loss_diff"] + expected.item(), rel=1e-5
    )


def test_symmetric_geometry_loss_scores_each_unordered_edge_once():
    teacher_bank, anchor, candidates, teacher_probs, candidate_idx, anchor_idx = (
        _geometry_loss_inputs()
    )
    anchor.requires_grad_()
    candidates.requires_grad_()
    criterion = HeatGeoDistillation(
        student_dim=3,
        teacher_dim=3,
        teacher_embeddings=teacher_bank,
        mass_weight=0.0,
        geo_weight=0.0,
        sym_weight=1.0,
    )

    total, metrics = criterion(
        anchor_embeddings=anchor,
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
        candidate_idx=candidate_idx,
        anchor_idx=anchor_idx,
    )

    directed_student = torch.einsum("bd,bcd->bc", anchor, candidates)
    student_edges = torch.stack(
        [
            0.5 * (directed_student[0, 0] + directed_student[1, 0]),
            directed_student[0, 1],
            directed_student[1, 1],
        ]
    )
    teacher_edges = torch.stack(
        [
            criterion.teacher_bank[0].float() @ criterion.teacher_bank[1].float(),
            criterion.teacher_bank[0].float() @ criterion.teacher_bank[2].float(),
            criterion.teacher_bank[1].float() @ criterion.teacher_bank[2].float(),
        ]
    )
    edge_weights = torch.tensor(
        [0.5 * (0.75 + 0.20), 0.5 * 0.25, 0.5 * 0.80]
    )
    expected = (
        edge_weights * (student_edges - teacher_edges).square()
    ).sum() / 3

    assert metrics["loss_sym"] == pytest.approx(expected.item(), rel=1e-5)
    assert metrics["sym_edges"] == 3.0
    assert total.item() == pytest.approx(
        metrics["loss_diff"] + expected.item(), rel=1e-5
    )
    total.backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    assert candidates.grad is not None and torch.isfinite(candidates.grad).all()


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
        student_dim=4,
        teacher_dim=4,
        row_temps=torch.tensor([0.02, 0.50, 0.10, 0.10, 0.10, 0.10]),
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
        "--num_walks",
        "--walk_length",
        "--walk_start_epoch",
    ),
)
def test_removed_heatgeo_cli_knobs_are_rejected(monkeypatch, removed_flag):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--method", "heatgeo", removed_flag, "1"]
    )
    with pytest.raises(SystemExit):
        main.parse_args()


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

    distiller.save_student_weights(4)
    distiller.save_student_weights(4)

    files = list((tmp_path / "weights").glob("*.pt"))
    assert [path.name for path in files] == ["student_epoch_5.pt"]
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    assert payload["epoch"] == 5
