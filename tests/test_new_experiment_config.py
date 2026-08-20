import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import main
from distiller import KnowledgeDistiller
from src.criterions.heatgeo_distillation import HeatGeoDistillation
from src.heatgeo.candidate_sampler import HeatGeoCandidateSampler


def test_heatgeo_cli_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--method",
            "heatgeo",
            "--walk_weight",
            "0.8",
            "--graph_temp",
            "0.04",
            "--walk_topk",
            "128",
            "--candidate_size",
            "96",
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

    assert config.walk_weight == 0.8
    assert config.graph_temp == 0.04
    assert not hasattr(config, "walk_temp")
    assert config.walk_topk == 128
    assert config.candidate_size == 96
    assert config.cache_path == "cache/teacher.pt"
    assert config.heatgeo_cache_path == "cache/graph.pt"
    assert config.pooling_method == "last_token"
    assert config.weights_dir == "weights"
    assert config.final_weights_only is True


def test_heatgeo_temperature_ties():
    criterion = HeatGeoDistillation(
        student_dim=4,
        teacher_dim=4,
        scale_weights=(1.0, 0.5, 0.25),
        broad_scale_temps=(0.07, 0.10),
        graph_temp=0.05,
    )
    # tau_1 = tau_w = graph_temp: not knobs, derived.
    assert criterion.walk_temp == pytest.approx(0.05)
    assert criterion.scale_temps.tolist() == pytest.approx([0.05, 0.07, 0.10])

    for removed in ("scale_temps", "walk_temp", "direct_student_temp"):
        with pytest.raises(ValueError, match=removed):
            HeatGeoDistillation(
                student_dim=4,
                teacher_dim=4,
                scale_weights=(1.0,),
                graph_temp=0.05,
                **{removed: 0.07},
            )


def test_heatgeo_tied_temperature_makes_teacher_row_attainable():
    """With tau_1 = graph_temp, a student that reproduces the teacher's cosines
    reaches loss 0 exactly -- the attainability statement behind the tie."""
    graph_temp = 0.05
    torch.manual_seed(0)
    anchor = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
    candidates = torch.nn.functional.normalize(torch.randn(1, 3, 4), dim=-1)
    cosines = torch.einsum("bd,bcd->bc", anchor, candidates)
    teacher_probs = torch.softmax(cosines / graph_temp, dim=-1).unsqueeze(1)

    criterion = HeatGeoDistillation(
        student_dim=4,
        teacher_dim=4,
        scale_weights=(1.0,),
        graph_temp=graph_temp,
    )
    loss, _ = criterion(
        anchor_embeddings=anchor,
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_walk_topk_truncates_and_normalizes_transition_rows():
    artifact = {
        "pool_indices": torch.tensor(
            [[[1, 2, 3]], [[0, 2, 3]]], dtype=torch.long
        ).squeeze(1),
        "pool_probs": torch.tensor(
            [[[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]]], dtype=torch.float32
        ),
        "hard_neg_indices": torch.tensor([[3], [3]], dtype=torch.long),
        "transition_neighbors": torch.tensor(
            [[1, 2, 3, -1], [0, 2, 3, -1]], dtype=torch.long
        ),
        "transition_probs": torch.tensor(
            [[0.1, 0.6, 0.3, 0.0], [0.5, 0.2, 0.3, 0.0]],
            dtype=torch.float32,
        ),
    }

    sampler = HeatGeoCandidateSampler(
        artifact=artifact,
        candidate_size=2,
        diffusion_quota=1,
        hard_neg_k=0,
        random_neg_k=1,
        scale_weights=(1.0,),
        seed=42,
        num_walks=1,
        walk_length=1,
        walk_topk=2,
    )

    assert sampler.trans_neighbors.shape == (2, 2)
    assert sampler.trans_neighbors[0].tolist() == [2, 3]
    np.testing.assert_allclose(sampler.trans_probs.sum(axis=1), 1.0)


def _walk_sampler(
    neighbors: list[list[int]],
    probs: list[list[float]],
    *,
    walk_length: int,
    non_backtracking: bool,
    num_walks: int = 8,
    seed: int = 0,
) -> HeatGeoCandidateSampler:
    """Sampler over a hand-built transition matrix, with the diffusion side inert.

    Only `_sample_walks` is under test here, so the pools are filled with whatever
    keeps the quota check happy; the walk path never reads them.
    """
    n_items = len(neighbors)
    pool = torch.tensor(
        [[(i + 1) % n_items] for i in range(n_items)], dtype=torch.long
    )
    artifact = {
        "pool_indices": pool,
        "pool_probs": torch.ones((1, n_items, 1), dtype=torch.float32),
        "hard_neg_indices": torch.full((n_items, 1), -1, dtype=torch.long),
        "transition_neighbors": torch.tensor(neighbors, dtype=torch.long),
        "transition_probs": torch.tensor(probs, dtype=torch.float32),
    }
    return HeatGeoCandidateSampler(
        artifact=artifact,
        candidate_size=2,
        diffusion_quota=1,
        hard_neg_k=0,
        random_neg_k=1,
        scale_weights=(1.0,),
        seed=seed,
        num_walks=num_walks,
        walk_length=walk_length,
        walk_non_backtracking=non_backtracking,
    )


# 4-cycle: every node has exactly two neighbours, so a non-backtracking walk has
# exactly one legal continuation at each step and must keep going round.
_CYCLE_NEIGHBORS = [[1, 3], [0, 2], [1, 3], [0, 2]]
_CYCLE_PROBS = [[0.5, 0.5]] * 4


def test_non_backtracking_walk_never_returns_to_the_previous_node():
    sampler = _walk_sampler(
        _CYCLE_NEIGHBORS, _CYCLE_PROBS, walk_length=3, non_backtracking=True
    )
    walks, _ = sampler._sample_walks(0, sampler._rng(0))

    assert not (walks[:, 2:] == walks[:, :-2]).any()
    # One legal continuation per step on this graph, so the walk sweeps the cycle:
    # 4 steps, 4 distinct nodes, every row of the teacher operator supervised once.
    for path in walks:
        assert len(set(path.tolist())) == path.size


def test_plain_walk_backtracks_and_covers_less():
    plain = _walk_sampler(
        _CYCLE_NEIGHBORS, _CYCLE_PROBS, walk_length=3, non_backtracking=False
    )
    plain_walks, _ = plain._sample_walks(0, plain._rng(0))
    non_backtracking = _walk_sampler(
        _CYCLE_NEIGHBORS, _CYCLE_PROBS, walk_length=3, non_backtracking=True
    )
    nb_walks, _ = non_backtracking._sample_walks(0, non_backtracking._rng(0))

    def distinct_per_walk(walks: np.ndarray) -> float:
        # Per walk, not unioned over walks: the union saturates on a 4-node graph
        # while the quantity the loss actually spends -- rows supervised per step of
        # walk budget -- is what differs.
        return float(
            np.mean([len(set(path[1:].tolist())) for path in walks])
        )

    # A plain walk on the 4-cycle returns to its previous node half the time, and
    # each such visit re-supervises a row it has already seen.
    assert (plain_walks[:, 2:] == plain_walks[:, :-2]).any()
    assert distinct_per_walk(plain_walks) < distinct_per_walk(nb_walks)


def test_non_backtracking_backtracks_out_of_a_degree_one_node():
    # Path graph 0 -- 1 -- 2: node 2 has a single edge, so the walk that arrives
    # there has to retreat rather than stall on a self-loop.
    neighbors = [[1, -1], [0, 2], [1, -1]]
    probs = [[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]]
    sampler = _walk_sampler(
        neighbors, probs, walk_length=3, non_backtracking=True, num_walks=4
    )
    walks, _ = sampler._sample_walks(0, sampler._rng(0))

    # 0 -> 1 -> 2 -> 1 is the only path of this length, and step 3 is a backtrack.
    for path in walks:
        assert path.tolist() == [0, 1, 2, 1]


def test_walk_paths_are_reproducible_under_the_same_seed():
    first = _walk_sampler(
        _CYCLE_NEIGHBORS, _CYCLE_PROBS, walk_length=4, non_backtracking=True
    )
    second = _walk_sampler(
        _CYCLE_NEIGHBORS, _CYCLE_PROBS, walk_length=4, non_backtracking=True
    )
    walks_first, _ = first._sample_walks(2, first._rng(2))
    walks_second, _ = second._sample_walks(2, second._rng(2))

    np.testing.assert_array_equal(walks_first, walks_second)


def test_walk_non_backtracking_cli_override():
    monkeypatched = [
        "main.py",
        "--method",
        "heatgeo",
        "--walk_non_backtracking",
        "0",
    ]
    original = sys.argv
    try:
        sys.argv = monkeypatched
        args = main.parse_args()
        config = main.get_config(args.method, args)
    finally:
        sys.argv = original

    assert config.walk_non_backtracking is False


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
