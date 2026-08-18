import sys
from types import SimpleNamespace

import numpy as np
import torch

import main
from distiller import KnowledgeDistiller
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
            "--walk_temp",
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
    assert config.walk_temp == 0.04
    assert config.walk_topk == 128
    assert config.candidate_size == 96
    assert config.cache_path == "cache/teacher.pt"
    assert config.heatgeo_cache_path == "cache/graph.pt"
    assert config.pooling_method == "last_token"
    assert config.weights_dir == "weights"
    assert config.final_weights_only is True


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
