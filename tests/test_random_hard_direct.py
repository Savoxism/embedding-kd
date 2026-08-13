import numpy as np
import torch

from src.data_utils.dataset_cache import TextPairWithTeacherAndHeatGeo
from src.heatgeo.candidate_sampler import RandomHardDirectCandidateSampler
from src.heatgeo import graph_builder
from src.heatgeo.hard_negative_builder import (
    build_or_load_hard_negative_artifact,
)


def _sampler() -> RandomHardDirectCandidateSampler:
    n_items = 100
    hard = torch.full((n_items, 30), -1, dtype=torch.long)
    for idx in range(n_items):
        pool = [node for node in range(n_items) if node != idx][:30]
        hard[idx] = torch.tensor(pool)
    return RandomHardDirectCandidateSampler(
        artifact={"hard_neg_indices": hard},
        candidate_size=64,
        random_candidate_k=32,
        hard_neg_k=24,
        random_neg_k=8,
        seed=17,
    )


def test_random_hard_sampler_has_exact_layout_and_no_duplicates():
    sampler = _sampler()
    candidates = sampler.sample(0)

    assert candidates.shape == (64,)
    assert np.unique(candidates).size == 64
    assert 0 not in candidates
    hard_pool = set(sampler.hard_neg_indices[0].tolist())
    assert set(candidates[32:56]).issubset(hard_pool)


def test_random_hard_sampler_is_epoch_deterministic_and_redraws():
    sampler = _sampler()
    first = sampler.sample(7)
    repeated = sampler.sample(7)
    sampler.set_epoch(1)
    next_epoch = sampler.sample(7)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, next_epoch)


def test_direct_only_dataset_omits_diffusion_targets():
    sampler = _sampler()
    dataset = TextPairWithTeacherAndHeatGeo(
        anchor_texts=[f"row {idx}" for idx in range(100)],
        teacher_cls=torch.zeros(100, 2),
        sampler=sampler,
    )

    item = dataset[0]

    assert item["candidate_idx"].shape == (64,)
    assert "teacher_probs" not in item


def test_hard_builder_prefers_same_source_then_falls_back(tmp_path):
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.20, 0.98],
            [0.90, 0.10],
        ]
    )
    source_ids = np.asarray([0, 1, 0, 1], dtype=np.int64)
    cache_path = tmp_path / "hard_negative_pool.pt"

    artifact = build_or_load_hard_negative_artifact(
        teacher_embeddings=embeddings,
        cache_path=str(cache_path),
        hard_neg_pool=2,
        source_ids=source_ids,
        chunk_size=2,
    )

    # Node 2 is the only same-source choice for anchor 0, so it must precede the
    # much closer cross-source nodes. The second slot is the closest fallback.
    assert artifact["hard_neg_indices"][0].tolist() == [2, 1]
    assert set(artifact) == {
        "hard_neg_indices",
        "source_ids",
        "hard_negative_stats",
        "metadata",
    }
    assert artifact["metadata"]["candidate_sampling_mode"] == "random_hard_direct"
    assert "pool_indices" not in artifact
    assert "pool_probs" not in artifact


def test_hard_artifact_rebuilds_when_teacher_changes(tmp_path):
    cache_path = tmp_path / "hard_negative_pool.pt"
    embeddings = torch.eye(4)
    source_ids = np.zeros(4, dtype=np.int64)

    first = build_or_load_hard_negative_artifact(
        embeddings, str(cache_path), 2, source_ids=source_ids, chunk_size=2
    )
    changed = embeddings.clone()
    changed[0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    second = build_or_load_hard_negative_artifact(
        changed, str(cache_path), 2, source_ids=source_ids, chunk_size=2
    )

    assert (
        first["metadata"]["teacher_fingerprint"]
        != second["metadata"]["teacher_fingerprint"]
    )


def test_hard_artifact_builder_never_calls_diffusion(tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("diffusion helper was called")

    monkeypatch.setattr(graph_builder, "_diffuse_block", fail_if_called)
    artifact = build_or_load_hard_negative_artifact(
        torch.eye(4),
        str(tmp_path / "hard_negative_pool.pt"),
        2,
        source_ids=np.zeros(4, dtype=np.int64),
        chunk_size=2,
    )

    assert artifact["hard_neg_indices"].shape == (4, 2)
