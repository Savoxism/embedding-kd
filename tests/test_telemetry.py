"""Tests for the run manifest, the per-epoch record, and the geometry probe."""

import json
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

from src.distill.geometry import build_probe_set, probe_geometry
from src.distill.telemetry import (
    append_epoch_record,
    load_epochs,
    new_run_id,
    write_run_manifest,
)


class _TinyEncoder(nn.Module):
    def __init__(self, vocab=64, dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyTokenizer:
    def __call__(
        self, texts, padding=True, truncation=True, max_length=8, return_tensors="pt"
    ):
        ids = [[(ord(c) % 60) + 1 for c in text[:max_length]] or [1] for text in texts]
        width = max(len(row) for row in ids)
        padded = [row + [0] * (width - len(row)) for row in ids]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in ids]
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(mask),
        }


def test_run_manifest_pins_config_git_and_artifact(tmp_path):
    run_id = new_run_id()
    config = SimpleNamespace(
        distill_method="heatgeo",
        row_weight=1.0,
        diffusion_scales=(1, 2, 4),
        seed=42,
        save_dir=str(tmp_path),
        _private="hidden",
        some_method=lambda: None,
    )
    artifact = {
        "metadata": {"teacher_fingerprint": "abc123", "graph_k": 200},
        "graph_stats": {"target_js_floor": 0.11, "unserializable": object()},
    }

    write_run_manifest(str(tmp_path), run_id, config, artifact=artifact)
    payload = json.loads((tmp_path / "run.json").read_text())

    assert payload["run_id"] == run_id
    assert payload["config"]["row_weight"] == 1.0
    assert payload["config"]["diffusion_scales"] == [1, 2, 4]
    # Private attributes and callables are not configuration.
    assert "_private" not in payload["config"]
    assert "some_method" not in payload["config"]
    # The graph is half the method, so its identity is recorded with the run.
    assert payload["artifact"]["metadata"]["teacher_fingerprint"] == "abc123"
    assert payload["artifact"]["graph_stats"]["target_js_floor"] == 0.11
    assert "unserializable" not in payload["artifact"]["graph_stats"]
    assert set(payload["git"]) == {"sha", "dirty", "branch"}
    assert "python" in payload["env"]


def test_epoch_records_stay_separable_across_runs(tmp_path):
    """`epochs.jsonl` is append-mode on purpose; run_id is what separates runs."""
    first, second = new_run_id(), new_run_id()
    append_epoch_record(str(tmp_path), first, {"epoch": 1, "val": {"avg": 74.1}})
    append_epoch_record(str(tmp_path), first, {"epoch": 2, "val": {"avg": 74.6}})
    append_epoch_record(str(tmp_path), second, {"epoch": 1, "val": {"avg": 73.2}})

    assert len(load_epochs(str(tmp_path))) == 3
    only_first = load_epochs(str(tmp_path), run_id=first)
    assert [row["epoch"] for row in only_first] == [1, 2]
    assert [row["val"]["avg"] for row in only_first] == [74.1, 74.6]
    assert load_epochs(str(tmp_path / "nowhere")) == []


def test_geometry_probe_reports_the_expected_families():
    torch.manual_seed(0)
    model, tokenizer = _TinyEncoder(), _TinyTokenizer()
    texts = [f"probe sentence {i}" for i in range(64)]
    pairs = [(f"a text {i}", f"a text {i} again") for i in range(16)]

    stats = probe_geometry(model, tokenizer, texts, pairs=pairs, batch_size=16)

    for key in (
        "anisotropy",
        "cos_p50",
        "cos_p90",
        "cos_p99",
        "cos_std",
        "effective_rank",
        "uniformity",
        "alignment",
        "positive_cos_mean",
        "separation",
        "embedding_norm_mean",
        "probe_size",
    ):
        assert key in stats, f"{key} missing"
        assert stats[key] == stats[key], f"{key} is NaN"
    assert -1.0 <= stats["anisotropy"] <= 1.0
    assert stats["effective_rank"] > 0
    assert stats["uniformity"] <= 0.0, "uniformity is a log of a mean of exp(-x)"
    assert stats["probe_size"] == 64.0


def test_geometry_probe_detects_a_collapsed_space():
    """The failure the loss cannot see: every text mapped to one direction."""

    class _Collapsed(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, input_ids, attention_mask=None):
            batch, seq = input_ids.shape
            constant = torch.ones(batch, seq, 16) * self.weight
            return SimpleNamespace(last_hidden_state=constant)

    stats = probe_geometry(
        _Collapsed(), _TinyTokenizer(), [f"t{i}" for i in range(32)], batch_size=8
    )
    assert stats["anisotropy"] > 0.99, "a collapsed space must read as anisotropic"
    assert stats["effective_rank"] < 2.0


def test_geometry_probe_restores_training_mode():
    model = _TinyEncoder()
    model.train()
    probe_geometry(model, _TinyTokenizer(), ["a", "b", "c", "d"], batch_size=2)
    assert model.training, "the probe must not leave the model in eval mode"


def test_probe_set_is_deterministic_and_bounded():
    frame = pd.DataFrame({"text": [f"row {i}" for i in range(500)]})
    a = build_probe_set(frame, "text", size=64, seed=0)
    b = build_probe_set(frame, "text", size=64, seed=0)
    assert a == b, "a probe set that moves between runs measures nothing"
    assert len(a) == 64
    assert build_probe_set(frame, "text", size=64, seed=1) != a
    small = pd.DataFrame({"text": ["one", "two"]})
    assert build_probe_set(small, "text", size=64) == ["one", "two"]


def test_teacher_student_spearman_is_one_when_they_agree():
    torch.manual_seed(0)
    model, tokenizer = _TinyEncoder(), _TinyTokenizer()
    texts = [f"probe {i}" for i in range(48)]

    from src.distill.geometry import _encode

    teacher = _encode(model, tokenizer, texts, 8, 16)
    stats = probe_geometry(
        model, tokenizer, texts, teacher_embeddings=teacher, batch_size=16
    )
    # Same vectors on both sides: the rank correlation must be exactly 1.
    assert stats["teacher_student_spearman"] > 0.999
