from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.amp import GradScaler

from config.talas_config import (
    DEFAULT_TALAS_PAIR,
    TALAS_PAPER_PAIRS,
    TALASConfig,
    get_talas_paper_pair,
)
from distiller import KnowledgeDistiller
from scripts.summarize_talas import SEEDS, TASKS, aggregate_run
from src.cache_teacher import validate_cached_embeddings

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_talas_paper_pair_presets_are_exact():
    assert DEFAULT_TALAS_PAIR == "qwen3_0_6b_to_minilmv2_h384"
    assert TALAS_PAPER_PAIRS == {
        "qwen3_0_6b_to_minilmv2_h384": {
            "teacher": "Qwen/Qwen3-Embedding-0.6B",
            "student": "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base",
            "pooling_method": "last_token",
        },
        "bge_m3_to_minilmv2_h768": {
            "teacher": "BAAI/bge-m3",
            "student": "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base",
            "pooling_method": "cls",
        },
        "qwen3_4b_to_bert_base": {
            "teacher": "Qwen/Qwen3-Embedding-4B",
            "student": "google-bert/bert-base-uncased",
            "pooling_method": "last_token",
        },
    }
    config = TALASConfig()
    preset = TALAS_PAPER_PAIRS[DEFAULT_TALAS_PAIR]
    assert config.teacher_model_name == preset["teacher"]
    assert config.student_model_name == preset["student"]
    assert config.pooling_method == preset["pooling_method"]
    assert config.student_dtype == "float32"


def test_unknown_talas_pair_fails():
    with pytest.raises(ValueError, match="Unknown TALAS paper pair"):
        get_talas_paper_pair("not-a-pair")


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(3, 4, dtype=torch.int64),
        torch.zeros(3, 2, 4),
        torch.tensor([[float("nan")], [0.0], [1.0]]),
        torch.tensor([[float("inf")], [0.0], [1.0]]),
    ],
)
def test_talas_cache_validation_rejects_invalid_tensors(tensor):
    with pytest.raises((TypeError, ValueError)):
        validate_cached_embeddings(
            tensor, 3, cache_path="teacher.pt", require_single_layer=True
        )


def test_talas_cache_validation_accepts_finite_matrix_and_rejects_wrong_rows():
    tensor = torch.randn(3, 4)
    assert (
        validate_cached_embeddings(
            tensor, 3, cache_path="teacher.pt", require_single_layer=True
        )
        is tensor
    )
    with pytest.raises(ValueError, match="row mismatch"):
        validate_cached_embeddings(
            tensor, 2, cache_path="teacher.pt", require_single_layer=True
        )


@pytest.mark.parametrize("pair", tuple(TALAS_PAPER_PAIRS))
def test_shell_launcher_resolves_root_and_pair_from_any_working_directory(
    tmp_path, pair
):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "train_talas.sh"),
            pair,
            "--seed",
            "43",
            "--gpu",
            "7",
            "--dry-run",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    preset = TALAS_PAPER_PAIRS[pair]
    assert str(REPO_ROOT / "main.py") in result.stdout
    assert (
        str(REPO_ROOT / "data" / "train_set" / "merged_3_data_5k_each.csv")
        in result.stdout
    )
    assert preset["teacher"] in result.stdout
    assert preset["student"] in result.stdout
    assert f"--talas_pair {pair}" in result.stdout
    assert "--seed 43" in result.stdout


def test_shell_launcher_rejects_unknown_pair(tmp_path):
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "train_talas.sh"), "bad-pair"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Unknown TALAS pair" in result.stderr


def test_powershell_launcher_contains_canonical_portable_mapping():
    text = (REPO_ROOT / "scripts" / "train_talas.ps1").read_text(encoding="utf-8")
    assert "$PSScriptRoot" in text
    assert ".venv\\Scripts\\python.exe" in text
    for pair, preset in TALAS_PAPER_PAIRS.items():
        assert pair in text
        assert preset["teacher"] in text
        assert preset["student"] in text


class _TinyStudent(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 8, layers: int = 3):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(0.2))
                for _ in range(layers)
            ]
        )
        self.config = SimpleNamespace(hidden_size=dim)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        hidden = self.emb(input_ids)
        states = [hidden]
        for block in self.blocks:
            hidden = block(hidden)
            states.append(hidden)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=tuple(states) if output_hidden_states else None,
        )


def test_talas_smoke_step_updates_student_with_equal_projection_lr():
    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = SimpleNamespace(
        distill_method="talas",
        temperature=0.1,
        last_layer_idx=2,
        start_rkd=0,
        w_task=0.001,
        w_kd=0.75,
        w_struct=1.0,
        eps_norm=1e-12,
        learning_rate=2e-5,
        rho=0.05,
        epochs=1,
        min_lr=2e-6,
        warmup_ratio=0.0,
    )
    distiller.device_s = torch.device("cpu")
    distiller.model_student = _TinyStudent()
    distiller.criterion = None
    distiller.optimizer = None
    distiller.scheduler = None
    distiller.scaler = GradScaler("cuda", enabled=False)
    distiller.train_loader = [None]
    distiller.current_epoch = 0
    distiller.current_step = 0

    ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    batch = {
        "input_ids1_stu": ids,
        "attention_mask1_stu": torch.ones_like(ids),
        "input_ids2_stu": ids.clone(),
        "attention_mask2_stu": torch.ones_like(ids),
        "teacher_cls": torch.randn(4, 12),
    }
    before = distiller.model_student.emb.weight.detach().clone()
    loss, _ = distiller.train_step(batch)

    assert torch.isfinite(loss)
    assert not torch.equal(before, distiller.model_student.emb.weight.detach())
    assert {group["lr"] for group in distiller.optimizer.param_groups} == {2e-6}
    assert {group["initial_lr"] for group in distiller.optimizer.param_groups} == {2e-5}


def _write_synthetic_run(run_root: Path, pair: str, seed: int) -> None:
    status_dir = run_root / "status"
    run_dir = run_root / "runs" / pair / f"seed_{seed}"
    weights_dir = run_dir / "weights"
    status_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{pair}.seed_{seed}.exit").write_text("0\n", encoding="utf-8")
    torch.save(
        {"epoch": 5, "model_state_dict": {}},
        weights_dir / "student_epoch_5.pt",
    )

    score = 0.70 + 0.01 * (seed - 42)
    test = {"classification": {}, "pair": {}, "sts": {}}
    for _, family, stem, metric in TASKS:
        path = f"data/test_set/{stem}.csv"
        test[family][path] = score if family == "sts" else {metric: score}
    records = [
        {"method": "talas", "seed": seed, "train": {"loss": 1.0}},
        {"method": "talas", "seed": seed, "test": test},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_talas_aggregation_uses_sample_standard_deviation(tmp_path):
    for pair in TALAS_PAPER_PAIRS:
        for seed in SEEDS:
            _write_synthetic_run(tmp_path, pair, seed)

    aggregate = aggregate_run(tmp_path)
    for pair in TALAS_PAPER_PAIRS:
        for metric in ("Banking77", "Avg In", "Avg Out", "Avg All"):
            assert aggregate[pair][metric]["mean"] == pytest.approx(71.0)
            assert aggregate[pair][metric]["std"] == pytest.approx(1.0)


def test_talas_aggregation_fails_when_a_seed_is_missing(tmp_path):
    for pair in TALAS_PAPER_PAIRS:
        for seed in SEEDS:
            if pair == DEFAULT_TALAS_PAIR and seed == 44:
                continue
            _write_synthetic_run(tmp_path, pair, seed)
    with pytest.raises(ValueError, match="Missing exit-code file"):
        aggregate_run(tmp_path)
