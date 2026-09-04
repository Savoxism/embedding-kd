#!/usr/bin/env python3
"""Plot teacher-weighted geometry error for Base / Batch-RKD / GGPKD.

The same deterministic probe, teacher-only ordering, and colour normalization
are used in all three panels. For a normalized teacher embedding matrix T and a
student embedding matrix S, each pixel is

    R_ij = p_i^T(j) * (cos_S(i, j) - cos_T(i, j))**2,
    p_i^T = softmax(cos_T(i, :) / temperature), j != i.

The row-sum mean of R is exactly the repository's geometry diagnostic E_hat.
For trained methods, R is averaged pixel-wise over the requested seeds; the
panel annotation reports mean +/- sample standard deviation of per-seed E_hat.

Typical usage is through ``heatmap.sh``. Direct usage:

    python scripts/ablation/geometry_heatmap.py \
      --runs-root runs/ablation/qwen3_0_6b_to_minilmv2_h384/paper_v1 \
      --teacher-cache cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/teacher_train.pt \
      --graph-artifact cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/paper_v1/graph_base.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable

# Matplotlib tries the user's config directory at import time. It is read-only in
# several training environments, so select a disposable cache before importing.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ggpkd-matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.distill.geometry import PROBE_DISTORTION_TEMP, build_probe_index


CHECKPOINT_RE = re.compile(r"student_epoch_(\d+)\.pt$")
SEED_RE = re.compile(r"seed(\d+)$")


def _load_teacher_cache(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        embeddings = payload
    elif isinstance(payload, dict) and torch.is_tensor(payload.get("embeddings")):
        embeddings = payload["embeddings"]
    else:
        raise TypeError(f"{path} does not contain an embedding tensor")
    if embeddings.ndim != 2:
        raise ValueError(
            f"teacher embeddings must have shape [N, D], got {tuple(embeddings.shape)}"
        )
    return embeddings.float()


def prepare_probe(
    train_data: Path,
    teacher_cache: Path,
    *,
    text_column: str,
    probe_size: int,
    probe_seed: int,
) -> tuple[list[str], torch.Tensor, np.ndarray, int]:
    """Reproduce GGPKD's exact-dedup corpus and deterministic geometry probe."""
    frame = pd.read_csv(train_data)
    if text_column not in frame.columns:
        raise ValueError(
            f"text column {text_column!r} not found in {train_data}; "
            f"available columns: {list(frame.columns)}"
        )
    original_texts = frame[text_column].astype(str)
    keep = np.flatnonzero(~original_texts.duplicated(keep="first").to_numpy())
    texts = original_texts.iloc[keep].tolist()

    teacher = _load_teacher_cache(teacher_cache)
    if len(teacher) == len(frame):
        teacher = teacher[torch.from_numpy(keep).long()]
    elif len(teacher) != len(texts):
        raise ValueError(
            f"teacher cache has {len(teacher)} rows, but {train_data} has "
            f"{len(frame)} raw and {len(texts)} exact-deduplicated rows"
        )

    probe_index = np.asarray(
        build_probe_index(len(texts), size=probe_size, seed=probe_seed),
        dtype=np.int64,
    )
    probe_texts = [texts[int(index)] for index in probe_index]
    probe_teacher = teacher[torch.from_numpy(probe_index).long()]
    return probe_texts, probe_teacher, probe_index, len(texts)


@torch.no_grad()
def encode_student(
    model,
    tokenizer,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Encode with the CLS convention used by GGPKD training and its probe."""
    model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            return_dict=True,
        )
        chunks.append(output.last_hidden_state[:, 0, :].float().cpu())
    return F.normalize(torch.cat(chunks, dim=0), p=2, dim=-1)


@torch.no_grad()
def teacher_weighted_error(
    teacher_embeddings: torch.Tensor,
    student_embeddings: torch.Tensor,
    *,
    temperature: float = PROBE_DISTORTION_TEMP,
) -> tuple[torch.Tensor, float]:
    """Return the per-relation E_hat contribution matrix and its scalar mean."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(teacher_embeddings) != len(student_embeddings):
        raise ValueError("teacher and student probes must contain the same examples")
    teacher = F.normalize(teacher_embeddings.float().cpu(), p=2, dim=-1)
    student = F.normalize(student_embeddings.float().cpu(), p=2, dim=-1)
    teacher_cos = teacher @ teacher.t()
    student_cos = student @ student.t()
    diagonal = torch.eye(len(teacher), dtype=torch.bool)
    weights = F.softmax(
        (teacher_cos / temperature).masked_fill(diagonal, float("-inf")), dim=-1
    )
    error = weights * (student_cos - teacher_cos).pow(2).masked_fill(diagonal, 0.0)
    e_hat = float(error.sum(dim=-1).mean())
    return error, e_hat


def teacher_graph_order(
    artifact_path: Path,
    probe_index: np.ndarray,
    *,
    corpus_size: int,
) -> np.ndarray:
    """Teacher-only graph seriation via reverse Cuthill-McKee on the probe graph."""
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    neighbors = artifact["transition_neighbors"].numpy()
    probabilities = artifact["transition_probs"].numpy()
    if len(neighbors) != corpus_size:
        raise ValueError(
            f"graph has {len(neighbors)} nodes, probe corpus has {corpus_size}; "
            "the graph and teacher cache are not aligned"
        )

    global_to_local = np.full(corpus_size, -1, dtype=np.int64)
    global_to_local[probe_index] = np.arange(len(probe_index), dtype=np.int64)
    probe_neighbors = neighbors[probe_index]
    probe_probabilities = probabilities[probe_index]
    valid_global = probe_neighbors >= 0
    safe_neighbors = np.where(valid_global, probe_neighbors, 0)
    local_columns = global_to_local[safe_neighbors]
    valid = valid_global & (local_columns >= 0) & (probe_probabilities > 0)
    local_rows = np.broadcast_to(
        np.arange(len(probe_index), dtype=np.int64)[:, None], probe_neighbors.shape
    )

    adjacency = coo_matrix(
        (
            probe_probabilities[valid].astype(np.float64),
            (local_rows[valid], local_columns[valid]),
        ),
        shape=(len(probe_index), len(probe_index)),
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    if adjacency.nnz == 0:
        raise ValueError("the induced teacher probe graph has no edges")
    return np.asarray(
        reverse_cuthill_mckee(adjacency, symmetric_mode=True), dtype=np.int64
    )


def _seed_of(path: Path) -> int:
    for part in reversed(path.parts):
        match = SEED_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"checkpoint path has no seed<N> directory: {path}")


def latest_checkpoints(
    root: Path, pattern: str, requested_seeds: Iterable[int]
) -> dict[int, Path]:
    """Resolve the highest saved epoch for every requested seed."""
    requested = set(requested_seeds)
    found: dict[int, tuple[int, Path]] = {}
    for path in root.glob(pattern):
        match = CHECKPOINT_RE.search(path.name)
        if not match:
            continue
        seed = _seed_of(path)
        if seed not in requested:
            continue
        epoch = int(match.group(1))
        if seed not in found or epoch > found[seed][0]:
            found[seed] = (epoch, path)
    missing = sorted(requested - found.keys())
    if missing:
        raise FileNotFoundError(
            f"missing checkpoints for seeds {missing}: {root / pattern}"
        )
    return {seed: found[seed][1] for seed in sorted(found)}


def load_student_weights(model, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint has no model_state_dict: {checkpoint_path}")
    model.load_state_dict(state, strict=True)


def method_error(
    model,
    tokenizer,
    checkpoints: dict[int, Path],
    texts: list[str],
    teacher: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    temperature: float,
) -> tuple[np.ndarray, float, float, dict[int, float]]:
    maps = []
    scores: dict[int, float] = {}
    for seed, checkpoint in checkpoints.items():
        print(f"encoding seed {seed}: {checkpoint}")
        load_student_weights(model, checkpoint)
        embeddings = encode_student(
            model,
            tokenizer,
            texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        error, e_hat = teacher_weighted_error(
            teacher, embeddings, temperature=temperature
        )
        maps.append(error.numpy())
        scores[seed] = e_hat
        print(f"  E_hat={e_hat:.6f}")
    mean_map = np.mean(np.stack(maps, axis=0), axis=0, dtype=np.float64)
    values = list(scores.values())
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_map, mean, std, scores


def _paper_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_heatmaps(
    maps: list[np.ndarray],
    labels: list[str],
    means: list[float],
    stds: list[float],
    order: np.ndarray,
    output_stem: Path,
) -> None:
    """Draw three seed-averaged maps with one honest shared colour scale."""
    ordered = [matrix[np.ix_(order, order)] for matrix in maps]
    non_diagonal = np.concatenate(
        [matrix[~np.eye(len(matrix), dtype=bool)] for matrix in ordered]
    )
    positive = non_diagonal[non_diagonal > 0]
    if positive.size == 0:
        raise ValueError("all geometry-error pixels are zero")
    vmax = float(np.percentile(positive, 99.0))
    norm = mpl.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=max(vmax, 1e-12))
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_bad("#efefec")

    _paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), constrained_layout=True)
    image = None
    diagonal = np.eye(len(order), dtype=bool)
    for axis, matrix, label, mean, std in zip(
        axes, ordered, labels, means, stds, strict=True
    ):
        shown = np.ma.array(matrix, mask=diagonal)
        image = axis.imshow(
            shown,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            rasterized=True,
            aspect="equal",
        )
        suffix = f"$\\hat{{E}}={mean:.4f}$"
        if std > 0:
            suffix = f"$\\hat{{E}}={mean:.4f}\\,\\pm\\,{std:.4f}$"
        axis.set_title(f"{label}\n{suffix}", pad=4)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#555555")

    assert image is not None
    colorbar = fig.colorbar(
        image, ax=axes, location="right", fraction=0.035, pad=0.02
    )
    colorbar.set_label("Teacher-weighted squared cosine error")
    colorbar.ax.tick_params(labelsize=7, width=0.5, length=2)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {output_stem.with_suffix('.png')}")


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--graph-artifact", type=Path, required=True)
    parser.add_argument(
        "--student-model",
        default="nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--seeds", type=_parse_seeds, default=(42, 43, 44))
    parser.add_argument("--probe-size", type=int, default=2048)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=PROBE_DISTORTION_TEMP)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--ours-pattern",
        default="full/full/seed*/weights/student_epoch_*.pt",
    )
    parser.add_argument(
        "--batch-pattern",
        default="support/batch_local/seed*/weights/student_epoch_*.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("latex/figures/fig_geometry_heatmap"),
        help="output path stem; PDF, PNG, JSON, and NPZ are written",
    )
    args = parser.parse_args()

    if args.probe_size < 2:
        parser.error("--probe-size must be at least 2")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    texts, teacher, probe_index, corpus_size = prepare_probe(
        args.train_data,
        args.teacher_cache,
        text_column=args.text_column,
        probe_size=args.probe_size,
        probe_seed=args.probe_seed,
    )
    order = teacher_graph_order(
        args.graph_artifact, probe_index, corpus_size=corpus_size
    )
    ours_checkpoints = latest_checkpoints(
        args.runs_root, args.ours_pattern, args.seeds
    )
    batch_checkpoints = latest_checkpoints(
        args.runs_root, args.batch_pattern, args.seeds
    )

    print(f"loading base student: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    model = AutoModel.from_pretrained(args.student_model).to(device)

    base_embeddings = encode_student(
        model,
        tokenizer,
        texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    base_error, base_e_hat = teacher_weighted_error(
        teacher, base_embeddings, temperature=args.temperature
    )
    print(f"student base E_hat={base_e_hat:.6f}")

    batch_map, batch_mean, batch_std, batch_scores = method_error(
        model,
        tokenizer,
        batch_checkpoints,
        texts,
        teacher,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
    )
    ours_map, ours_mean, ours_std, ours_scores = method_error(
        model,
        tokenizer,
        ours_checkpoints,
        texts,
        teacher,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
    )

    maps = [base_error.numpy(), batch_map, ours_map]
    labels = ["(a) Student base", "(b) Batch-relational KD", "(c) GGPKD (ours)"]
    means = [base_e_hat, batch_mean, ours_mean]
    stds = [0.0, batch_std, ours_std]
    plot_heatmaps(maps, labels, means, stds, order, args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        probe_index=probe_index,
        teacher_order=order,
        student_base=maps[0],
        batch_relational_kd=maps[1],
        ggpkd=maps[2],
    )
    metadata = {
        "definition": "p_teacher(i,j) * (cos_student(i,j) - cos_teacher(i,j))^2",
        "temperature": args.temperature,
        "probe_size": len(texts),
        "probe_seed": args.probe_seed,
        "student_base_e_hat": base_e_hat,
        "batch_relational_kd": {
            "mean_e_hat": batch_mean,
            "std_e_hat": batch_std,
            "per_seed": batch_scores,
            "checkpoints": {str(k): str(v) for k, v in batch_checkpoints.items()},
        },
        "ggpkd": {
            "mean_e_hat": ours_mean,
            "std_e_hat": ours_std,
            "per_seed": ours_scores,
            "checkpoints": {str(k): str(v) for k, v in ours_checkpoints.items()},
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.output.with_suffix('.npz')}")
    print(f"wrote {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
