#!/usr/bin/env python3
"""Three-panel motivation figure for batch-local versus graph-aware supervision.

Panels (a) and (b) aggregate teacher-relevant relation mass after ordering the
whole corpus by the teacher graph.  The old batch-local panel uses the exact
pair co-occurrence probability of a random full mini-batch, accumulated over
the requested epochs; it therefore reports an expectation rather than a
cherry-picked shuffle.  The new panel shows the deterministic top-K diffusion
support exposed by GGPKD.  Panel (c) aggregates the teacher-weighted reduction
in squared cosine error from batch-local KD to GGPKD on the same relations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ggpkd-matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ablation.geometry_heatmap import (  # noqa: E402
    _load_teacher_cache,
    encode_student,
    latest_checkpoints,
    load_student_weights,
    teacher_graph_order,
)
from src.ggpkd.candidate_sampler import GGPKDCandidateSampler  # noqa: E402
from src.ggpkd.policy import (  # noqa: E402
    derive_diffusion_quota,
    normalized_diffusion_weights,
)


def load_aligned_corpus(
    train_data: Path, teacher_cache: Path, text_column: str
) -> tuple[list[str], torch.Tensor]:
    """Load the exact-deduplicated graph corpus and its aligned teacher bank."""
    frame = pd.read_csv(train_data)
    if text_column not in frame.columns:
        raise ValueError(
            f"text column {text_column!r} not found; available={list(frame.columns)}"
        )
    raw = frame[text_column].astype(str)
    keep = np.flatnonzero(~raw.duplicated(keep="first").to_numpy())
    texts = raw.iloc[keep].tolist()

    teacher = _load_teacher_cache(teacher_cache)
    if len(teacher) == len(frame):
        teacher = teacher[torch.from_numpy(keep).long()]
    elif len(teacher) != len(texts):
        raise ValueError(
            f"teacher cache has {len(teacher)} rows; corpus has {len(frame)} raw "
            f"and {len(texts)} exact-deduplicated rows"
        )
    return texts, F.normalize(teacher.float(), p=2, dim=-1)


def block_ids_from_teacher_order(order: np.ndarray, n_bins: int) -> np.ndarray:
    """Map corpus node id to a contiguous block in teacher-graph order."""
    if not 2 <= n_bins <= len(order):
        raise ValueError(f"n_bins must be in [2, {len(order)}], got {n_bins}")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    return np.minimum(rank * n_bins // len(order), n_bins - 1)


def aggregate_edges(
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    block_ids: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Aggregate edge values, normalized by anchors in each row block."""
    matrix = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(matrix, (block_ids[rows], block_ids[columns]), values)
    anchors_per_block = np.bincount(block_ids, minlength=n_bins).astype(np.float64)
    matrix /= np.maximum(anchors_per_block[:, None], 1.0)
    return matrix


def relation_arrays(artifact: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten valid diffusion-pool edges and their teacher mixture mass."""
    pool = artifact["pool_indices"].cpu().numpy()
    probs = artifact["pool_probs"].cpu().numpy()
    scales = tuple(artifact["metadata"]["diffusion_scales"])
    weights = normalized_diffusion_weights(scales).astype(np.float64)
    mixture = np.einsum("s,sij->ij", weights, probs, dtype=np.float64)
    valid = (pool >= 0) & (mixture > 0)
    rows = np.broadcast_to(np.arange(len(pool))[:, None], pool.shape)[valid]
    return rows.astype(np.int64), pool[valid].astype(np.int64), mixture[valid]


def new_support_edges(
    artifact: dict, quota: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GGPKD top-K support and its captured teacher mixture mass."""
    sampler = GGPKDCandidateSampler(
        artifact=artifact,
        diffusion_quota=quota,
        hard_neg_k=0,
        random_neg_k=0,
        seed=seed,
        support_policy="topk",
    )
    weights = sampler.weights.astype(np.float64)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    for anchor in range(sampler.n_items):
        candidates, targets = sampler.sample(anchor)
        mixture = np.einsum("s,sc->c", weights, targets, dtype=np.float64)
        keep = mixture > 0
        rows.append(np.full(int(keep.sum()), anchor, dtype=np.int64))
        columns.append(candidates[keep].astype(np.int64))
        masses.append(mixture[keep])
    return np.concatenate(rows), np.concatenate(columns), np.concatenate(masses)


@torch.no_grad()
def edge_geometry_improvement(
    teacher: torch.Tensor,
    batch_student: torch.Tensor,
    ours_student: torch.Tensor,
    rows: np.ndarray,
    columns: np.ndarray,
    teacher_mass: np.ndarray,
    *,
    chunk_size: int = 200_000,
) -> tuple[np.ndarray, float, float]:
    """Per-edge weighted error reduction, positive when GGPKD is better."""
    values = np.empty(len(rows), dtype=np.float64)
    batch_total = 0.0
    ours_total = 0.0
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        r = torch.from_numpy(rows[start:stop]).long()
        c = torch.from_numpy(columns[start:stop]).long()
        mass = torch.from_numpy(teacher_mass[start:stop]).double()
        teacher_cos = (teacher[r] * teacher[c]).sum(dim=-1).double()
        batch_cos = (batch_student[r] * batch_student[c]).sum(dim=-1).double()
        ours_cos = (ours_student[r] * ours_student[c]).sum(dim=-1).double()
        batch_error = mass * (batch_cos - teacher_cos).square()
        ours_error = mass * (ours_cos - teacher_cos).square()
        values[start:stop] = (batch_error - ours_error).cpu().numpy()
        batch_total += float(batch_error.sum())
        ours_total += float(ours_error.sum())
    n_anchors = len(teacher)
    return values, batch_total / n_anchors, ours_total / n_anchors


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


def plot_three_panels(
    old_map: np.ndarray,
    new_map: np.ndarray,
    improvement_map: np.ndarray,
    *,
    old_coverage: float,
    new_coverage: float,
    batch_error: float,
    ours_error: float,
    output: Path,
) -> None:
    """Plot two exposure maps and one signed error-reduction map."""
    _paper_style()
    positive = np.concatenate([old_map.ravel(), new_map.ravel()])
    vmax = max(float(np.percentile(positive[positive > 0], 99.0)), 1e-12)
    exposure_norm = mpl.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    limit = max(float(np.percentile(np.abs(improvement_map), 99.0)), 1e-12)
    improvement_norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), constrained_layout=True)
    exposure_image = None
    for axis, matrix, title in (
        (axes[0], old_map, f"(a) Random mini-batch\ncoverage={old_coverage:.1%}"),
        (axes[1], new_map, f"(b) Graph-aware batch\ncoverage={new_coverage:.1%}"),
    ):
        exposure_image = axis.imshow(
            matrix,
            cmap="magma",
            norm=exposure_norm,
            interpolation="nearest",
            rasterized=True,
            aspect="equal",
        )
        axis.set_title(title, pad=4)
        axis.set_xticks([])
        axis.set_yticks([])

    improvement_image = axes[2].imshow(
        improvement_map,
        cmap="RdBu_r",
        norm=improvement_norm,
        interpolation="nearest",
        rasterized=True,
        aspect="equal",
    )
    axes[2].set_title(
        "(c) Error reduction by ours\n"
        f"$\\hat{{E}}$: {batch_error:.4f} $\\to$ {ours_error:.4f}",
        pad=4,
    )
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    assert exposure_image is not None
    exposure_bar = fig.colorbar(
        exposure_image, ax=axes[:2], location="bottom", fraction=0.06, pad=0.035
    )
    exposure_bar.set_label("Captured teacher-relation mass per anchor block")
    exposure_bar.ax.tick_params(labelsize=7, width=0.5, length=2)
    improvement_bar = fig.colorbar(
        improvement_image, ax=axes[2], location="bottom", fraction=0.06, pad=0.035
    )
    improvement_bar.set_label("Weighted error reduction (red: ours better)")
    improvement_bar.ax.tick_params(labelsize=7, width=0.5, length=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), dpi=400, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--diffusion-quota", type=int)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--ours-pattern", default="full/full/seed*/weights/student_epoch_*.pt"
    )
    parser.add_argument(
        "--batch-pattern",
        default="support/batch_local/seed*/weights/student_epoch_*.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("latex/figures/fig_batching_heatmap"),
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size must be at least 2")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    texts, teacher = load_aligned_corpus(
        args.train_data, args.teacher_cache, args.text_column
    )
    artifact = torch.load(args.graph_artifact, map_location="cpu", weights_only=False)
    if len(artifact["pool_indices"]) != len(texts):
        raise ValueError("graph artifact and exact-deduplicated corpus are misaligned")
    scales = tuple(artifact["metadata"]["diffusion_scales"])
    quota = args.diffusion_quota
    if quota is None:
        quota = derive_diffusion_quota(artifact["pool_probs"].numpy(), scales)

    print("ordering the full corpus by the teacher graph")
    order = teacher_graph_order(
        args.graph_artifact, np.arange(len(texts), dtype=np.int64), corpus_size=len(texts)
    )
    blocks = block_ids_from_teacher_order(order, args.bins)

    rows, columns, mass = relation_arrays(artifact)
    # For a uniformly shuffled DataLoader with drop_last=True, this is the exact
    # probability that two fixed nodes land in the same retained full batch in
    # one epoch. Accumulating by union avoids an arbitrary shuffle realization.
    retained = (len(texts) // args.batch_size) * args.batch_size
    p_one_epoch = (retained / len(texts)) * (
        (args.batch_size - 1) / (len(texts) - 1)
    )
    p_seen = 1.0 - (1.0 - p_one_epoch) ** args.epochs
    old_values = mass * p_seen
    old_map = aggregate_edges(rows, columns, old_values, blocks, args.bins)
    old_coverage = float(old_values.sum() / len(texts))

    new_rows, new_columns, new_mass = new_support_edges(artifact, quota, args.seed)
    # A selected graph relation is supervised whenever its anchor survives
    # drop_last. Account for the small remainder with the same union convention.
    p_anchor_seen = 1.0 - (1.0 - retained / len(texts)) ** args.epochs
    new_mass_seen = new_mass * p_anchor_seen
    new_map = aggregate_edges(
        new_rows, new_columns, new_mass_seen, blocks, args.bins
    )
    new_coverage = float(new_mass_seen.sum() / len(texts))

    checkpoints_batch = latest_checkpoints(
        args.runs_root, args.batch_pattern, (args.seed,)
    )
    checkpoints_ours = latest_checkpoints(
        args.runs_root, args.ours_pattern, (args.seed,)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    model = AutoModel.from_pretrained(args.student_model).to(device)

    print(f"encoding batch-local checkpoint: {checkpoints_batch[args.seed]}")
    load_student_weights(model, checkpoints_batch[args.seed])
    batch_student = encode_student(
        model,
        tokenizer,
        texts,
        device=device,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    print(f"encoding GGPKD checkpoint: {checkpoints_ours[args.seed]}")
    load_student_weights(model, checkpoints_ours[args.seed])
    ours_student = encode_student(
        model,
        tokenizer,
        texts,
        device=device,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )

    improvement, batch_error, ours_error = edge_geometry_improvement(
        teacher, batch_student, ours_student, rows, columns, mass
    )
    improvement_map = aggregate_edges(
        rows, columns, improvement, blocks, args.bins
    )
    plot_three_panels(
        old_map,
        new_map,
        improvement_map,
        old_coverage=old_coverage,
        new_coverage=new_coverage,
        batch_error=batch_error,
        ours_error=ours_error,
        output=args.output,
    )

    metadata = {
        "definition": {
            "old": "teacher mixture mass * expected cumulative batch co-occurrence",
            "new": "teacher mixture mass captured by top-K diffusion support",
            "improvement": "teacher mixture mass * (batch squared error - ours squared error)",
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "retained_rows_per_epoch": retained,
        "old_pair_seen_probability": p_seen,
        "new_anchor_seen_probability": p_anchor_seen,
        "bins": args.bins,
        "diffusion_scales": scales,
        "diffusion_quota": quota,
        "old_expected_coverage": old_coverage,
        "new_coverage": new_coverage,
        "batch_graph_weighted_error": batch_error,
        "ours_graph_weighted_error": ours_error,
        "batch_checkpoint": str(checkpoints_batch[args.seed]),
        "ours_checkpoint": str(checkpoints_ours[args.seed]),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        teacher_order=order,
        block_ids=blocks,
        old_exposure=old_map,
        new_exposure=new_map,
        error_reduction=improvement_map,
    )
    print(f"wrote {args.output.with_suffix('.pdf')}")
    print(f"wrote {args.output.with_suffix('.png')}")
    print(f"wrote {args.output.with_suffix('.json')}")
    print(f"wrote {args.output.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
