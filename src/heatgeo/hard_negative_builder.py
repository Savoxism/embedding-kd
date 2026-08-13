import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


HARD_NEGATIVE_ARTIFACT_VERSION = 1


def _fingerprint(embeddings: torch.Tensor) -> str:
    array = embeddings.detach().to(torch.float32).cpu().numpy()
    digest = hashlib.sha1(np.ascontiguousarray(array).tobytes())
    digest.update(str(array.shape).encode("utf-8"))
    return digest.hexdigest()


def _metadata_matches(artifact: dict, metadata: dict) -> tuple[bool, str]:
    old = artifact.get("metadata", {})
    for key, value in metadata.items():
        if old.get(key) != value:
            return False, f"{key}: cached={old.get(key)!r} requested={value!r}"
    return True, ""


def _build_hard_negative_indices(
    teacher_embeddings: torch.Tensor,
    hard_neg_pool: int,
    source_ids: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    embeddings = F.normalize(teacher_embeddings.float(), p=2, dim=-1).cpu()
    n_items = int(embeddings.size(0))
    indices = np.full((n_items, hard_neg_pool), -1, dtype=np.int64)
    same_source_fill = np.zeros(n_items, dtype=np.int64)
    total_fill = np.zeros(n_items, dtype=np.int64)
    source_tensor = torch.from_numpy(source_ids)

    for start in tqdm(
        range(0, n_items, chunk_size), desc="Hard-negative teacher cosine"
    ):
        end = min(start + chunk_size, n_items)
        similarities = embeddings[start:end] @ embeddings.T
        row_ids = torch.arange(start, end)
        similarities[torch.arange(end - start), row_ids] = -float("inf")

        for local_row, anchor in enumerate(range(start, end)):
            row = similarities[local_row]
            same_mask = source_tensor == source_ids[anchor]
            same_mask[anchor] = False
            same_take = min(hard_neg_pool, int(same_mask.sum().item()))
            selected: list[int] = []
            if same_take:
                same_scores = row.masked_fill(~same_mask, -float("inf"))
                selected.extend(
                    torch.topk(same_scores, k=same_take).indices.tolist()
                )
            same_source_fill[anchor] = len(selected)

            remaining = hard_neg_pool - len(selected)
            if remaining:
                cross_mask = ~same_mask
                cross_mask[anchor] = False
                cross_take = min(remaining, int(cross_mask.sum().item()))
                if cross_take:
                    cross_scores = row.masked_fill(~cross_mask, -float("inf"))
                    selected.extend(
                        torch.topk(cross_scores, k=cross_take).indices.tolist()
                    )

            total_fill[anchor] = len(selected)
            if selected:
                indices[anchor, : len(selected)] = np.asarray(selected, dtype=np.int64)

    stats = {
        "n_items": float(n_items),
        "hard_neg_pool": float(hard_neg_pool),
        "hard_pool_fill_avg": float(total_fill.mean()),
        "hard_pool_fill_min": float(total_fill.min()),
        "same_source_fill_avg": float(same_source_fill.mean()),
        "same_source_fill_min": float(same_source_fill.min()),
    }
    return indices, stats


def build_or_load_hard_negative_artifact(
    teacher_embeddings: torch.Tensor,
    cache_path: str,
    hard_neg_pool: int,
    source_ids: Sequence[int] | None = None,
    chunk_size: int = 1024,
) -> dict:
    n_items = int(teacher_embeddings.size(0))
    if n_items < 2:
        raise ValueError("hard-negative mining requires at least two corpus rows")
    if hard_neg_pool <= 0:
        raise ValueError(f"hard_neg_pool must be positive, got {hard_neg_pool}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if source_ids is None:
        source_array = np.zeros(n_items, dtype=np.int64)
    else:
        source_array = np.asarray(source_ids, dtype=np.int64)
        if source_array.shape != (n_items,):
            raise ValueError(
                f"source_ids has shape {source_array.shape}, expected ({n_items},)"
            )

    metadata = {
        "artifact_version": HARD_NEGATIVE_ARTIFACT_VERSION,
        "candidate_sampling_mode": "random_hard_direct",
        "n_items": n_items,
        "hard_neg_pool": int(hard_neg_pool),
        "teacher_fingerprint": _fingerprint(teacher_embeddings),
        "source_fingerprint": hashlib.sha1(source_array.tobytes()).hexdigest(),
    }
    artifact_path = Path(cache_path)
    if artifact_path.exists():
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        matches, reason = _metadata_matches(artifact, metadata)
        if matches:
            print(f"Loaded hard-negative artifact from: {artifact_path}")
            return artifact
        print(f"Hard-negative artifact config mismatch, rebuilding: {artifact_path}")
        print(f"  first mismatch -> {reason}")

    hard_indices, stats = _build_hard_negative_indices(
        teacher_embeddings=teacher_embeddings,
        hard_neg_pool=int(hard_neg_pool),
        source_ids=source_array,
        chunk_size=int(chunk_size),
    )
    artifact = {
        "hard_neg_indices": torch.from_numpy(hard_indices).long(),
        "source_ids": torch.from_numpy(source_array).long(),
        "hard_negative_stats": stats,
        "metadata": metadata,
    }
    if str(artifact_path.parent):
        os.makedirs(artifact_path.parent, exist_ok=True)
    torch.save(artifact, artifact_path)
    print(
        f"Saved hard-negative artifact to: {artifact_path} | "
        f"fill={stats['hard_pool_fill_avg']:.1f}/{hard_neg_pool} "
        f"(min {stats['hard_pool_fill_min']:.0f}), "
        f"same_source={stats['same_source_fill_avg']:.1f}"
    )
    return artifact
