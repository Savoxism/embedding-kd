import os
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import eigsh
from tqdm import tqdm


def _as_tuple(values: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


def _compute_topk_cosine(embeddings: torch.Tensor, k: int, chunk_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = F.normalize(embeddings.float(), p=2, dim=-1).cpu()
    n_items = embeddings.size(0)
    k_eff = min(k + 1, n_items)
    all_indices = []
    all_scores = []

    for start in tqdm(range(0, n_items, chunk_size), desc="HeatGeo top-k cosine"):
        end = min(start + chunk_size, n_items)
        sims = embeddings[start:end] @ embeddings.T
        row_ids = torch.arange(start, end)
        sims[torch.arange(end - start), row_ids] = -float("inf")
        scores, indices = torch.topk(sims, k=k_eff, dim=-1)
        all_indices.append(indices[:, :k].cpu().numpy().astype(np.int64))
        all_scores.append(scores[:, :k].cpu().numpy().astype(np.float32))

    return np.concatenate(all_indices, axis=0), np.concatenate(all_scores, axis=0)


def _build_transition(
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    graph_k: int,
    graph_temp: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, sparse.csr_matrix]:
    n_items = top_indices.shape[0]
    top_sets = [set(top_indices[i, :graph_k].tolist()) for i in range(n_items)]
    row_neighbors: List[np.ndarray] = []
    row_probs: List[np.ndarray] = []
    row_scores: List[np.ndarray] = []
    fallback_flags = np.zeros(n_items, dtype=bool)
    rows = []
    cols = []
    vals = []

    for i in tqdm(range(n_items), desc="HeatGeo mutual kNN graph"):
        neighbors = []
        scores = []
        for pos, j in enumerate(top_indices[i, :graph_k]):
            j_int = int(j)
            if i in top_sets[j_int]:
                neighbors.append(j_int)
                scores.append(float(top_scores[i, pos]))

        if not neighbors:
            fallback_flags[i] = True
            fallback_k = min(graph_k, top_indices.shape[1])
            neighbors = [int(j) for j in top_indices[i, :fallback_k]]
            scores = [float(s) for s in top_scores[i, :fallback_k]]

        weights = np.exp(np.asarray(scores, dtype=np.float64) / max(graph_temp, 1e-6))
        weights = weights / max(float(weights.sum()), 1e-12)
        neighbor_arr = np.asarray(neighbors, dtype=np.int64)
        prob_arr = weights.astype(np.float32)

        row_neighbors.append(neighbor_arr)
        row_probs.append(prob_arr)
        row_scores.append(np.asarray(scores, dtype=np.float32))
        rows.extend([i] * len(neighbor_arr))
        cols.extend(neighbor_arr.tolist())
        vals.extend(prob_arr.tolist())

    transition = sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
    return row_neighbors, row_probs, row_scores, fallback_flags, transition


def _write_knn_graph_log(
    log_dir: str,
    row_neighbors: List[np.ndarray],
    row_probs: List[np.ndarray],
    row_scores: List[np.ndarray],
    fallback_flags: np.ndarray,
    graph_k: int,
) -> Tuple[str, Dict[str, float]]:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "knn_graph_neighbors.jsonl")
    degrees = np.asarray([len(neighbors) for neighbors in row_neighbors], dtype=np.float32)
    fallback_count = int(fallback_flags.sum())
    stats = {
        "n_items": int(len(row_neighbors)),
        "graph_k": int(graph_k),
        "fallback_count": fallback_count,
        "fallback_rate": float(fallback_count / max(1, len(row_neighbors))),
        "avg_degree": float(degrees.mean()) if degrees.size else 0.0,
        "min_degree": float(degrees.min()) if degrees.size else 0.0,
        "max_degree": float(degrees.max()) if degrees.size else 0.0,
    }

    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "summary", **stats}, sort_keys=True) + "\n")
        for idx, (neighbors, probs, scores) in enumerate(zip(row_neighbors, row_probs, row_scores)):
            handle.write(
                json.dumps(
                    {
                        "type": "node",
                        "idx": idx,
                        "fallback_used": bool(fallback_flags[idx]),
                        "neighbors": [int(value) for value in neighbors.tolist()],
                        "transition_probs": [float(value) for value in probs.tolist()],
                        "cosine_scores": [float(value) for value in scores.tolist()],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    print(
        "HeatGeo kNN graph log saved: "
        f"{log_path} | fallback={fallback_count}/{len(row_neighbors)} "
        f"({stats['fallback_rate']:.2%}), avg_degree={stats['avg_degree']:.2f}"
    )
    if fallback_count:
        fallback_examples = np.flatnonzero(fallback_flags)[:10].tolist()
        print(f"HeatGeo fallback node examples: {fallback_examples}")
    return log_path, stats


def _diffuse_one(
    start_idx: int,
    steps: int,
    row_neighbors: List[np.ndarray],
    row_probs: List[np.ndarray],
    keep_topk: int,
) -> Dict[int, float]:
    dist = {start_idx: 1.0}

    for _ in range(steps):
        next_dist: Dict[int, float] = {}
        for node, mass in dist.items():
            for nbr, prob in zip(row_neighbors[node], row_probs[node]):
                next_dist[int(nbr)] = next_dist.get(int(nbr), 0.0) + float(mass) * float(prob)

        if len(next_dist) > keep_topk:
            top_items = sorted(next_dist.items(), key=lambda item: item[1], reverse=True)[:keep_topk]
            next_dist = dict(top_items)
        dist = next_dist

    dist.pop(start_idx, None)
    return dist


def _build_diffusion_candidates(
    top_indices: np.ndarray,
    scales: Tuple[int, ...],
    row_neighbors: List[np.ndarray],
    row_probs: List[np.ndarray],
    diffusion_topk: int,
    hard_neg_k: int,
    random_neg_k: int,
    candidate_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_items = top_indices.shape[0]
    n_scales = len(scales)
    candidate_indices = np.zeros((n_items, candidate_size), dtype=np.int64)
    teacher_probs = np.zeros((n_scales, n_items, candidate_size), dtype=np.float32)
    rng = np.random.default_rng(seed)
    keep_topk = max(candidate_size * 4, diffusion_topk * 4, 64)

    for i in tqdm(range(n_items), desc="HeatGeo diffusion candidates"):
        scale_dists = [
            _diffuse_one(i, scale, row_neighbors, row_probs, keep_topk=keep_topk)
            for scale in scales
        ]

        candidates = []
        seen = {i}
        for dist in scale_dists:
            for node, _ in sorted(dist.items(), key=lambda item: item[1], reverse=True)[:diffusion_topk]:
                if node not in seen:
                    candidates.append(node)
                    seen.add(node)
                if len(candidates) >= candidate_size:
                    break

        for node in top_indices[i, : max(hard_neg_k * 4, hard_neg_k)]:
            node = int(node)
            if node not in seen:
                candidates.append(node)
                seen.add(node)
            if len(candidates) >= candidate_size - random_neg_k:
                break

        random_budget = max(0, candidate_size - len(candidates))
        random_budget = min(random_budget, random_neg_k if random_neg_k > 0 else random_budget)
        random_added = 0
        while random_added < random_budget and len(seen) < n_items:
            node = int(rng.integers(0, n_items))
            if node in seen:
                continue
            candidates.append(node)
            seen.add(node)
            random_added += 1

        for node in top_indices[i]:
            node = int(node)
            if len(candidates) >= candidate_size:
                break
            if node not in seen:
                candidates.append(node)
                seen.add(node)

        while len(candidates) < candidate_size:
            node = int(rng.integers(0, n_items))
            if node != i:
                candidates.append(node)

        candidate_arr = np.asarray(candidates[:candidate_size], dtype=np.int64)
        candidate_indices[i] = candidate_arr

        for scale_idx, dist in enumerate(scale_dists):
            probs = np.asarray([dist.get(int(node), 0.0) for node in candidate_arr], dtype=np.float32)
            prob_sum = float(probs.sum())
            if prob_sum <= 0.0:
                probs[:] = 1.0 / candidate_size
            else:
                probs /= prob_sum
            teacher_probs[scale_idx, i] = probs

    return candidate_indices, teacher_probs


def _compute_spectral_coords(
    transition: sparse.csr_matrix,
    spectral_dim: int,
) -> np.ndarray:
    n_items = transition.shape[0]
    if spectral_dim <= 0:
        return np.zeros((n_items, 0), dtype=np.float32)
    if n_items <= spectral_dim + 2:
        return np.zeros((n_items, spectral_dim), dtype=np.float32)

    adjacency = transition.maximum(transition.T).tocsr()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    nonzero = degrees > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    norm_adj = sparse.diags(inv_sqrt) @ adjacency @ sparse.diags(inv_sqrt)
    laplacian = sparse.eye(n_items, format="csr", dtype=np.float32) - norm_adj

    try:
        _, eigvecs = eigsh(laplacian, k=spectral_dim + 1, which="SM", tol=1e-3)
        coords = eigvecs[:, 1 : spectral_dim + 1].astype(np.float32)
    except Exception as exc:
        print(f"Warning: HeatGeo spectral decomposition failed: {exc}")
        coords = np.zeros((n_items, spectral_dim), dtype=np.float32)

    return coords


def _metadata_matches(artifact: Dict, metadata: Dict) -> bool:
    old = artifact.get("metadata", {})
    keys = [
        "n_items",
        "graph_k",
        "graph_temp",
        "diffusion_scales",
        "diffusion_topk",
        "hard_neg_k",
        "random_neg_k",
        "candidate_size",
        "spectral_dim",
        "use_spectral",
        "graph_log_version",
    ]
    return all(old.get(key) == metadata.get(key) for key in keys)


def build_or_load_heatgeo_artifact(
    teacher_embeddings: torch.Tensor,
    cache_path: str,
    log_dir: str,
    graph_k: int,
    graph_temp: float,
    diffusion_scales: Sequence[int],
    diffusion_topk: int,
    hard_neg_k: int,
    random_neg_k: int,
    candidate_size: int,
    spectral_dim: int,
    use_spectral: bool,
    seed: int,
) -> Dict[str, torch.Tensor]:
    n_items = int(teacher_embeddings.size(0))
    scales = _as_tuple(diffusion_scales)
    metadata = {
        "n_items": n_items,
        "graph_k": int(graph_k),
        "graph_temp": float(graph_temp),
        "diffusion_scales": scales,
        "diffusion_topk": int(diffusion_topk),
        "hard_neg_k": int(hard_neg_k),
        "random_neg_k": int(random_neg_k),
        "candidate_size": int(candidate_size),
        "spectral_dim": int(spectral_dim),
        "use_spectral": bool(use_spectral),
        "graph_log_version": 1,
    }

    artifact_path = Path(cache_path)
    if artifact_path.exists():
        artifact = torch.load(artifact_path, map_location="cpu")
        if _metadata_matches(artifact, metadata):
            graph_log_path = artifact.get("graph_log_path")
            if graph_log_path and os.path.exists(graph_log_path):
                print(f"Loaded HeatGeo artifact from: {artifact_path}")
                graph_stats = artifact.get("graph_stats", {})
                if graph_stats:
                    print(
                        "HeatGeo kNN graph summary: "
                        f"fallback={graph_stats.get('fallback_count', 0)}/{graph_stats.get('n_items', 0)} "
                        f"({float(graph_stats.get('fallback_rate', 0.0)):.2%}), "
                        f"avg_degree={float(graph_stats.get('avg_degree', 0.0)):.2f}"
                    )
                return artifact
            print(f"HeatGeo graph log missing, rebuilding artifact: {graph_log_path}")
        else:
            print(f"HeatGeo artifact config mismatch, rebuilding: {artifact_path}")

    if str(artifact_path.parent):
        os.makedirs(artifact_path.parent, exist_ok=True)
    topk_for_graph = max(graph_k, hard_neg_k + diffusion_topk, candidate_size)
    top_indices, top_scores = _compute_topk_cosine(teacher_embeddings, k=topk_for_graph)
    row_neighbors, row_probs, row_scores, fallback_flags, transition = _build_transition(
        top_indices=top_indices,
        top_scores=top_scores,
        graph_k=graph_k,
        graph_temp=graph_temp,
    )
    graph_log_path, graph_stats = _write_knn_graph_log(
        log_dir=log_dir,
        row_neighbors=row_neighbors,
        row_probs=row_probs,
        row_scores=row_scores,
        fallback_flags=fallback_flags,
        graph_k=graph_k,
    )
    candidate_indices, teacher_probs = _build_diffusion_candidates(
        top_indices=top_indices,
        scales=scales,
        row_neighbors=row_neighbors,
        row_probs=row_probs,
        diffusion_topk=diffusion_topk,
        hard_neg_k=hard_neg_k,
        random_neg_k=random_neg_k,
        candidate_size=candidate_size,
        seed=seed,
    )
    spectral_coords = _compute_spectral_coords(transition, spectral_dim if use_spectral else 0)

    artifact = {
        "candidate_indices": torch.from_numpy(candidate_indices).long(),
        "teacher_probs": torch.from_numpy(teacher_probs).float(),
        "spectral_coords": torch.from_numpy(spectral_coords).float(),
        "graph_log_path": graph_log_path,
        "graph_stats": graph_stats,
        "metadata": metadata,
    }
    torch.save(artifact, artifact_path)
    print(f"Saved HeatGeo artifact to: {artifact_path}")
    return artifact
