import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _as_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(v) for v in values}))


def _compute_topk_cosine(
    embeddings: torch.Tensor, k: int, chunk_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    n_items = top_indices.shape[0]
    top_sets = [set(top_indices[i, :graph_k].tolist()) for i in range(n_items)]
    row_neighbors: list[np.ndarray] = []
    row_probs: list[np.ndarray] = []
    row_scores: list[np.ndarray] = []
    fallback_flags = np.zeros(n_items, dtype=bool)

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

        # Softmax over neighbour cosines: sharpness is controlled by graph_temp, and it
        # directly sets how much ranking information the diffusion targets can carry.
        centered = np.asarray(scores, dtype=np.float64)
        centered = centered - centered.max()
        weights = np.exp(centered / max(graph_temp, 1e-6))
        weights = weights / max(float(weights.sum()), 1e-12)

        row_neighbors.append(np.asarray(neighbors, dtype=np.int64))
        row_probs.append(weights.astype(np.float32))
        row_scores.append(np.asarray(scores, dtype=np.float32))

    return row_neighbors, row_probs, row_scores, fallback_flags


def _write_knn_graph_log(
    log_dir: str,
    row_neighbors: list[np.ndarray],
    row_probs: list[np.ndarray],
    row_scores: list[np.ndarray],
    fallback_flags: np.ndarray,
    graph_k: int,
) -> tuple[str, dict[str, float]]:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "knn_graph_neighbors.jsonl")
    degrees = np.asarray(
        [len(neighbors) for neighbors in row_neighbors], dtype=np.float32
    )
    fallback_count = int(fallback_flags.sum())
    stats = {
        "n_items": len(row_neighbors),
        "graph_k": int(graph_k),
        "fallback_count": fallback_count,
        "fallback_rate": float(fallback_count / max(1, len(row_neighbors))),
        "avg_degree": float(degrees.mean()) if degrees.size else 0.0,
        "min_degree": float(degrees.min()) if degrees.size else 0.0,
        "max_degree": float(degrees.max()) if degrees.size else 0.0,
    }

    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "summary", **stats}, sort_keys=True) + "\n")
        handle.writelines(
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
            for idx, (neighbors, probs, scores) in enumerate(
                zip(row_neighbors, row_probs, row_scores)
            )
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


def _diffuse_snapshots(
    start_idx: int,
    scales: tuple[int, ...],
    row_neighbors: list[np.ndarray],
    row_probs: list[np.ndarray],
    keep_topk: int,
) -> list[dict[int, float]]:
    """Lazy random walk x <- (x + xP)/2, snapshotted at each scale.

    The lazy walk is what makes the scales a graded family: a plain walk on a mutual
    kNN graph mixes within two steps, so (P)^2 and (P)^4 collapse onto the same
    distribution and the multi-scale objective degenerates into a duplicated term.
    """
    snapshots: dict[int, dict[int, float]] = {}
    dist: dict[int, float] = {start_idx: 1.0}
    max_scale = max(scales)

    for step in range(1, max_scale + 1):
        next_dist: dict[int, float] = {}
        for node, mass in dist.items():
            for nbr, prob in zip(row_neighbors[node], row_probs[node]):
                next_dist[int(nbr)] = next_dist.get(int(nbr), 0.0) + float(
                    mass
                ) * float(prob)
        for node, mass in dist.items():
            next_dist[node] = next_dist.get(node, 0.0) + float(mass)
        next_dist = {node: mass * 0.5 for node, mass in next_dist.items()}

        if len(next_dist) > keep_topk:
            top_items = sorted(
                next_dist.items(), key=lambda item: item[1], reverse=True
            )[:keep_topk]
            next_dist = dict(top_items)
        dist = next_dist

        if step in scales:
            snapshot = dict(dist)
            snapshot.pop(start_idx, None)
            snapshots[step] = snapshot

    return [snapshots[scale] for scale in scales]


def _build_diffusion_candidates(
    top_indices: np.ndarray,
    scales: tuple[int, ...],
    row_neighbors: list[np.ndarray],
    row_probs: list[np.ndarray],
    diffusion_topk: int,
    hard_neg_k: int,
    random_neg_k: int,
    candidate_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n_items = top_indices.shape[0]
    n_scales = len(scales)
    candidate_indices = np.zeros((n_items, candidate_size), dtype=np.int64)
    teacher_probs = np.zeros((n_scales, n_items, candidate_size), dtype=np.float32)
    rng = np.random.default_rng(seed)
    keep_topk = max(candidate_size * 4, diffusion_topk * 4, 64)

    # Explicit slot quotas. The previous implementation appended greedily and checked
    # its break conditions after appending, so hard_neg_k / random_neg_k were never
    # honoured -- the actual composition was whatever the diffusion supports left over.
    diffusion_quota = max(0, candidate_size - hard_neg_k - random_neg_k)
    source_counts = {"diffusion": 0, "hard": 0, "random": 0, "pad": 0}
    uniform_fallback = np.zeros(n_scales, dtype=np.int64)

    for i in tqdm(range(n_items), desc="HeatGeo diffusion candidates"):
        scale_dists = _diffuse_snapshots(i, scales, row_neighbors, row_probs, keep_topk)
        ranked_per_scale = [
            [
                node
                for node, _ in sorted(
                    dist.items(), key=lambda item: item[1], reverse=True
                )[:diffusion_topk]
            ]
            for dist in scale_dists
        ]

        candidates: list[int] = []
        seen = {i}

        # Round-robin across scales so every scale contributes candidates. Taking the
        # scales in order instead lets the sharpest scale exhaust the budget, which
        # leaves the broader scales with no support to be scored on.
        cursors = [0] * n_scales
        while len(candidates) < diffusion_quota and any(
            cursors[s] < len(ranked_per_scale[s]) for s in range(n_scales)
        ):
            for s in range(n_scales):
                while cursors[s] < len(ranked_per_scale[s]):
                    node = int(ranked_per_scale[s][cursors[s]])
                    cursors[s] += 1
                    if node not in seen:
                        candidates.append(node)
                        seen.add(node)
                        break
                if len(candidates) >= diffusion_quota:
                    break
        source_counts["diffusion"] += len(candidates)

        hard_start = len(candidates)
        hard_target = hard_start + hard_neg_k
        for node in top_indices[i]:
            if len(candidates) >= hard_target:
                break
            node = int(node)
            if node not in seen:
                candidates.append(node)
                seen.add(node)
        source_counts["hard"] += len(candidates) - hard_start

        random_target = min(candidate_size, len(candidates) + random_neg_k)
        random_start = len(candidates)
        attempts = 0
        while (
            len(candidates) < random_target
            and len(seen) < n_items
            and attempts < 20 * n_items
        ):
            attempts += 1
            node = int(rng.integers(0, n_items))
            if node in seen:
                continue
            candidates.append(node)
            seen.add(node)
        source_counts["random"] += len(candidates) - random_start

        pad_start = len(candidates)
        for node in top_indices[i]:
            if len(candidates) >= candidate_size:
                break
            node = int(node)
            if node not in seen:
                candidates.append(node)
                seen.add(node)
        while len(candidates) < candidate_size:
            node = int(rng.integers(0, n_items))
            if node != i:
                candidates.append(node)
        source_counts["pad"] += len(candidates) - pad_start

        candidate_arr = np.asarray(candidates[:candidate_size], dtype=np.int64)
        candidate_indices[i] = candidate_arr

        for scale_idx, dist in enumerate(scale_dists):
            probs = np.asarray(
                [dist.get(int(node), 0.0) for node in candidate_arr], dtype=np.float32
            )
            prob_sum = float(probs.sum())
            if prob_sum <= 0.0:
                uniform_fallback[scale_idx] += 1
                probs[:] = 1.0 / candidate_size
            else:
                probs /= prob_sum
            teacher_probs[scale_idx, i] = probs

    stats = {
        f"candidate_src_{key}": float(value / max(1, n_items))
        for key, value in source_counts.items()
    }
    for scale_idx, scale in enumerate(scales):
        stats[f"uniform_fallback_r{scale}"] = float(
            uniform_fallback[scale_idx] / max(1, n_items)
        )
    return candidate_indices, teacher_probs, stats


def _target_sharpness_stats(
    teacher_probs: np.ndarray, scales: tuple[int, ...]
) -> dict[str, float]:
    """Is the target actually informative, and are the scales actually different?

    KL(p || uniform-on-support) near zero means the target degenerates into a binary
    neighbour/non-neighbour label and carries no ranking signal; cross-scale KL near
    zero means an extra scale adds a duplicated term rather than new structure.
    """
    stats: dict[str, float] = {}
    probs = np.clip(teacher_probs.astype(np.float64), 0.0, None)
    support = probs > 0

    for scale_idx, scale in enumerate(scales):
        p = probs[scale_idx]
        mask = support[scale_idx]
        supp_size = mask.sum(axis=-1).astype(np.float64)
        safe = np.where(mask, p, 1.0)
        entropy = -(np.where(mask, p * np.log(safe), 0.0)).sum(axis=-1)
        kl_uniform = np.log(np.maximum(supp_size, 1.0)) - entropy
        stats[f"target_support_r{scale}"] = float(supp_size.mean())
        stats[f"target_kl_uniform_r{scale}"] = float(kl_uniform.mean())
        stats[f"target_top1_r{scale}"] = float(p.max(axis=-1).mean())

    clamped = np.clip(probs, 1e-12, None)
    clamped = clamped / clamped.sum(axis=-1, keepdims=True)
    for scale_idx in range(len(scales) - 1):
        a, b = clamped[scale_idx], clamped[scale_idx + 1]
        cross = (a * (np.log(a) - np.log(b))).sum(axis=-1)
        stats[f"target_cross_kl_r{scales[scale_idx]}_r{scales[scale_idx + 1]}"] = float(
            cross.mean()
        )
    return stats


def _metadata_matches(artifact: dict, metadata: dict) -> bool:
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
        "lazy_walk",
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
    seed: int,
) -> dict[str, torch.Tensor]:
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
        "lazy_walk": True,
        "graph_log_version": 2,
    }

    artifact_path = Path(cache_path)
    if artifact_path.exists():
        artifact = torch.load(artifact_path, map_location="cpu")
        if _metadata_matches(artifact, metadata):
            graph_log_path = artifact.get("graph_log_path")
            if graph_log_path and os.path.exists(graph_log_path):
                print(f"Loaded HeatGeo artifact from: {artifact_path}")
                _print_graph_summary(artifact.get("graph_stats", {}), scales)
                return artifact
            print(f"HeatGeo graph log missing, rebuilding artifact: {graph_log_path}")
        else:
            print(f"HeatGeo artifact config mismatch, rebuilding: {artifact_path}")

    if str(artifact_path.parent):
        os.makedirs(artifact_path.parent, exist_ok=True)
    topk_for_graph = max(graph_k, hard_neg_k + diffusion_topk, candidate_size)
    top_indices, top_scores = _compute_topk_cosine(teacher_embeddings, k=topk_for_graph)
    row_neighbors, row_probs, row_scores, fallback_flags = _build_transition(
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
    candidate_indices, teacher_probs, candidate_stats = _build_diffusion_candidates(
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
    graph_stats.update(candidate_stats)
    graph_stats.update(_target_sharpness_stats(teacher_probs, scales))

    artifact = {
        "candidate_indices": torch.from_numpy(candidate_indices).long(),
        "teacher_probs": torch.from_numpy(teacher_probs).float(),
        "graph_log_path": graph_log_path,
        "graph_stats": graph_stats,
        "metadata": metadata,
    }
    torch.save(artifact, artifact_path)
    print(f"Saved HeatGeo artifact to: {artifact_path}")
    _print_graph_summary(graph_stats, scales)
    return artifact


def _print_graph_summary(
    graph_stats: dict[str, float], scales: tuple[int, ...]
) -> None:
    if not graph_stats:
        return
    print(
        "HeatGeo kNN graph summary: "
        f"fallback={graph_stats.get('fallback_count', 0)}/{graph_stats.get('n_items', 0)} "
        f"({float(graph_stats.get('fallback_rate', 0.0)):.2%}), "
        f"avg_degree={float(graph_stats.get('avg_degree', 0.0)):.2f}"
    )
    if "candidate_src_diffusion" in graph_stats:
        print(
            "HeatGeo candidate slots (avg): "
            f"diffusion={graph_stats['candidate_src_diffusion']:.2f}, "
            f"hard={graph_stats['candidate_src_hard']:.2f}, "
            f"random={graph_stats['candidate_src_random']:.2f}, "
            f"pad={graph_stats['candidate_src_pad']:.2f}"
        )
    for scale in scales:
        key = f"target_kl_uniform_r{scale}"
        if key in graph_stats:
            print(
                f"HeatGeo target r={scale}: support={graph_stats[f'target_support_r{scale}']:.1f}, "
                f"KL(p||uniform_on_support)={graph_stats[key]:.4f}, "
                f"top1={graph_stats[f'target_top1_r{scale}']:.4f}"
            )
    cross_keys = [key for key in graph_stats if key.startswith("target_cross_kl_")]
    for key in sorted(cross_keys):
        print(f"HeatGeo {key}={graph_stats[key]:.4f}")
    low = [s for s in scales if graph_stats.get(f"target_kl_uniform_r{s}", 1.0) < 0.05]
    if low:
        print(
            f"WARNING: HeatGeo targets at scales {low} are close to uniform on their support "
            f"(KL < 0.05 nats) -- lower graph_temp, otherwise L_diff degenerates into a "
            f"binary neighbour/non-neighbour objective."
        )
