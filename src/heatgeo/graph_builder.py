import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ARTIFACT_VERSION = 3


def _as_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(v) for v in values}))


def _normalized_weights(scale_weights: Sequence[float], n_scales: int) -> np.ndarray:
    weights = np.asarray(list(scale_weights), dtype=np.float64)
    if weights.size == 0:
        weights = np.ones(n_scales, dtype=np.float64)
    if weights.size < n_scales:
        weights = np.concatenate(
            [weights, np.repeat(weights[-1], n_scales - weights.size)]
        )
    weights = weights[:n_scales]
    return weights / max(float(weights.sum()), 1e-12)


def _fingerprint(embeddings: torch.Tensor) -> str:
    """Content hash of the teacher embeddings.

    Without this, changing the teacher, the pooling, or the corpus while keeping
    n_items constant silently reuses a stale graph and every downstream ablation
    is measured against the wrong targets.
    """
    array = embeddings.detach().to(torch.float32).cpu().numpy()
    digest = hashlib.sha1(np.ascontiguousarray(array).tobytes())
    digest.update(str(array.shape).encode("utf-8"))
    return digest.hexdigest()


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

    Truncation to keep_topk is followed by renormalization. Truncating without
    renormalizing leaks mass at every step, and the leak compounds with r, so the
    later scales end up systematically under-weighted relative to r=1 -- which
    silently distorts the mixture target the loss actually optimizes.
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
        total = sum(next_dist.values())
        if total > 0.0:
            next_dist = {node: mass / total for node, mass in next_dist.items()}
        dist = next_dist

        if step in scales:
            snapshot = dict(dist)
            snapshot.pop(start_idx, None)
            total = sum(snapshot.values())
            if total > 0.0:
                snapshot = {node: mass / total for node, mass in snapshot.items()}
            snapshots[step] = snapshot

    return [snapshots[scale] for scale in scales]


def _build_diffusion_pools(
    top_indices: np.ndarray,
    scales: tuple[int, ...],
    weights: np.ndarray,
    row_neighbors: list[np.ndarray],
    row_probs: list[np.ndarray],
    pool_size: int,
    hard_neg_pool: int,
    source_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Per-anchor sparse diffusion support plus a same-source hard-negative pool.

    Candidate *sets* are no longer frozen here. The previous design precomputed one
    fixed 64-slot set per anchor and reused it for every epoch, so after epoch 1 the
    student had fit those exact comparisons and the objective had nothing left to
    say. What is precomputed now is the diffusion support; the candidate set is
    resampled from it each epoch by HeatGeoCandidateSampler.
    """
    n_items = top_indices.shape[0]
    n_scales = len(scales)
    keep_topk = max(pool_size * 4, 256)

    pool_indices = np.full((n_items, pool_size), -1, dtype=np.int64)
    pool_probs = np.zeros((n_scales, n_items, pool_size), dtype=np.float32)
    hard_neg_indices = np.full((n_items, hard_neg_pool), -1, dtype=np.int64)

    residual_mass = np.zeros(n_scales, dtype=np.float64)
    pool_fill = np.zeros(n_items, dtype=np.int64)
    hard_fill = np.zeros(n_items, dtype=np.int64)
    empty_scale = np.zeros(n_scales, dtype=np.int64)

    for i in tqdm(range(n_items), desc="HeatGeo diffusion pools"):
        scale_dists = _diffuse_snapshots(i, scales, row_neighbors, row_probs, keep_topk)

        mixture: dict[int, float] = {}
        for scale_idx, dist in enumerate(scale_dists):
            weight = float(weights[scale_idx])
            for node, mass in dist.items():
                mixture[node] = mixture.get(node, 0.0) + weight * mass

        ranked = sorted(mixture.items(), key=lambda item: item[1], reverse=True)
        selected = [node for node, _ in ranked[:pool_size]]
        pool_fill[i] = len(selected)
        if selected:
            pool_indices[i, : len(selected)] = np.asarray(selected, dtype=np.int64)

        pool_set = set(selected)
        for scale_idx, dist in enumerate(scale_dists):
            if not dist:
                empty_scale[scale_idx] += 1
                continue
            kept = np.asarray(
                [dist.get(node, 0.0) for node in selected], dtype=np.float64
            )
            kept_sum = float(kept.sum())
            total = float(sum(dist.values()))
            residual_mass[scale_idx] += max(0.0, total - kept_sum) / max(total, 1e-12)
            if kept_sum <= 0.0:
                empty_scale[scale_idx] += 1
                continue
            pool_probs[scale_idx, i, : len(selected)] = (kept / kept_sum).astype(
                np.float32
            )

        # Hard negatives: nearest teacher neighbours from the SAME source corpus that
        # carry no diffusion mass. Sampling negatives uniformly from a corpus made of
        # three disjoint datasets makes ~2/3 of them separable by domain alone, which
        # is why the discrimination task was exhausted within one epoch.
        source_i = source_ids[i]
        hard: list[int] = []
        for node in top_indices[i]:
            if len(hard) >= hard_neg_pool:
                break
            node = int(node)
            if node == i or node in pool_set:
                continue
            if source_ids[node] != source_i:
                continue
            hard.append(node)
        if len(hard) < hard_neg_pool:
            # Same-source pool exhausted: top up with cross-source nearest neighbours
            # rather than leaving the quota unfilled.
            for node in top_indices[i]:
                if len(hard) >= hard_neg_pool:
                    break
                node = int(node)
                if node == i or node in pool_set or node in hard:
                    continue
                hard.append(node)
        hard_fill[i] = len(hard)
        if hard:
            hard_neg_indices[i, : len(hard)] = np.asarray(hard, dtype=np.int64)

    stats = {
        "pool_size": float(pool_size),
        "pool_fill_avg": float(pool_fill.mean()),
        "pool_fill_min": float(pool_fill.min()),
        "hard_pool_fill_avg": float(hard_fill.mean()),
        "hard_pool_fill_min": float(hard_fill.min()),
    }
    for scale_idx, scale in enumerate(scales):
        stats[f"pool_residual_mass_r{scale}"] = float(
            residual_mass[scale_idx] / max(1, n_items)
        )
        stats[f"pool_empty_r{scale}"] = float(empty_scale[scale_idx] / max(1, n_items))
    return pool_indices, pool_probs, hard_neg_indices, stats


def _target_sharpness_stats(
    pool_indices: np.ndarray,
    pool_probs: np.ndarray,
    scales: tuple[int, ...],
    weights: np.ndarray,
) -> dict[str, float]:
    """Is the target informative, are the scales different, and what is the loss floor?

    KL(p || uniform-on-support) near zero means the target degenerates into a binary
    neighbour/non-neighbour label and carries no ranking signal. The Jensen-Shannon
    term is the exact irreducible value of L_diff when all scales share one student
    distribution, so it is the number the training loss can never go below.
    """
    stats: dict[str, float] = {}
    valid = pool_indices >= 0
    probs = np.clip(pool_probs.astype(np.float64), 0.0, None)
    probs = np.where(valid[None, :, :], probs, 0.0)

    entropies = np.zeros((len(scales), pool_indices.shape[0]), dtype=np.float64)
    for scale_idx, scale in enumerate(scales):
        p = probs[scale_idx]
        mask = p > 0
        supp_size = mask.sum(axis=-1).astype(np.float64)
        safe = np.where(mask, p, 1.0)
        entropy = -(np.where(mask, p * np.log(safe), 0.0)).sum(axis=-1)
        entropies[scale_idx] = entropy
        kl_uniform = np.log(np.maximum(supp_size, 1.0)) - entropy
        stats[f"target_support_r{scale}"] = float(supp_size.mean())
        stats[f"target_kl_uniform_r{scale}"] = float(kl_uniform.mean())
        stats[f"target_top1_r{scale}"] = float(p.max(axis=-1).mean())

    mixture = (probs * weights.reshape(-1, 1, 1)).sum(axis=0)
    mask = mixture > 0
    safe = np.where(mask, mixture, 1.0)
    mixture_entropy = -(np.where(mask, mixture * np.log(safe), 0.0)).sum(axis=-1)
    js = mixture_entropy - (entropies * weights.reshape(-1, 1)).sum(axis=0)
    stats["target_js_floor"] = float(js.mean())
    stats["target_js_floor_p90"] = float(np.percentile(js, 90))
    stats["target_mixture_entropy"] = float(mixture_entropy.mean())
    stats["target_mixture_top1"] = float(mixture.max(axis=-1).mean())

    clamped = np.clip(probs, 1e-12, None)
    clamped = clamped / clamped.sum(axis=-1, keepdims=True)
    for scale_idx in range(len(scales) - 1):
        a, b = clamped[scale_idx], clamped[scale_idx + 1]
        cross = (a * (np.log(a) - np.log(b))).sum(axis=-1)
        stats[f"target_cross_kl_r{scales[scale_idx]}_r{scales[scale_idx + 1]}"] = float(
            cross.mean()
        )
    return stats


_METADATA_KEYS = (
    "n_items",
    "graph_k",
    "graph_temp",
    "diffusion_scales",
    "scale_weights",
    "pool_size",
    "hard_neg_pool",
    "lazy_walk",
    "artifact_version",
    "teacher_fingerprint",
    "source_fingerprint",
)


def _metadata_matches(artifact: dict, metadata: dict) -> tuple[bool, str]:
    old = artifact.get("metadata", {})
    for key in _METADATA_KEYS:
        if old.get(key) != metadata.get(key):
            return False, f"{key}: cached={old.get(key)!r} requested={metadata.get(key)!r}"
    return True, ""


def build_or_load_heatgeo_artifact(
    teacher_embeddings: torch.Tensor,
    cache_path: str,
    log_dir: str,
    graph_k: int,
    graph_temp: float,
    diffusion_scales: Sequence[int],
    scale_weights: Sequence[float],
    pool_size: int,
    hard_neg_pool: int,
    source_ids: Sequence[int] | None = None,
) -> dict:
    n_items = int(teacher_embeddings.size(0))
    scales = _as_tuple(diffusion_scales)
    weights = _normalized_weights(scale_weights, len(scales))

    if source_ids is None:
        source_array = np.zeros(n_items, dtype=np.int64)
    else:
        source_array = np.asarray(source_ids, dtype=np.int64)
        if source_array.shape[0] != n_items:
            raise ValueError(
                f"source_ids has {source_array.shape[0]} entries but there are "
                f"{n_items} teacher embeddings"
            )

    metadata = {
        "n_items": n_items,
        "graph_k": int(graph_k),
        "graph_temp": float(graph_temp),
        "diffusion_scales": scales,
        "scale_weights": tuple(round(float(w), 8) for w in weights),
        "pool_size": int(pool_size),
        "hard_neg_pool": int(hard_neg_pool),
        "lazy_walk": True,
        "artifact_version": ARTIFACT_VERSION,
        "teacher_fingerprint": _fingerprint(teacher_embeddings),
        "source_fingerprint": hashlib.sha1(source_array.tobytes()).hexdigest(),
    }

    artifact_path = Path(cache_path)
    if artifact_path.exists():
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        matches, reason = _metadata_matches(artifact, metadata)
        if matches:
            graph_log_path = artifact.get("graph_log_path")
            if graph_log_path and os.path.exists(graph_log_path):
                print(f"Loaded HeatGeo artifact from: {artifact_path}")
                _print_graph_summary(artifact.get("graph_stats", {}), scales)
                return artifact
            print(f"HeatGeo graph log missing, rebuilding artifact: {graph_log_path}")
        else:
            print(f"HeatGeo artifact config mismatch, rebuilding: {artifact_path}")
            print(f"  first mismatch -> {reason}")

    if str(artifact_path.parent):
        os.makedirs(artifact_path.parent, exist_ok=True)
    topk_for_graph = min(n_items - 1, max(graph_k, hard_neg_pool, pool_size))
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
    pool_indices, pool_probs, hard_neg_indices, pool_stats = _build_diffusion_pools(
        top_indices=top_indices,
        scales=scales,
        weights=weights,
        row_neighbors=row_neighbors,
        row_probs=row_probs,
        pool_size=pool_size,
        hard_neg_pool=hard_neg_pool,
        source_ids=source_array,
    )
    graph_stats.update(pool_stats)
    graph_stats.update(
        _target_sharpness_stats(pool_indices, pool_probs, scales, weights)
    )

    artifact = {
        "pool_indices": torch.from_numpy(pool_indices).long(),
        "pool_probs": torch.from_numpy(pool_probs).float(),
        "hard_neg_indices": torch.from_numpy(hard_neg_indices).long(),
        "source_ids": torch.from_numpy(source_array).long(),
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
    if "pool_fill_avg" in graph_stats:
        print(
            "HeatGeo pools: "
            f"diffusion_fill={graph_stats['pool_fill_avg']:.1f}/"
            f"{graph_stats['pool_size']:.0f} (min {graph_stats['pool_fill_min']:.0f}), "
            f"hard_neg_fill={graph_stats['hard_pool_fill_avg']:.1f} "
            f"(min {graph_stats['hard_pool_fill_min']:.0f})"
        )
    for scale in scales:
        key = f"target_kl_uniform_r{scale}"
        if key in graph_stats:
            print(
                f"HeatGeo target r={scale}: support={graph_stats[f'target_support_r{scale}']:.1f}, "
                f"KL(p||uniform_on_support)={graph_stats[key]:.4f}, "
                f"top1={graph_stats[f'target_top1_r{scale}']:.4f}, "
                f"residual_mass_outside_pool={graph_stats.get(f'pool_residual_mass_r{scale}', 0.0):.2e}"
            )
    cross_keys = [key for key in graph_stats if key.startswith("target_cross_kl_")]
    for key in sorted(cross_keys):
        print(f"HeatGeo {key}={graph_stats[key]:.4f}")
    if "target_js_floor" in graph_stats:
        print(
            "HeatGeo irreducible L_diff floor (tied student temperature): "
            f"JS_omega={graph_stats['target_js_floor']:.4f} nats "
            f"(p90={graph_stats['target_js_floor_p90']:.4f}); "
            f"mixture H={graph_stats['target_mixture_entropy']:.4f}, "
            f"top1={graph_stats['target_mixture_top1']:.4f}"
        )
    low = [s for s in scales if graph_stats.get(f"target_kl_uniform_r{s}", 1.0) < 0.05]
    if low:
        print(
            f"WARNING: HeatGeo targets at scales {low} are close to uniform on their support "
            f"(KL < 0.05 nats) -- lower graph_temp, otherwise L_diff degenerates into a "
            f"binary neighbour/non-neighbour objective."
        )
    if float(graph_stats.get("target_js_floor", 1.0)) < 0.01:
        print(
            "WARNING: JS_omega < 0.01 nats -- the diffusion scales carry almost the same "
            "target, so with a tied student temperature the multi-scale objective is "
            "numerically identical to a single-scale one."
        )
