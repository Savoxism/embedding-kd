import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .policy import (
    FIXED_BANDWIDTH_TEMP,
    hard_negative_pool_size,
    normalized_diffusion_weights,
)

# 8: restore padded transition rows required by L_row. Version 7 diffusion targets
#    remain mathematically valid, but do not contain the arrays needed to sample and
#    supervise non-anchor rows, so those caches must be rebuilt once.
ARTIFACT_VERSION = 8

# Anchors diffused per sparse matrix product. The intermediate X @ P holds up to
# block_size * keep_topk * max_degree nonzeros, so this trades memory for the
# number of scipy calls; 256 keeps the intermediate under a few hundred MB at
# graph_k=200.
DIFFUSION_BLOCK = 256

# Memory guards, not modelling choices, which is why they are constants here and
# not configuration. What a row actually keeps is decided by truncation_tolerance;
# these only bound the arrays while that decision is being made, and the build
# reports pool_capped_rows / walk_capped_rows if either ever binds first -- at
# which point the tolerance is no longer a guarantee and the number must go up.
DIFFUSION_ROW_CAP = 4096
POOL_ROW_CAP = 2048


def _as_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(v) for v in values}))


def _softmax_at(scores: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    """Softmax of `beta * scores` and its Shannon entropy, in nats."""
    shifted = beta * scores
    shifted -= shifted.max()
    weights = np.exp(shifted)
    probs = weights / weights.sum()
    entropy = -float((probs * np.log(np.maximum(probs, 1e-300))).sum())
    return probs, entropy


def _entropic_affinity(
    scores: np.ndarray,
    target_entropy: float,
    max_iter: int = 60,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float, bool]:
    """Softmax over `scores` at the unique temperature that hits `target_entropy`.

    Writing p_j(beta) proportional to exp(beta * s_j) with beta = 1/tau,

        H(beta) = log Z(beta) - beta * E_p[s],    dH/dbeta = -beta * Var_p(s),

    so H is strictly decreasing in beta -- strictly increasing in tau -- whenever
    the scores are not all equal, and it sweeps (0, log d) as tau sweeps (0, inf).
    The root is therefore unique for any target in that interval, and bisection
    cannot fail; the bracket is found by doubling first. This is the entropic
    affinity of Hinton and Roweis (2002), whose properties and numerics are
    analysed by Vladymyrov and Carreira-Perpinan (2013).

    The reason to prefer it over one global temperature is not adaptivity per se.
    Under an affine rescaling of the teacher's similarity scale, s -> a*s + b with
    a > 0, the solution moves to tau' = a*tau and the row is *unchanged*:

        exp((a*s_j + b) / (a*tau)) proportional to exp(s_j / tau).

    A fixed temperature has no such invariance, which is exactly why it has to be
    retuned for every teacher whose cosines are spread differently.

    Returns:
        probs: the transition row.
        tau: its temperature -- the student must match this row at the same value.
        clamped: True if the target entropy was unreachable (perplexity >= degree)
            and the row was solved against the largest attainable entropy instead.
    """
    degree = scores.size
    if degree <= 1:
        return np.ones(degree, dtype=np.float64), 1.0, True

    max_entropy = float(np.log(degree))
    # log d is the supremum, attained only as tau -> inf. Asking for it exactly
    # would send the bracket to infinity, so a row whose degree is at or below the
    # requested perplexity is solved just under its own ceiling instead.
    ceiling = max_entropy * (1.0 - 1e-3)
    clamped = target_entropy >= ceiling
    goal = min(target_entropy, ceiling)

    if float(np.ptp(scores)) <= 0.0:
        # Every neighbour scores the same: the row is uniform at every temperature.
        return np.full(degree, 1.0 / degree, dtype=np.float64), 1.0, True

    # Bracket: low beta is the high-entropy end, so grow beta until entropy drops
    # below the goal.
    beta_lo = 1e-12
    beta_hi = 1.0
    for _ in range(max_iter):
        _, entropy_hi = _softmax_at(scores, beta_hi)
        if entropy_hi <= goal:
            break
        beta_lo = beta_hi
        beta_hi *= 2.0

    probs = None
    for _ in range(max_iter):
        beta = 0.5 * (beta_lo + beta_hi)
        probs, entropy = _softmax_at(scores, beta)
        if abs(entropy - goal) <= tol:
            break
        # H decreases in beta: too much entropy means beta must grow.
        if entropy > goal:
            beta_lo = beta
        else:
            beta_hi = beta

    beta = 0.5 * (beta_lo + beta_hi)
    if probs is None:
        probs, _ = _softmax_at(scores, beta)
    return probs, 1.0 / beta, clamped


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
    perplexity: float | None,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict
]:
    """Mutual-kNN neighbour lists and their transition rows.

    With `perplexity` set, each row gets its own temperature from the entropic
    affinity above, so every node's neighbour distribution carries the same
    effective number of neighbours and the graph is invariant to the teacher's
    similarity scale. With `perplexity=None` every row uses `graph_temp`, which is
    the fixed-bandwidth baseline that arm exists to be compared against.
    """
    n_items = top_indices.shape[0]
    top_sets = [set(top_indices[i, :graph_k].tolist()) for i in range(n_items)]
    row_neighbors: list[np.ndarray] = []
    row_probs: list[np.ndarray] = []
    row_scores: list[np.ndarray] = []
    row_temps = np.zeros(n_items, dtype=np.float64)
    fallback_flags = np.zeros(n_items, dtype=bool)
    target_entropy = None if perplexity is None else float(np.log(float(perplexity)))
    clamped_rows = 0

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

        # Softmax over neighbour cosines. The temperature is either solved per row
        # for a fixed perplexity, or shared across rows in the baseline arm; either
        # way the value used here is the one the student has to match this row at,
        # so it is stored alongside the row.
        score_array = np.asarray(scores, dtype=np.float64)
        if target_entropy is None:
            centered = score_array - score_array.max()
            weights = np.exp(centered / max(graph_temp, 1e-6))
            weights = weights / max(float(weights.sum()), 1e-12)
            tau = float(graph_temp)
        else:
            weights, tau, clamped = _entropic_affinity(score_array, target_entropy)
            clamped_rows += int(clamped)

        row_temps[i] = tau
        row_neighbors.append(np.asarray(neighbors, dtype=np.int64))
        row_probs.append(weights.astype(np.float32))
        row_scores.append(np.asarray(scores, dtype=np.float32))

    temp_stats = {
        "row_temp_mean": float(row_temps.mean()) if n_items else 0.0,
        "row_temp_min": float(row_temps.min()) if n_items else 0.0,
        "row_temp_max": float(row_temps.max()) if n_items else 0.0,
        "row_temp_p50": float(np.median(row_temps)) if n_items else 0.0,
        # Rows whose degree was at or below the requested perplexity: their target
        # entropy was unreachable and they were solved just under their own ceiling.
        "perplexity_clamped_rows": int(clamped_rows),
        "perplexity_clamped_rate": float(clamped_rows / max(1, n_items)),
    }
    return row_neighbors, row_probs, row_scores, fallback_flags, row_temps, temp_stats


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


def _transition_matrix(
    row_neighbors: list[np.ndarray],
    row_probs: list[np.ndarray],
    n_items: int,
) -> sp.csr_matrix:
    """Row-stochastic P as CSR, so a diffusion step is one sparse matrix product."""
    sizes = np.fromiter(
        (int(neighbors.size) for neighbors in row_neighbors),
        dtype=np.int64,
        count=n_items,
    )
    indptr = np.zeros(n_items + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    matrix = sp.csr_matrix(
        (
            np.concatenate(row_probs).astype(np.float64),
            np.concatenate(row_neighbors).astype(np.int64),
            indptr,
        ),
        shape=(n_items, n_items),
    )
    matrix.sort_indices()
    return matrix


def _mass_prefix(data: np.ndarray, total: float, tolerance: float) -> np.ndarray:
    """Positions of the smallest set of entries carrying at least 1 - tolerance.

    Truncating a row to a set S with p(S) = 1 - delta and renormalizing gives a
    row ptilde with, exactly,

        TV(p, ptilde) = delta,   KL(ptilde || p) = -log(1 - delta) <= delta/(1-delta),

    so a tolerance on the discarded mass is a bound on the target perturbation in
    nats -- the units of the loss itself. That is what lets one stated tolerance
    replace a capacity knob per truncation site. The bound is per truncation; the
    lazy walk truncates once per step, so after r steps the total is at most r
    times it.
    """
    order = np.argsort(-data)
    cumulative = np.cumsum(data[order])
    needed = (1.0 - tolerance) * total
    keep = int(np.searchsorted(cumulative, needed) + 1)
    return order[: min(keep, order.size)]


def _truncate_renormalize(
    matrix: sp.csr_matrix,
    keep_topk: int,
    tolerance: float | None = None,
    protect: np.ndarray | None = None,
) -> tuple[sp.csr_matrix, float, int]:
    """Row-wise truncation to `tolerance` of discarded mass, then renormalize.

    `keep_topk` is a memory ceiling, not a modelling choice: the row is cut to the
    smallest prefix carrying 1 - tolerance, and only clipped at keep_topk if that
    prefix would be larger. Rows where the ceiling binds are counted and reported,
    because for those the tolerance above is no longer a guarantee.

    `protect` names one column per row that is kept unconditionally and left out of
    the tolerance budget. It exists because the row this function truncates is not
    the row that becomes a target: the lazy walk's snapshot has its self-mass
    dropped and is renormalized afterwards. At step r the self entry alone carries
    at least 2^-r, so spending the budget on the whole row spends it against a total
    the target never sees -- at r=1 the self entry is half the row, the surviving
    non-self mass came out at 1 - 2*tolerance, and the stated per-step tolerance
    was off by a factor of two on the one scale the temperature tie binds
    (measured: 0.9874 mean retained against a claimed 0.99). Excluding the
    protected column makes the discarded fraction *of the target* equal to
    `tolerance` at every scale, which is what the TV/KL bound in `_mass_prefix` is
    stated over.

    Truncating without renormalizing leaks mass at every step, and the leak
    compounds with r, so the later scales end up systematically under-weighted
    relative to r=1 -- which silently distorts the mixture target the loss
    actually optimizes.

    Renormalizing hides the leak from the row sums but not from the distribution:
    dropping the tail and rescaling the head makes the walk *sharper* than the true
    lazy walk, and the distortion grows with r. Since the broad scales exist
    precisely to be broad, this is the one approximation in the build that can
    quietly collapse the multi-scale objective, so the dropped fraction is returned
    and logged rather than discarded.
    """
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    n_rows = matrix.shape[0]
    kept_indices: list[np.ndarray] = []
    kept_data: list[np.ndarray] = []
    new_indptr = np.zeros(n_rows + 1, dtype=np.int64)
    dropped = 0.0
    capped = 0

    for row in range(n_rows):
        start, end = indptr[row], indptr[row + 1]
        row_indices, row_data = indices[start:end], data[start:end]
        full_total = float(row_data.sum())
        # Split off the protected column before anything else, so neither the
        # keep_topk cut nor the tolerance budget can spend itself on mass the
        # target will discard anyway.
        held_index = np.empty(0, dtype=row_indices.dtype)
        held_data = np.empty(0, dtype=row_data.dtype)
        if protect is not None:
            is_held = row_indices == protect[row]
            if is_held.any():
                held_index, held_data = row_indices[is_held], row_data[is_held]
                row_indices, row_data = row_indices[~is_held], row_data[~is_held]
        # The budget is stated over the mass that survives to the target, i.e. the
        # row minus whatever is held out.
        budget_total = full_total - float(held_data.sum())
        room = max(keep_topk - held_index.size, 1)
        if row_data.size > room:
            # Cheap O(n) cut to the ceiling first, so the sort inside _mass_prefix
            # only ever runs on `room` entries.
            top = np.argpartition(-row_data, room - 1)[:room]
            row_indices, row_data = row_indices[top], row_data[top]
            if float(row_data.sum()) < (1.0 - (tolerance or 0.0)) * budget_total:
                capped += 1
        if tolerance is not None and budget_total > 0.0 and row_data.size > 1:
            keep = _mass_prefix(row_data, budget_total, tolerance)
            row_indices, row_data = row_indices[keep], row_data[keep]
        # Reported before the protected column is put back: the discarded fraction
        # is what the tolerance bounds, and it is stated over the mass the target
        # actually keeps, not over a row whose self entry is dropped downstream.
        if budget_total > 0.0:
            dropped += 1.0 - float(row_data.sum()) / budget_total
        if held_index.size:
            row_indices = np.concatenate([held_index, row_indices])
            row_data = np.concatenate([held_data, row_data])
        total = float(row_data.sum())
        if total > 0.0:
            row_data = row_data / total
        kept_indices.append(row_indices)
        kept_data.append(row_data)
        new_indptr[row + 1] = new_indptr[row] + row_data.size

    out = sp.csr_matrix(
        (np.concatenate(kept_data), np.concatenate(kept_indices), new_indptr),
        shape=matrix.shape,
    )
    # argpartition scrambles column order; searchsorted downstream needs it sorted.
    out.sort_indices()
    return out, dropped / max(1, n_rows), capped


def _drop_self_renormalize(dist: sp.csr_matrix, start_ids: np.ndarray) -> sp.csr_matrix:
    """Remove each row's own start node and renormalize what is left."""
    out = dist.copy()
    indptr, indices, data = out.indptr, out.indices, out.data
    for row, start in enumerate(start_ids):
        lo, hi = indptr[row], indptr[row + 1]
        pos = lo + int(np.searchsorted(indices[lo:hi], start))
        if pos < hi and indices[pos] == start:
            data[pos] = 0.0

    row_sums = np.asarray(out.sum(axis=1)).ravel()
    scale = np.where(row_sums > 0.0, 1.0 / np.maximum(row_sums, 1e-300), 0.0)
    out = (sp.diags(scale) @ out).tocsr()
    out.eliminate_zeros()
    out.sort_indices()
    return out


def _diffuse_block(
    start_ids: np.ndarray,
    scales: tuple[int, ...],
    transition: sp.csr_matrix,
    keep_topk: int,
    tolerance: float | None = None,
) -> tuple[list[sp.csr_matrix], dict[int, float], int]:
    """Lazy random walk X <- (X + XP)/2 for a whole block of anchors, snapshotted
    at each scale.

    The lazy walk is what makes the scales a graded family: a plain walk on a mutual
    kNN graph mixes within two steps, so (P)^2 and (P)^4 collapse onto the same
    distribution and the multi-scale objective degenerates into a duplicated term.

    Diffusing one anchor at a time through Python dicts costs ~250k interpreter-level
    operations per anchor; the same recursion expressed as a sparse product over a
    block of anchors is the identical arithmetic (agreement with the dict version is
    at the 1e-17 level) with the inner loops in compiled code.
    """
    n_items = transition.shape[0]
    block = start_ids.size
    dist = sp.csr_matrix(
        (
            np.ones(block, dtype=np.float64),
            (np.arange(block, dtype=np.int64), start_ids.astype(np.int64)),
        ),
        shape=(block, n_items),
    )

    snapshots: dict[int, sp.csr_matrix] = {}
    truncation_loss: dict[int, float] = {}
    capped_total = 0
    for step in range(1, max(scales) + 1):
        dist, dropped, capped = _truncate_renormalize(
            ((dist + dist @ transition) * 0.5).tocsr(),
            keep_topk,
            tolerance,
            # The anchor's own column is dropped by `_drop_self_renormalize` before
            # the snapshot becomes a target, so it is held out of the tolerance
            # budget rather than consuming it.
            protect=start_ids.astype(np.int64),
        )
        truncation_loss[step] = dropped
        capped_total += capped
        if step in scales:
            snapshots[step] = _drop_self_renormalize(dist, start_ids)

    return [snapshots[scale] for scale in scales], truncation_loss, capped_total


def _select_pool(
    supports: list[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    row_cap: int,
    tolerance: float | None = None,
) -> tuple[np.ndarray, bool]:
    """Nodes of the weighted mixture carrying all but `tolerance` of its mass.

    `row_cap` is the memory guard. Returns the selection and whether the guard bound
    before the tolerance was met.
    """
    nodes = np.concatenate([support[0] for support in supports])
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64), False
    mass = np.concatenate(
        [
            float(weights[scale_idx]) * support[1]
            for scale_idx, support in enumerate(supports)
        ]
    )
    unique_nodes, inverse = np.unique(nodes, return_inverse=True)
    mixture = np.bincount(inverse, weights=mass, minlength=unique_nodes.size)

    if tolerance is None:
        order = np.argsort(-mixture, kind="stable")[:row_cap]
        return unique_nodes[order].astype(np.int64), False

    total = float(mixture.sum())
    keep = (
        _mass_prefix(mixture, total, tolerance)
        if total > 0.0
        else np.empty(0, np.int64)
    )
    capped = keep.size > row_cap
    keep = keep[:row_cap]
    return unique_nodes[keep].astype(np.int64), capped


def _gather_masses(
    support: tuple[np.ndarray, np.ndarray], nodes: np.ndarray
) -> np.ndarray:
    """Diffusion mass of `nodes` under one scale; 0 for nodes outside its support."""
    support_indices, support_data = support
    if support_indices.size == 0 or nodes.size == 0:
        return np.zeros(nodes.size, dtype=np.float64)
    pos = np.searchsorted(support_indices, nodes)
    in_range = pos < support_indices.size
    clipped = np.where(in_range, pos, 0)
    hit = in_range & (support_indices[clipped] == nodes)
    return np.where(hit, support_data[clipped], 0.0)


def _build_diffusion_pools(
    top_indices: np.ndarray,
    scales: tuple[int, ...],
    weights: np.ndarray,
    row_neighbors: list[np.ndarray],
    row_probs: list[np.ndarray],
    hard_neg_pool: int,
    source_ids: np.ndarray,
    tolerance: float,
    block_size: int = DIFFUSION_BLOCK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Per-anchor sparse diffusion support plus a same-source hard-negative pool.

    Candidate *sets* are no longer frozen here. The previous design precomputed one
    fixed 64-slot set per anchor and reused it for every epoch, so after epoch 1 the
    student had fit those exact comparisons and the objective had nothing left to
    say. What is precomputed now is the diffusion support; the candidate set is
    resampled from it each epoch by HeatGeoCandidateSampler.

    There is no pool-size argument. How many nodes an anchor keeps is decided by
    `tolerance` alone, per anchor, and the array is allocated afterwards at the
    width the widest anchor actually needed. A configured size could only be one of
    two things: larger than the tolerance requires, which wastes memory and pads
    with -1, or smaller, which silently breaks the guarantee.
    """
    n_items = top_indices.shape[0]
    n_scales = len(scales)

    # Collected per anchor, then packed once the true width is known.
    selected_nodes: list[np.ndarray] = [None] * n_items
    selected_probs: list[np.ndarray] = [None] * n_items
    hard_neg_indices = np.full((n_items, hard_neg_pool), -1, dtype=np.int64)

    residual_mass = np.zeros(n_scales, dtype=np.float64)
    pool_fill = np.zeros(n_items, dtype=np.int64)
    hard_fill = np.zeros(n_items, dtype=np.int64)
    empty_scale = np.zeros(n_scales, dtype=np.int64)

    transition = _transition_matrix(row_neighbors, row_probs, n_items)
    truncation_totals: dict[int, float] = {}
    n_blocks = 0
    # Rows where the allocation ceiling bound before the tolerance was met. For
    # those the TV/KL guarantee on the targets does not hold, so they are counted
    # rather than absorbed.
    pool_capped_rows = 0
    walk_capped_rows = 0

    for block_start in tqdm(
        range(0, n_items, block_size), desc="HeatGeo diffusion pools"
    ):
        block_ids = np.arange(
            block_start, min(block_start + block_size, n_items), dtype=np.int64
        )
        scale_matrices, truncation_loss, walk_capped = _diffuse_block(
            block_ids, scales, transition, DIFFUSION_ROW_CAP, tolerance
        )
        walk_capped_rows += walk_capped
        n_blocks += 1
        for step, dropped in truncation_loss.items():
            truncation_totals[step] = truncation_totals.get(step, 0.0) + dropped

        for row, i in enumerate(block_ids):
            i = int(i)
            supports = [
                (
                    matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]],
                    matrix.data[matrix.indptr[row] : matrix.indptr[row + 1]],
                )
                for matrix in scale_matrices
            ]

            selected, pool_capped = _select_pool(
                supports, weights, POOL_ROW_CAP, tolerance
            )
            pool_capped_rows += int(pool_capped)
            pool_fill[i] = selected.size
            selected_nodes[i] = selected

            row_probs_per_scale = np.zeros((n_scales, selected.size), dtype=np.float32)
            for scale_idx, support in enumerate(supports):
                if support[0].size == 0:
                    empty_scale[scale_idx] += 1
                    continue
                kept = _gather_masses(support, selected)
                kept_sum = float(kept.sum())
                total = float(support[1].sum())
                residual_mass[scale_idx] += max(0.0, total - kept_sum) / max(
                    total, 1e-12
                )
                if kept_sum <= 0.0:
                    empty_scale[scale_idx] += 1
                    continue
                row_probs_per_scale[scale_idx] = (kept / kept_sum).astype(np.float32)
            selected_probs[i] = row_probs_per_scale

            # Hard negatives: nearest teacher neighbours from the SAME source corpus
            # that carry no diffusion mass. Sampling negatives uniformly from a corpus
            # made of three disjoint datasets makes ~2/3 of them separable by domain
            # alone, which is why the discrimination task was exhausted within one
            # epoch.
            candidates = top_indices[i]
            outside_pool = ~np.isin(candidates, selected)
            eligible = outside_pool & (candidates != i)
            hard = candidates[eligible & (source_ids[candidates] == source_ids[i])][
                :hard_neg_pool
            ]
            if hard.size < hard_neg_pool:
                # Same-source pool exhausted: top up with cross-source nearest
                # neighbours rather than leaving the quota unfilled.
                extra = candidates[eligible & ~np.isin(candidates, hard)][
                    : hard_neg_pool - hard.size
                ]
                hard = np.concatenate([hard, extra])
            hard_fill[i] = hard.size
            if hard.size:
                hard_neg_indices[i, : hard.size] = hard.astype(np.int64)

    # Pack at the width the tolerance actually asked for, not at a configured one.
    width = max(1, int(pool_fill.max()))
    pool_indices = np.full((n_items, width), -1, dtype=np.int64)
    pool_probs = np.zeros((n_scales, n_items, width), dtype=np.float32)
    for i in range(n_items):
        nodes = selected_nodes[i]
        if nodes is None or nodes.size == 0:
            continue
        pool_indices[i, : nodes.size] = nodes
        pool_probs[:, i, : nodes.size] = selected_probs[i]

    stats = {
        "pool_width": float(width),
        "pool_fill_avg": float(pool_fill.mean()),
        "pool_fill_min": float(pool_fill.min()),
        "hard_pool_fill_avg": float(hard_fill.mean()),
        "hard_pool_fill_min": float(hard_fill.min()),
        "truncation_tolerance": -1.0 if tolerance is None else float(tolerance),
        # Non-zero means the ceiling, not the tolerance, decided the truncation for
        # that many rows -- raise POOL_ROW_CAP / DIFFUSION_ROW_CAP until both are 0,
        # or the stated guarantee is not the one the targets actually satisfy.
        "pool_capped_rows": float(pool_capped_rows),
        "walk_capped_rows": float(walk_capped_rows),
    }
    # Mass discarded by the top-keep_topk truncation at each walk step, before the
    # renormalization hides it. This is the only number that says whether keep_topk
    # is large enough; pool_residual_mass_r* cannot see it, because it compares the
    # pool against the already-truncated support.
    cumulative = 0.0
    for step in sorted(truncation_totals):
        per_step = truncation_totals[step] / max(1, n_blocks)
        cumulative = 1.0 - (1.0 - cumulative) * (1.0 - per_step)
        stats[f"walk_truncation_step{step}"] = float(per_step)
        if step in scales:
            stats[f"walk_truncation_cum_r{step}"] = float(cumulative)
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

    # Anchors whose sharpest-scale target is (near) one-hot. These sit in tiny
    # connected components: the lazy walk never leaves them, so every scale returns
    # the same point mass, JS_omega is 0, and the loss reduces to "drive this one
    # cosine to 1" at the sharpest temperature against every negative in the batch.
    # That is a memorization signal, not geometry, so it is worth counting.
    degenerate = probs[0].max(axis=-1) > 0.99
    stats["target_degenerate_count"] = float(degenerate.sum())
    stats["target_degenerate_rate"] = float(degenerate.mean())
    stats["target_min_support_r%d" % scales[0]] = float(
        (probs[0] > 0).sum(axis=-1).min()
    )

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
    "perplexity",
    "graph_temp",
    "diffusion_scales",
    "scale_weights",
    "hard_neg_pool",
    "truncation_tolerance",
    "lazy_walk",
    "artifact_version",
    "teacher_fingerprint",
    "source_fingerprint",
)


def _metadata_matches(artifact: dict, metadata: dict) -> tuple[bool, str]:
    old = artifact.get("metadata", {})
    for key in _METADATA_KEYS:
        if old.get(key) != metadata.get(key):
            return (
                False,
                f"{key}: cached={old.get(key)!r} requested={metadata.get(key)!r}",
            )
    return True, ""


def build_or_load_heatgeo_artifact(
    teacher_embeddings: torch.Tensor,
    cache_path: str,
    log_dir: str,
    graph_k: int,
    diffusion_scales: Sequence[int],
    source_ids: Sequence[int] | None = None,
    perplexity: float | None = None,
    truncation_tolerance: float = 0.01,
) -> dict:
    n_items = int(teacher_embeddings.size(0))
    scales = _as_tuple(diffusion_scales)
    # `pool_probs`'s first axis is positional. Reject an implicit reorder so every
    # downstream component derives omega_r = 1/r from the same scale sequence.
    requested = tuple(int(r) for r in diffusion_scales)
    if requested != scales:
        raise ValueError(
            f"diffusion_scales must be sorted and unique, got {requested}: they are "
            f"stored as {scales} and consumed by position"
        )
    if scales[0] != 1:
        raise ValueError(
            f"diffusion_scales must start at 1, got {scales}: the criterion's "
            f"temperature ladder is anchored to the r=1 target being the "
            f"transition row"
        )
    weights = normalized_diffusion_weights(scales)
    hard_neg_pool = hard_negative_pool_size(graph_k)
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
        # The scalar bandwidth is a fixed baseline policy. Canonical entropic
        # affinities ignore it and store one solved temperature per row.
        "perplexity": None if perplexity is None else float(perplexity),
        "graph_temp": FIXED_BANDWIDTH_TEMP,
        "diffusion_scales": scales,
        "scale_weights": tuple(round(float(w), 8) for w in weights),
        "hard_neg_pool": int(hard_neg_pool),
        "truncation_tolerance": float(truncation_tolerance),
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
    # Hard-negative storage is derived from graph width. Retrieval must still cover
    # both the graph pool and that derived hard-negative pool.
    topk_for_graph = min(n_items - 1, max(graph_k, hard_neg_pool + graph_k))
    top_indices, top_scores = _compute_topk_cosine(teacher_embeddings, k=topk_for_graph)
    (
        row_neighbors,
        row_probs,
        row_scores,
        fallback_flags,
        row_temps,
        temp_stats,
    ) = _build_transition(
        top_indices=top_indices,
        top_scores=top_scores,
        graph_k=graph_k,
        graph_temp=FIXED_BANDWIDTH_TEMP,
        perplexity=perplexity,
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
        hard_neg_pool=hard_neg_pool,
        source_ids=source_array,
        tolerance=truncation_tolerance,
    )
    graph_stats.update(temp_stats)
    graph_stats.update(pool_stats)
    graph_stats.update(
        _target_sharpness_stats(pool_indices, pool_probs, scales, weights)
    )

    max_degree = max(len(neighbors) for neighbors in row_neighbors)
    transition_neighbors = np.full((n_items, max_degree), -1, dtype=np.int64)
    transition_probs = np.zeros((n_items, max_degree), dtype=np.float32)
    for row, (neighbors, probs) in enumerate(zip(row_neighbors, row_probs)):
        transition_neighbors[row, : len(neighbors)] = neighbors
        transition_probs[row, : len(probs)] = probs

    artifact = {
        "pool_indices": torch.from_numpy(pool_indices).long(),
        "pool_probs": torch.from_numpy(pool_probs).float(),
        "hard_neg_indices": torch.from_numpy(hard_neg_indices).long(),
        "source_ids": torch.from_numpy(source_array).long(),
        "transition_neighbors": torch.from_numpy(transition_neighbors).long(),
        "transition_probs": torch.from_numpy(transition_probs).float(),
        # The temperature each transition row was built at. The criterion matches
        # row i at row_temps[i]: the tie is per row, and the shift family that
        # makes zero loss attainable does not depend on the value.
        "row_temps": torch.from_numpy(row_temps).float(),
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
            f"{graph_stats['pool_width']:.0f} (min {graph_stats['pool_fill_min']:.0f}), "
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
                f"residual_mass_outside_pool={graph_stats.get(f'pool_residual_mass_r{scale}', 0.0):.2e}, "
                f"walk_truncation={graph_stats.get(f'walk_truncation_cum_r{scale}', 0.0):.2e}"
            )
    worst_truncation = max(
        (graph_stats.get(f"walk_truncation_cum_r{scale}", 0.0) for scale in scales),
        default=0.0,
    )
    tolerance = graph_stats.get("truncation_tolerance", 0.0)
    if worst_truncation > 0.05:
        print(
            f"WARNING: HeatGeo lazy walk drops {worst_truncation:.1%} of the mass at the "
            f"broadest scale before renormalizing, against a per-step tolerance of "
            f"{tolerance:.1%}. The per-step bound compounds across steps, so r*tolerance "
            f"is the figure to compare against; if it is exceeded, DIFFUSION_ROW_CAP bound "
            f"first (walk_capped_rows={int(graph_stats.get('walk_capped_rows', 0))})."
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
    degenerate = int(graph_stats.get("target_degenerate_count", 0))
    if degenerate:
        print(
            f"WARNING: HeatGeo {degenerate} anchors "
            f"({graph_stats.get('target_degenerate_rate', 0.0):.2%}) have a near one-hot "
            f"target at r={scales[0]} (min support "
            f"{int(graph_stats.get(f'target_min_support_r{scales[0]}', 0))}). They sit in "
            f"tiny components, contribute JS_omega=0, and their loss is just "
            f"'push one cosine to 1' at the sharpest temperature -- raise graph_k or "
            f"drop them from training if the count is large."
        )
    low = [s for s in scales if graph_stats.get(f"target_kl_uniform_r{s}", 1.0) < 0.05]
    if low:
        print(
            f"WARNING: HeatGeo targets at scales {low} are close to uniform on their support "
            f"(KL < 0.05 nats) -- lower the target perplexity (or the internal "
            f"fixed-baseline temperature), otherwise L_diff degenerates into a binary "
            f"neighbour/non-neighbour objective."
        )
    if float(graph_stats.get("target_js_floor", 1.0)) < 0.01:
        print(
            "WARNING: JS_omega < 0.01 nats -- the diffusion scales carry almost the same "
            "target, so with a tied student temperature the multi-scale objective is "
            "numerically identical to a single-scale one."
        )
