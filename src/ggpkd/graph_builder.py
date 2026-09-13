import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .policy import (
    FIXED_BANDWIDTH_TEMP,
    TRUNCATION_TOLERANCE,
    hard_negative_pool_size,
)

# 8: restore padded transition rows required by L_row. Version 7 diffusion targets
#    remain mathematically valid, but do not contain the arrays needed to sample and
#    supervise non-anchor rows, so those caches must be rebuilt once.
# 9: DIFFUSION_ROW_CAP raised 4096 -> 16384, forcing r>=2 caches to rebuild.
# 10: multi-hop diffusion removed. The method supervises one hop, so a pool IS a
#    transition row; `diffusion_scales` and `scale_weights` have left the metadata
#    and every cache written under them is rejected.
ARTIFACT_VERSION = 10

# Anchors per block in `_target_sharpness_stats`. Pure memory control: the stats
# are per-anchor reductions, so the block size cannot change any reported number.
ANCHOR_STATS_CHUNK = 4096


# Floor for a degenerate bandwidth. A row whose k retrieved neighbours all carry
# the identical cosine has no scale of its own; it is uniform at every positive
# temperature, so any floor gives the same row and this one only keeps the
# division finite.
MIN_BANDWIDTH = 1e-6


def _knn_bandwidths(top_scores: np.ndarray, graph_k: int) -> np.ndarray:
    """Per-row temperature read straight off the retrieval width.

        tau_i = (s_i(1) - s_i(k)) / log(k)

    where s_i(j) is the j-th largest cosine from i to another node. Reading it back:
    the k-th neighbour sits log(k) nats below the nearest, so it is exactly k times
    less likely -- for every node, with no constant left to choose.

    The log(k) is not decoration. A bare span of 1 nat leaves the row far too flat:
    spread over k neighbours it gives KL(p || uniform) ~ 0.04, right at the
    degeneracy warning this build already emits, against 0.757 measured on the
    production graph under the perplexity solve. Sharpness turns out to be set by
    the span alone and to be almost independent of k (0.774 at k=40 vs 0.755 at
    k=400 for a fixed span), so the span is what has to carry it. log(k) is the
    choice that carries it without adding a second constant, and it lands at
    KL ~ 0.70 for k=200 -- next to the 0.757 the graph already trains at.

    Two properties this is here for.

    *Exactly affine invariant.* Under s -> a*s + b the bandwidth scales as
    tau -> a*tau, so the logits become s_j/tau_i + b/(a*tau_i); the second term
    does not depend on j and a softmax is shift invariant, so the row is
    unchanged. This is the invariance that ruled out a single fixed temperature,
    and it survives here in closed form -- no bisection, no target entropy.

    *A fixed sample size for every node.* The scores come from the raw top-k, read
    before any edge filter, so all k values exist for every node whenever
    graph_k < n_items. The entropic-affinity solve this replaces ran on the
    filtered neighbour list, where degree varies and 18.9% of the production
    corpus had degree at or below the requested perplexity: those rows never
    reached the target entropy at all and were solved against their own ceiling
    instead, yielding a near-uniform target at a very large tau. That failure mode
    cannot occur here, because there is no target to miss.

    Calibrating on the raw top-k rather than on the surviving edges is deliberate.
    It is the wider set, so the gap is larger and the rows come out flatter than a
    degree-matched calibration would give; the trade is that the sample size stops
    depending on how many edges an edge filter happened to leave. Under the
    canonical ``directed`` mode nothing is filtered and the two sets coincide.
    """
    k_eff = min(int(graph_k), top_scores.shape[1])
    if k_eff < 2:
        raise ValueError(f"graph_k must retrieve at least 2 neighbours, got {k_eff}")
    span = top_scores[:, 0].astype(np.float64) - top_scores[:, k_eff - 1].astype(
        np.float64
    )
    return np.maximum(span / np.log(k_eff), MIN_BANDWIDTH)


def heldout_edge_mask(
    rows: np.ndarray, cols: np.ndarray, seed: int, frac: float
) -> np.ndarray:
    """Which teacher edges are withheld from every training support.

    True means "withheld". The split has to satisfy three things at once, and a
    stored list of edges satisfies none of them cheaply:

    *Identical across arms and seeds.* The held-out set is the measuring
    instrument for the whole support study. If it moved with the training seed,
    the arms would be scored on different relations and their held-out numbers
    would not be comparable. It is therefore a pure function of the unordered
    pair and `seed`, with no state and no file to keep in sync.

    *Symmetric.* An edge withheld as (i, j) must also be withheld as (j, i).
    Otherwise L_row supervises from the other endpoint and the "never supervised"
    claim is false for exactly the rows the evaluation reads.

    *Reproducible outside the build.* The evaluation recomputes the mask from the
    teacher cache rather than reading it out of the artifact, so the two cannot
    drift apart across a rebuild.

    The hash is splitmix64's finalizer over the ordered pair, which is uniform
    enough for a 20% split and costs one vectorised pass.
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"holdout fraction must be in [0, 1), got {frac}")
    if frac == 0.0:
        return np.zeros(np.shape(rows), dtype=bool)
    low = np.minimum(rows, cols).astype(np.uint64)
    high = np.maximum(rows, cols).astype(np.uint64)
    # Unordered pair -> one integer, then mixed with the seed.
    key = low * np.uint64(0x9E3779B97F4A7C15) + high
    # The seed term is folded in Python, where the wraparound is explicit. Doing
    # it in numpy scalars is the same arithmetic but raises an overflow warning on
    # every call, and a warning that is expected on the correct path is a warning
    # nobody reads.
    seed_term = np.uint64((int(seed) * 0xBF58476D1CE4E5B9) % (1 << 64))
    key = key ^ seed_term
    key = (key ^ (key >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    key = (key ^ (key >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    key = key ^ (key >> np.uint64(31))
    # Top 53 bits to a double in [0, 1): the same construction numpy uses.
    uniform = (key >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
    return uniform < float(frac)


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
    embeddings: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours, on the accelerator when there is one.

    This is an O(n^2 d) matmul -- ~6e11 FLOPs at n=15k, d=2560 -- and it used to be
    pinned to the CPU on a machine that had just finished running the teacher on a
    GPU, which made a cold graph build take minutes to tens of minutes. Only the
    per-chunk top-k results come back to host memory, so the resident cost is the
    embeddings plus one `chunk_size x n_items` score block.

    TF32 is disabled for the duration: on Ampere and later it would silently drop
    the matmul to ~10 mantissa bits, and these cosines decide graph membership and
    feed the entropic-affinity bandwidths. The remaining CPU/GPU difference is
    reduction order alone, but that is still enough to reorder near-ties in the
    top-k, so an artifact built on one device is not bit-identical to one built on
    the other.
    """
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else embeddings.device
        )

    def _run(target: torch.device) -> tuple[np.ndarray, np.ndarray]:
        normalized = F.normalize(embeddings.float(), p=2, dim=-1).to(target)
        n_items = normalized.size(0)
        k_eff = min(k + 1, n_items)
        all_indices = []
        all_scores = []
        for start in tqdm(range(0, n_items, chunk_size), desc="GGPKD top-k cosine"):
            end = min(start + chunk_size, n_items)
            sims = normalized[start:end] @ normalized.T
            row_ids = torch.arange(start, end, device=target)
            sims[torch.arange(end - start, device=target), row_ids] = -float("inf")
            scores, indices = torch.topk(sims, k=k_eff, dim=-1)
            all_indices.append(indices[:, :k].cpu().numpy().astype(np.int64))
            all_scores.append(scores[:, :k].cpu().numpy().astype(np.float32))
        return (
            np.concatenate(all_indices, axis=0),
            np.concatenate(all_scores, axis=0),
        )

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _run(device)
    except torch.cuda.OutOfMemoryError:
        # n_items x d plus one score block did not fit. Falling back is far better
        # than failing a build that used to work.
        print(f"GGPKD top-k cosine: out of memory on {device}, falling back to CPU")
        torch.cuda.empty_cache()
        return _run(torch.device("cpu"))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


KNN_MODES = ("mutual", "directed", "symmetrized")
# Whose kNN decides a row's columns. `student` is the ladder's student_knn arm.
NEIGHBOR_SOURCES = ("teacher", "student")


def _reverse_adjacency(
    top_indices: np.ndarray, top_scores: np.ndarray, graph_k: int
) -> list[list[tuple[int, float]]]:
    """For each node i, the (j, cos) pairs of every j whose top-k contains i.

    Only the symmetrized arm needs this. Cosine is symmetric, so the score stored
    on j's row for i is exactly the score i's row would carry for j; taking it
    from j's row is what makes the union arm buildable without a second retrieval.
    """
    n_items = top_indices.shape[0]
    reverse: list[list[tuple[int, float]]] = [[] for _ in range(n_items)]
    for j in range(n_items):
        for pos, i in enumerate(top_indices[j, :graph_k]):
            reverse[int(i)].append((j, float(top_scores[j, pos])))
    return reverse


def _build_transition(
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    graph_k: int,
    graph_temp: float,
    fixed_bandwidth: bool,
    knn_mode: str = "directed",
    holdout_edge_frac: float = 0.0,
    holdout_seed: int = 0,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict
]:
    """Neighbour lists and their transition rows, under one of three kNN rules.

    `knn_mode` selects which edges survive retrieval, and nothing else in the
    build changes with it -- same bandwidth rule, same truncation, same pools --
    so the three arms differ only in the edge set:

    * ``directed`` (canonical): keep all of topk(i). Every node has degree
      graph_k, hubs keep every edge pointing at them, and a row is supervised on
      exactly the relations the teacher retrieved for it.
    * ``mutual``: keep j iff j in topk(i) *and* i in topk(j). A hub that
      everything retrieves but that retrieves nothing back loses those edges.
      This was the earlier default; it suppresses hubness but discards teacher
      mass, and E2 measured it 0.36 points below ``directed``.
    * ``symmetrized``: keep the union, j in topk(i) *or* i in topk(j). Degrees are
      the largest of the three and hubs are amplified rather than suppressed.

    Bandwidth is read off the retrieval width by `_knn_bandwidths`, so the graph
    is invariant to the teacher's similarity scale and `graph_k` is the only
    quantity that sets it. With `fixed_bandwidth=True` every row uses `graph_temp`
    instead, which is the baseline that arm exists to be compared against.

    `holdout_edge_frac` withholds a symmetric random subset of the surviving edges
    from the graph entirely, before the rows are normalized. Those relations then
    reach no training support of any arm -- not the candidate draw, not L_row --
    which is what lets the evaluation ask whether a student recovered teacher
    structure it was never shown. The rows renormalize over what is left, so the
    targets stay probability distributions and the only change is which columns
    exist.
    """
    if knn_mode not in KNN_MODES:
        raise ValueError(f"knn_mode must be one of {KNN_MODES}, got {knn_mode!r}")
    n_items = top_indices.shape[0]
    # Only the mutual arm reads this, and building it is n_items sets of graph_k
    # ints. The canonical directed graph filters nothing, so it never looks.
    top_sets = (
        [set(top_indices[i, :graph_k].tolist()) for i in range(n_items)]
        if knn_mode == "mutual"
        else None
    )
    reverse = (
        _reverse_adjacency(top_indices, top_scores, graph_k)
        if knn_mode == "symmetrized"
        else None
    )
    row_neighbors: list[np.ndarray] = []
    row_probs: list[np.ndarray] = []
    row_scores: list[np.ndarray] = []
    fallback_flags = np.zeros(n_items, dtype=bool)
    held_out_edges = 0
    holdout_starved = 0
    # Bandwidths come from the raw top-k, so they are one vectorised subtraction
    # over the whole corpus rather than a bisection per row.
    row_temps = (
        np.full(n_items, float(graph_temp), dtype=np.float64)
        if fixed_bandwidth
        else _knn_bandwidths(top_scores, graph_k)
    )

    for i in tqdm(range(n_items), desc=f"GGPKD {knn_mode} kNN graph"):
        neighbors = []
        scores = []
        seen: set[int] = set()
        for pos, j in enumerate(top_indices[i, :graph_k]):
            j_int = int(j)
            if top_sets is not None and i not in top_sets[j_int]:
                continue
            neighbors.append(j_int)
            scores.append(float(top_scores[i, pos]))
            seen.add(j_int)
        if reverse is not None:
            for j_int, score in reverse[i]:
                if j_int == i or j_int in seen:
                    continue
                neighbors.append(j_int)
                scores.append(score)
                seen.add(j_int)

        if not neighbors:
            fallback_flags[i] = True
            fallback_k = min(graph_k, top_indices.shape[1])
            neighbors = [int(j) for j in top_indices[i, :fallback_k]]
            scores = [float(s) for s in top_scores[i, :fallback_k]]

        if holdout_edge_frac > 0.0 and neighbors:
            # Applied after the fallback, so a row rescued by its raw top-k does
            # not smuggle held-out edges back in through that path.
            neighbor_array = np.asarray(neighbors, dtype=np.int64)
            withheld = heldout_edge_mask(
                np.full(neighbor_array.shape, i, dtype=np.int64),
                neighbor_array,
                holdout_seed,
                holdout_edge_frac,
            )
            kept = ~withheld
            held_out_edges += int(withheld.sum())
            if kept.any():
                neighbors = [int(j) for j in neighbor_array[kept]]
                scores = [float(s) for s, keep in zip(scores, kept) if keep]
            else:
                # Every edge of this row drew into the held-out set. At any
                # sensible fraction this is vanishingly rare, but a row with no
                # columns has no target at all, so it keeps its single nearest
                # neighbour and is counted: the held-out claim is then false for
                # that one edge, and the number has to be visible rather than
                # rounded away.
                holdout_starved += 1
                neighbors = [int(neighbor_array[0])]
                scores = [float(scores[0])]

        # Softmax over neighbour cosines at this row's bandwidth. That value is the
        # one the student has to match this row at, so it is stored alongside the
        # row and the criterion reads it back rather than re-deriving it.
        score_array = np.asarray(scores, dtype=np.float64)
        tau = float(row_temps[i])
        centered = score_array - score_array.max()
        weights = np.exp(centered / tau)
        weights = weights / max(float(weights.sum()), 1e-12)
        row_neighbors.append(np.asarray(neighbors, dtype=np.int64))
        row_probs.append(weights.astype(np.float32))
        row_scores.append(np.asarray(scores, dtype=np.float32))

    temp_stats = {
        "row_temp_mean": float(row_temps.mean()) if n_items else 0.0,
        "row_temp_min": float(row_temps.min()) if n_items else 0.0,
        "row_temp_max": float(row_temps.max()) if n_items else 0.0,
        "row_temp_p50": float(np.median(row_temps)) if n_items else 0.0,
        # Rows whose k retrieved neighbours were tied in cosine and so had no
        # scale of their own. Expected 0; reported because such a row is uniform
        # at every temperature and contributes no gradient.
        "degenerate_bandwidth_rows": int((row_temps <= MIN_BANDWIDTH).sum()),
        # Directed edge slots withheld, and rows that had to keep one anyway.
        "held_out_edges": int(held_out_edges),
        "holdout_starved_rows": int(holdout_starved),
    }
    return row_neighbors, row_probs, row_scores, fallback_flags, row_temps, temp_stats


def _hubness_stats(row_neighbors: list[np.ndarray]) -> dict[str, float]:
    """Indegree concentration of the edge set.

    The kNN-mode comparison is not settled by a downstream score alone: mutual
    kNN is claimed to suppress hubness, and that claim is about the *graph*.
    The method keeps the unfiltered top-k, so this is the statistic that says
    what that choice costs on the graph side. A hub
    is a node that appears in many other nodes' neighbour lists, so the quantity
    is the indegree distribution -- its tail (max, p99) and how much of the total
    edge mass the top 1% of nodes absorb. Reported for every build so the three
    arms are comparable without rebuilding.
    """
    n_items = len(row_neighbors)
    if n_items == 0:
        return {}
    flat = (
        np.concatenate([n for n in row_neighbors if n.size])
        if any(n.size for n in row_neighbors)
        else np.empty(0, dtype=np.int64)
    )
    indegree = np.bincount(flat.astype(np.int64), minlength=n_items)
    total_edges = int(indegree.sum())
    order = np.sort(indegree)[::-1]
    top_count = max(1, int(round(0.01 * n_items)))
    return {
        "indegree_mean": float(indegree.mean()),
        "indegree_max": float(indegree.max()),
        "indegree_p99": float(np.percentile(indegree, 99)),
        "indegree_p50": float(np.percentile(indegree, 50)),
        "indegree_gini": _gini(indegree.astype(np.float64)),
        "hub_edge_share_top1pct": (
            float(order[:top_count].sum() / total_edges) if total_edges else 0.0
        ),
        "isolated_indegree_rate": float((indegree == 0).mean()),
    }


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector; 0 = flat, 1 = one node holds all."""
    if values.size == 0:
        return 0.0
    total = values.sum()
    if total <= 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    rank = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (rank * sorted_values).sum()) / (n * total) - (n + 1.0) / n)


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
        **_hubness_stats(row_neighbors),
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
        "GGPKD kNN graph log saved: "
        f"{log_path} | fallback={fallback_count}/{len(row_neighbors)} "
        f"({stats['fallback_rate']:.2%}), avg_degree={stats['avg_degree']:.2f}"
    )
    if fallback_count:
        fallback_examples = np.flatnonzero(fallback_flags)[:10].tolist()
        print(f"GGPKD fallback node examples: {fallback_examples}")
    return log_path, stats


def _mass_prefix(data: np.ndarray, total: float, tolerance: float) -> np.ndarray:
    """Positions of the smallest set of entries carrying at least 1 - tolerance.

    Truncating a row to a set S with p(S) = 1 - delta and renormalizing gives a
    row ptilde with, exactly,

        TV(p, ptilde) = delta,   KL(ptilde || p) = -log(1 - delta) <= delta/(1-delta),

    so a tolerance on the discarded mass is a bound on the target perturbation in
    nats -- the units of the loss itself. That is what lets one stated tolerance
    replace a capacity knob per truncation site. The bound is per truncation; a
    multi-hop arm truncates once per diffusion step, so after r steps the total is
    at most r times it. The method runs at R={1} with tolerance 0, where nothing
    is discarded and the bound is vacuous.
    """
    order = np.argsort(-data)
    cumulative = np.cumsum(data[order])
    needed = (1.0 - tolerance) * total
    keep = int(np.searchsorted(cumulative, needed) + 1)
    return order[: min(keep, order.size)]


# The "walk" the diffusion_* stats describe is the lazy diffusion walk
# X <- (X + XP)/2 below, not the row-selection walk L_row used to run. They were
# renamed from walk_* so a later cleanup does not mistake one for the other.


def _build_pools(
    row_neighbors: list[np.ndarray],
    row_probs: list[np.ndarray],
    tolerance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Candidate pool of an anchor: its transition row, as it stands.

    There is nothing else it could be. The method supervises one hop, so the
    support of anchor i *is* row i; the multi-hop machinery this replaced built a
    sparse transition matrix and multiplied it only to read back the rows it was
    built from.

    With `tolerance` falsy the row is kept whole and the pool is exactly the
    teacher's top-k. A positive tolerance keeps the smallest prefix carrying
    1 - tolerance of the row's mass, which is the truncation ablation arm.
    """
    n_items = len(row_neighbors)
    selected_nodes: list[np.ndarray] = [None] * n_items
    selected_probs: list[np.ndarray] = [None] * n_items
    pool_fill = np.zeros(n_items, dtype=np.int64)
    residual = 0.0
    empty = 0

    for i in range(n_items):
        nodes = np.asarray(row_neighbors[i], dtype=np.int64)
        probs = np.asarray(row_probs[i], dtype=np.float64)
        total = float(probs.sum())
        if tolerance:
            # Sorted back into the row's own column order: the prefix decides which
            # columns survive, not how they are laid out.
            keep = np.sort(_mass_prefix(probs, total, float(tolerance)))
            nodes, probs = nodes[keep], probs[keep]
            kept = float(probs.sum())
            residual += max(0.0, total - kept) / max(total, 1e-12)
            total = kept
        if nodes.size == 0 or total <= 0.0:
            empty += 1
            continue
        selected_nodes[i] = nodes
        selected_probs[i] = (probs / total).astype(np.float32)
        pool_fill[i] = nodes.size

    width = max(1, int(pool_fill.max()))
    pool_indices = np.full((n_items, width), -1, dtype=np.int64)
    pool_probs = np.zeros((1, n_items, width), dtype=np.float32)
    for i in range(n_items):
        nodes = selected_nodes[i]
        if nodes is None or nodes.size == 0:
            continue
        pool_indices[i, : nodes.size] = nodes
        pool_probs[0, i, : nodes.size] = selected_probs[i]

    stats = {
        "pool_width": float(width),
        "pool_fill_avg": float(pool_fill.mean()),
        "pool_fill_min": float(pool_fill.min()),
        "hard_pool_fill_avg": 0.0,
        "hard_pool_fill_min": 0.0,
        "truncation_tolerance": 0.0 if not tolerance else float(tolerance),
        "pool_capped_rows": 0.0,
        "diffusion_capped_rows": 0.0,
        "pool_residual_mass_r1": float(residual / max(1, n_items)),
        "pool_empty_r1": float(empty / max(1, n_items)),
    }
    return pool_indices, pool_probs, np.full((n_items, 0), -1, dtype=np.int64), stats


def _target_sharpness_stats(
    pool_indices: np.ndarray,
    pool_probs: np.ndarray,
) -> dict[str, float]:
    """Is the target informative?

    KL(p || uniform-on-support) near zero means the target has degenerated into a
    binary neighbour/non-neighbour label and carries no ranking signal. That is
    the failure `graph_k` can walk into: the bandwidth is the retrieval span, so a
    wide enough k measures the distance out of the neighbourhood rather than the
    local decay and flattens every row.
    """
    stats: dict[str, float] = {}
    _, n_items, _ = pool_probs.shape

    # Reductions along the candidate axis, walked in blocks: a float64 copy of the
    # whole pool_probs costs hundreds of megabytes at corpus scale (>2 GB peak
    # measured) purely for diagnostics. The per-anchor vectors kept here are
    # n_items floats each and the final reductions run over them in full, so the
    # reported numbers are unchanged rather than merely close.
    supp_size = np.zeros(n_items, dtype=np.float64)
    entropies = np.zeros(n_items, dtype=np.float64)
    top1 = np.zeros(n_items, dtype=np.float64)

    for start in range(0, n_items, ANCHOR_STATS_CHUNK):
        block = slice(start, min(start + ANCHOR_STATS_CHUNK, n_items))
        valid = pool_indices[block] >= 0
        p = np.clip(pool_probs[0, block, :].astype(np.float64), 0.0, None)
        p = np.where(valid, p, 0.0)
        mask = p > 0
        supp_size[block] = mask.sum(axis=-1)
        safe = np.where(mask, p, 1.0)
        entropies[block] = -(np.where(mask, p * np.log(safe), 0.0)).sum(axis=-1)
        top1[block] = p.max(axis=-1)

    kl_uniform = np.log(np.maximum(supp_size, 1.0)) - entropies
    stats["target_support_r1"] = float(supp_size.mean())
    stats["target_kl_uniform_r1"] = float(kl_uniform.mean())
    stats["target_top1_r1"] = float(top1.mean())
    stats["target_min_support_r1"] = float(supp_size.min())

    # Anchors whose target is (near) one-hot. They sit in tiny neighbourhoods, and
    # their loss reduces to "drive this one cosine to 1" against every other column
    # in the batch -- a memorization signal rather than geometry, worth counting.
    degenerate = top1 > 0.99
    stats["target_degenerate_count"] = float(degenerate.sum())
    stats["target_degenerate_rate"] = float(degenerate.mean())
    return stats


_METADATA_KEYS = (
    "n_items",
    "graph_k",
    "bandwidth",
    "graph_temp",
    "hard_neg_pool",
    "truncation_tolerance",
    "knn_mode",
    "holdout_edge_frac",
    "holdout_seed",
    "artifact_version",
    "teacher_fingerprint",
    "source_fingerprint",
    "neighbor_source",
)


# Keys introduced after an artifact version was already in use. A cache written
# before the key existed was built at this value, so reading it as the default
# keeps those caches valid instead of forcing a rebuild that would change nothing.
_METADATA_DEFAULTS = {
    "knn_mode": "mutual",
    "holdout_edge_frac": 0.0,
    "holdout_seed": 0,
    "neighbor_source": "teacher",
}


def _metadata_matches(artifact: dict, metadata: dict) -> tuple[bool, str]:
    old = artifact.get("metadata", {})
    for key in _METADATA_KEYS:
        default = _METADATA_DEFAULTS.get(key)
        if old.get(key, default) != metadata.get(key):
            return (
                False,
                f"{key}: cached={old.get(key, default)!r} "
                f"requested={metadata.get(key)!r}",
            )
    return True, ""


def build_or_load_ggpkd_artifact(
    teacher_embeddings: torch.Tensor,
    cache_path: str,
    log_dir: str,
    graph_k: int,
    source_ids: Sequence[int] | None = None,
    fixed_bandwidth: bool = False,
    truncation_tolerance: float = TRUNCATION_TOLERANCE,
    hard_negatives: bool = False,
    knn_mode: str = "directed",
    holdout_edge_frac: float = 0.0,
    holdout_seed: int = 0,
    neighbor_source: str = "teacher",
    neighbor_embeddings: Callable[[], torch.Tensor] | None = None,
) -> dict:
    """Build the transition graph, or load it when every metadata key matches.

    `neighbor_source` other than "teacher" is a label for another encoder whose
    kNN decides each row's columns; `neighbor_embeddings` produces that encoder's
    corpus embeddings and is only called on a build. The row temperatures stay
    the teacher's own bandwidths, so the neighbour set is the only thing that
    differs from the teacher graph. The label is part of the cache key: an
    artifact built from another encoder must never load under the teacher's name.
    """
    n_items = int(teacher_embeddings.size(0))
    # Sized from graph_k only when an arm actually draws hard negatives. The
    # method draws none, and an unconditional pool cost every build a top-2k
    # retrieval plus an (N, graph_k) table nothing read.
    hard_neg_pool = hard_negative_pool_size(graph_k) if hard_negatives else 0
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
        # The scalar bandwidth is a fixed baseline policy. Canonical rows ignore it
        # and store one temperature per row, derived from graph_k.
        "bandwidth": "fixed" if fixed_bandwidth else "knn",
        "graph_temp": FIXED_BANDWIDTH_TEMP,
        "hard_neg_pool": int(hard_neg_pool),
        "truncation_tolerance": float(truncation_tolerance),
        # In _METADATA_KEYS below: two arms of the kNN ablation share every other
        # key, so without it the second arm would silently load the first's graph.
        "knn_mode": str(knn_mode),
        # Same reason, and the stakes are higher: a held-out family that loaded a
        # full graph would train on the very edges its evaluation calls unseen.
        # `holdout_seed` is only recorded when a holdout is actually taken, so
        # turning it off does not invalidate a cache built without one.
        "holdout_edge_frac": float(holdout_edge_frac),
        "holdout_seed": int(holdout_seed) if holdout_edge_frac > 0.0 else 0,
        "artifact_version": ARTIFACT_VERSION,
        "teacher_fingerprint": _fingerprint(teacher_embeddings),
        "source_fingerprint": hashlib.sha1(source_array.tobytes()).hexdigest(),
        "neighbor_source": str(neighbor_source),
    }

    artifact_path = Path(cache_path)
    if artifact_path.exists():
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        matches, reason = _metadata_matches(artifact, metadata)
        if matches:
            # graph_stats are embedded in the artifact. The verbose neighbour
            # JSONL is disposable after a successful paper run and must not make
            # an otherwise valid cache rebuild itself merely because that log was
            # compacted away.
            print(f"Loaded GGPKD artifact from: {artifact_path}")
            _print_graph_summary(artifact.get("graph_stats", {}))
            return artifact
        else:
            print(f"GGPKD artifact config mismatch, rebuilding: {artifact_path}")
            print(f"  first mismatch -> {reason}")

    if str(artifact_path.parent):
        os.makedirs(artifact_path.parent, exist_ok=True)
    # Hard-negative storage is derived from graph width. Retrieval must still cover
    # both the graph pool and that derived hard-negative pool.
    topk_for_graph = min(n_items - 1, max(graph_k, hard_neg_pool + graph_k))
    topk_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    teacher_top_scores = None
    if neighbor_source == "teacher":
        top_indices, top_scores = _compute_topk_cosine(
            teacher_embeddings, k=topk_for_graph, device=topk_device
        )
    else:
        if neighbor_embeddings is None:
            raise ValueError(
                f"neighbor_source={neighbor_source!r} needs neighbor_embeddings to "
                "build the graph from"
            )
        neighbor_matrix = neighbor_embeddings()
        if int(neighbor_matrix.size(0)) != n_items:
            raise ValueError(
                f"neighbor embeddings have {int(neighbor_matrix.size(0))} rows but "
                f"there are {n_items} teacher embeddings"
            )
        # Columns and their order come from the other encoder; the teacher's
        # top-k is read only for its bandwidths.
        top_indices, top_scores = _compute_topk_cosine(
            neighbor_matrix, k=topk_for_graph, device=topk_device
        )
        _, teacher_top_scores = _compute_topk_cosine(
            teacher_embeddings, k=graph_k, device=topk_device
        )
        metadata["neighbor_fingerprint"] = _fingerprint(neighbor_matrix)
    # Recorded but deliberately NOT in _METADATA_KEYS. The teacher fingerprint
    # cannot see this: the same embeddings top-k'd on CPU and on GPU give the same
    # hash but slightly different neighbour lists, because the two reduction orders
    # break near-ties differently. Validating on it would invalidate every existing
    # cache; storing it means a graph whose numbers look odd can at least be traced
    # to the device that built it.
    metadata["topk_device"] = topk_device.type
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
        fixed_bandwidth=fixed_bandwidth,
        knn_mode=knn_mode,
        holdout_edge_frac=holdout_edge_frac,
        holdout_seed=holdout_seed,
    )
    if teacher_top_scores is not None and not fixed_bandwidth:
        # The rows above were ranked and softmaxed by the other encoder, which is
        # what the candidate draw selects on. The temperature the criterion reads
        # back is the teacher's, exactly as in the teacher graph.
        neighbor_temps = row_temps
        row_temps = _knn_bandwidths(teacher_top_scores, graph_k)
        temp_stats.update(
            {
                "row_temp_mean": float(row_temps.mean()),
                "row_temp_min": float(row_temps.min()),
                "row_temp_max": float(row_temps.max()),
                "row_temp_p50": float(np.median(row_temps)),
                "degenerate_bandwidth_rows": int((row_temps <= MIN_BANDWIDTH).sum()),
                "neighbor_row_temp_p50": float(np.median(neighbor_temps)),
            }
        )
    graph_log_path, graph_stats = _write_knn_graph_log(
        log_dir=log_dir,
        row_neighbors=row_neighbors,
        row_probs=row_probs,
        row_scores=row_scores,
        fallback_flags=fallback_flags,
        graph_k=graph_k,
    )
    pool_indices, pool_probs, hard_neg_indices, pool_stats = _build_pools(
        row_neighbors=row_neighbors,
        row_probs=row_probs,
        tolerance=truncation_tolerance,
    )
    graph_stats.update(temp_stats)
    graph_stats.update(pool_stats)
    graph_stats.update(_target_sharpness_stats(pool_indices, pool_probs))

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
    print(f"Saved GGPKD artifact to: {artifact_path}")
    _print_graph_summary(graph_stats)
    return artifact


def _print_graph_summary(graph_stats: dict[str, float]) -> None:
    if not graph_stats:
        return
    print(
        "GGPKD kNN graph summary: "
        f"fallback={graph_stats.get('fallback_count', 0)}/{graph_stats.get('n_items', 0)} "
        f"({float(graph_stats.get('fallback_rate', 0.0)):.2%}), "
        f"avg_degree={float(graph_stats.get('avg_degree', 0.0)):.2f}"
    )
    if "pool_fill_avg" in graph_stats:
        print(
            "GGPKD pools: "
            f"fill={graph_stats['pool_fill_avg']:.1f}/"
            f"{graph_stats['pool_width']:.0f} (min {graph_stats['pool_fill_min']:.0f}), "
            f"hard_neg_fill={graph_stats['hard_pool_fill_avg']:.1f} "
            f"(min {graph_stats['hard_pool_fill_min']:.0f})"
        )
    if "target_kl_uniform_r1" in graph_stats:
        print(
            f"GGPKD target: support={graph_stats['target_support_r1']:.1f}, "
            f"KL(p||uniform_on_support)={graph_stats['target_kl_uniform_r1']:.4f}, "
            f"top1={graph_stats['target_top1_r1']:.4f}, "
            f"residual_mass_outside_pool="
            f"{graph_stats.get('pool_residual_mass_r1', 0.0):.2e}"
        )
    degenerate = int(graph_stats.get("target_degenerate_count", 0))
    if degenerate:
        print(
            f"WARNING: GGPKD {degenerate} anchors "
            f"({graph_stats.get('target_degenerate_rate', 0.0):.2%}) have a near one-hot "
            f"target (min support "
            f"{int(graph_stats.get('target_min_support_r1', 0))}). They sit in "
            f"tiny neighbourhoods, and their loss is just 'push one cosine to 1' "
            f"-- raise graph_k or drop them from training if the count is large."
        )
    if graph_stats.get("target_kl_uniform_r1", 1.0) < 0.05:
        print(
            "WARNING: GGPKD targets are close to uniform on their support "
            "(KL < 0.05 nats) -- lower graph_k, otherwise the graph scale "
            "degenerates into a binary neighbour/non-neighbour objective."
        )
