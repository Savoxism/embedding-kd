"""Measure canonical non-backtracking walk coverage before training.

The walk term supervises the teacher's transition row at every node a walk visits,
weighted by visit count. What a walk buys is therefore measured in distinct rows per
unit of walk budget, and that quantity is fixed by the graph alone -- no student,
loss, or training run is needed. Non-backtracking is a fixed RIPPLE policy.

Reported per arm:

    distinct_per_walk   mean distinct nodes visited by one walk (of walk_length steps)
    revisit_rate        fraction of visits landing on a node the same walk already saw
    immediate_return    fraction of steps that go straight back where they came from
    hops_reached        mean BFS distance from the anchor at the final step
    distinct_per_anchor mean distinct nodes over all num_walks walks of one anchor

`distinct_per_anchor` is the one that feeds `walk_rows` and `walk_node_hit_ratio`
during training: those count the union over an anchor's walks, intersected with the
shared pool. `hops_reached` is the structural claim -- backtracking keeps a walk
pinned near its start, and the diffusion scales already supervise that neighbourhood.

Usage:
    python scripts/walk_coverage_probe.py cache/heatgeo/<artifact>.pt
    python scripts/walk_coverage_probe.py --synthetic          # no artifact needed
    python scripts/walk_coverage_probe.py <artifact>.pt --walk-length 8 --anchors 2000
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.heatgeo.candidate_sampler import HeatGeoCandidateSampler
from src.heatgeo.policy import FIXED_BANDWIDTH_TEMP


def _synthetic_artifact(
    n_items: int, n_clusters: int, dim: int, graph_k: int, seed: int
) -> dict:
    """A clustered mutual-kNN graph, built the way graph_builder builds the real one.

    Cluster structure matters: on a graph that is locally near-complete, backtracking
    costs little because every neighbour is one hop from every other. Teacher
    embeddings are not like that, and neither is this.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim))
    assignments = rng.integers(0, n_clusters, size=n_items)
    embeddings = centers[assignments] + 0.6 * rng.normal(size=(n_items, dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    top = np.argsort(-similarity, axis=1)[:, :graph_k]
    top_sets = [set(row.tolist()) for row in top]

    rows = []
    for i in range(n_items):
        # Mutual kNN, with the same fallback the builder uses for isolated nodes.
        neighbors = [int(j) for j in top[i] if i in top_sets[int(j)]]
        if not neighbors:
            neighbors = [int(j) for j in top[i]]
        scores = similarity[i, neighbors]
        weights = np.exp((scores - scores.max()) / FIXED_BANDWIDTH_TEMP)
        rows.append((neighbors, weights / weights.sum()))

    max_degree = max(len(neighbors) for neighbors, _ in rows)
    transition_neighbors = np.full((n_items, max_degree), -1, dtype=np.int64)
    transition_probs = np.zeros((n_items, max_degree), dtype=np.float32)
    for i, (neighbors, weights) in enumerate(rows):
        transition_neighbors[i, : len(neighbors)] = neighbors
        transition_probs[i, : len(neighbors)] = weights

    return {
        "pool_indices": torch.from_numpy(transition_neighbors),
        "pool_probs": torch.from_numpy(transition_probs).unsqueeze(0),
        "hard_neg_indices": torch.full((n_items, 1), -1, dtype=torch.long),
        "transition_neighbors": torch.from_numpy(transition_neighbors),
        "transition_probs": torch.from_numpy(transition_probs),
        "metadata": {"diffusion_scales": (1,)},
    }


def _hop_distances(
    neighbors: np.ndarray, source: int, targets: set[int], max_hops: int
) -> dict[int, int]:
    """BFS from `source`, stopping once every target is reached or `max_hops` is up."""
    seen = {source: 0}
    frontier = deque([source])
    remaining = set(targets) - {source}
    while frontier and remaining:
        node = frontier.popleft()
        if seen[node] >= max_hops:
            continue
        for neighbor in neighbors[node]:
            neighbor = int(neighbor)
            if neighbor < 0 or neighbor in seen:
                continue
            seen[neighbor] = seen[node] + 1
            remaining.discard(neighbor)
            frontier.append(neighbor)
    return seen


def _probe(
    artifact: dict,
    anchors: np.ndarray,
    num_walks: int,
    walk_length: int,
    seed: int,
    measure_hops: bool,
) -> dict[str, float]:
    sampler = HeatGeoCandidateSampler(
        artifact=artifact,
        diffusion_quota=1,
        hard_neg_k=0,
        random_neg_k=1,
        seed=seed,
        num_walks=num_walks,
        walk_length=walk_length,
    )
    neighbors = sampler.trans_neighbors

    distinct_per_walk = []
    distinct_per_anchor = []
    revisits = 0
    visits = 0
    returns = 0
    return_chances = 0
    hops = []

    for anchor in anchors:
        anchor = int(anchor)
        walks, node_set = sampler._sample_walks(anchor, sampler._rng(anchor))
        distinct_per_anchor.append(len(node_set))
        for path in walks:
            steps = path[1:]
            distinct_per_walk.append(len(set(steps.tolist())))
            seen = {int(path[0])}
            for step in steps.tolist():
                visits += 1
                if step in seen:
                    revisits += 1
                seen.add(step)
            # An immediate return needs a step before it to return to.
            return_chances += max(steps.size - 1, 0)
            returns += int((path[2:] == path[:-2]).sum())
        if measure_hops:
            endpoints = {int(path[-1]) for path in walks}
            distances = _hop_distances(
                neighbors, anchor, endpoints, max_hops=walk_length
            )
            hops.extend(distances.get(endpoint, walk_length) for endpoint in endpoints)

    return {
        "distinct_per_walk": float(np.mean(distinct_per_walk)),
        "revisit_rate": revisits / max(visits, 1),
        "immediate_return": returns / max(return_chances, 1),
        "hops_reached": float(np.mean(hops)) if hops else float("nan"),
        "distinct_per_anchor": float(np.mean(distinct_per_anchor)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", nargs="?", default=None, help="path to a HeatGeo artifact .pt"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="probe a generated clustered kNN graph instead of an artifact",
    )
    parser.add_argument("--num-walks", type=int, default=4)
    parser.add_argument("--walk-length", type=int, default=4)
    parser.add_argument("--anchors", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-hops", action="store_true", help="skip the BFS distances")
    parser.add_argument("--synthetic-items", type=int, default=4000)
    parser.add_argument("--synthetic-clusters", type=int, default=40)
    parser.add_argument("--synthetic-dim", type=int, default=64)
    parser.add_argument("--graph-k", type=int, default=200)
    args = parser.parse_args()

    if args.artifact:
        artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
        if "transition_neighbors" not in artifact:
            print(
                f"{args.artifact} has no transition rows; rebuild the graph cache.",
                file=sys.stderr,
            )
            return 1
        source = args.artifact
    elif args.synthetic:
        artifact = _synthetic_artifact(
            n_items=args.synthetic_items,
            n_clusters=args.synthetic_clusters,
            dim=args.synthetic_dim,
            graph_k=args.graph_k,
            seed=args.seed,
        )
        source = (
            f"synthetic: {args.synthetic_items} items, {args.synthetic_clusters} "
            f"clusters, graph_k={args.graph_k}, fixed_temp={FIXED_BANDWIDTH_TEMP}"
        )
    else:
        parser.error("pass an artifact path or --synthetic")

    n_items = int(artifact["transition_neighbors"].shape[0])
    rng = np.random.default_rng(args.seed)
    anchors = rng.choice(n_items, size=min(args.anchors, n_items), replace=False)

    degrees = (artifact["transition_neighbors"].numpy() >= 0).sum(axis=1)
    print(f"graph: {source}")
    print(
        f"  n_items={n_items}  degree: mean={degrees.mean():.1f} "
        f"min={degrees.min()} max={degrees.max()}  "
        f"deg1_nodes={int((degrees == 1).sum())}"
    )
    print(
        f"  walks: num_walks={args.num_walks} walk_length={args.walk_length} "
        f"anchors={len(anchors)}"
    )

    metrics = _probe(
        artifact=artifact,
        anchors=anchors,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        seed=args.seed,
        measure_hops=not args.no_hops,
    )

    keys = [
        "distinct_per_walk",
        "distinct_per_anchor",
        "revisit_rate",
        "immediate_return",
        "hops_reached",
    ]
    width = max(len(key) for key in keys)
    print(f"\n  {'metric':<{width}}  {'value':>10}")
    for key in keys:
        print(f"  {key:<{width}}  {metrics[key]:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
