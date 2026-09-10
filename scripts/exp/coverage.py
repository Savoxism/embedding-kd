#!/usr/bin/env python3
"""Teacher-relevant relation exposure, per support rule. No training required.

This is the measurement behind the study's first claim: that a mini-batch decides
which relations get supervised, and decides it without reference to teacher
relevance. It reads the teacher embedding cache and the graph artifact, and writes
one tidy CSV.

Three quantities per support rule, for anchors sampled from the corpus:

    TRE@K   teacher-relevant exposure. Of the anchor's teacher top-K neighbours,
            the fraction the rule ever exposes. For batch rules this is the
            cumulative union over `--epochs` independent reshuffles, because the
            question is what the student was ever shown, not what one step showed.
    C       teacher mass coverage, sum of p_T(j | i) over the exposed columns.
            Recall counts neighbours; this weights them by how much the teacher
            cares, so exposing rank 1 is not scored the same as exposing rank 900.
    |S|     support size, so no row of the table can be read without its budget.

Two decisions in here are what keep the numbers from being circular, and both are
worth stating because getting either wrong produces a flattering result:

*The evaluation universe is not the training graph.* Neighbours come from a fresh
teacher top-K over the cached embeddings (`--eval_k`, default 1000), computed
independently of `graph_k`, of the mutual filter and of the truncation tolerance.
Scoring the method against its own graph rows would report ~0.99 by construction:
the artifact keeps each row to within 1% of its mass, so "coverage of the columns
we selected, measured over the columns we selected" is a tautology.

*The teacher distribution is at a fixed probe temperature*, not at the run's own
per-row tau_i. tau_i is derived from graph_k, so weighting by it would make
coverage a function of the very graph under evaluation. The constant reused here
is `PROBE_DISTORTION_TEMP` from src/distill/geometry.py, for the same reason it
exists there: a quantity compared across arms cannot have its weighting move with
the arm.

For the batch rules there is also a closed form worth checking the empirical
number against. Under random batching a specific neighbour co-occurs with a given
anchor in one epoch with probability (B-1)/(N-1), so after E independent
reshuffles

    P(ever exposed) = 1 - (1 - (B-1)/(N-1))^E

which the CSV carries as `analytic_exposure`. It is exact for `batch_random` and
is what makes the scaling grid in exp4 computable at corpus sizes no training run
in this repo reaches.

Usage:
    python scripts/exp/coverage.py \
        --cache cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/teacher_train.pt \
        --artifact cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/graph.pt \
        --out runs/exp1/coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.distill.geometry import PROBE_DISTORTION_TEMP  # noqa: E402
from src.ggpkd.graph_builder import heldout_edge_mask  # noqa: E402

# Where the exposure curve is read. K=10 is the neighbourhood a retrieval metric
# would look at; K=100 and K=1000 ask whether a rule reaches past the immediate
# neighbours into the broader structure the paper claims to preserve.
RECALL_KS = (10, 100, 1000)


def load_teacher(cache_path: Path) -> torch.Tensor:
    """The cached teacher matrix, however the cache chose to wrap it."""
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(blob, torch.Tensor):
        embeddings = blob
    elif isinstance(blob, dict):
        for key in ("teacher_cls", "embeddings", "teacher", "cls"):
            if key in blob and torch.is_tensor(blob[key]):
                embeddings = blob[key]
                break
        else:
            tensors = [v for v in blob.values() if torch.is_tensor(v) and v.dim() == 2]
            if len(tensors) != 1:
                raise ValueError(
                    f"cannot identify the teacher matrix in {cache_path}: "
                    f"keys={list(blob)}"
                )
            embeddings = tensors[0]
    else:
        raise ValueError(f"unsupported teacher cache object: {type(blob)}")
    return embeddings.float()


def teacher_universe(
    embeddings: torch.Tensor, eval_k: int, chunk: int = 512, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher top-`eval_k` neighbours and their probe-temperature masses.

    The masses are a softmax over the retrieved cosines at the fixed probe
    temperature, renormalized over the retrieved set. Truncating at eval_k and
    renormalizing is the same operation the artifact performs on its rows, so the
    two are comparable; at eval_k=1000 on a 13.5k corpus the discarded tail is far
    below the level any coverage difference in the table turns on.
    """
    normalized = F.normalize(embeddings, p=2, dim=-1).to(device)
    n_items = normalized.size(0)
    k = min(int(eval_k), n_items - 1)
    indices = np.zeros((n_items, k), dtype=np.int32)
    masses = np.zeros((n_items, k), dtype=np.float32)

    for start in range(0, n_items, chunk):
        stop = min(start + chunk, n_items)
        block = normalized[start:stop] @ normalized.t()
        rows = torch.arange(start, stop, device=block.device)
        block[rows - start, rows] = float("-inf")
        scores, neighbors = block.topk(k, dim=-1)
        probabilities = F.softmax(scores / PROBE_DISTORTION_TEMP, dim=-1)
        indices[start:stop] = neighbors.cpu().numpy().astype(np.int32)
        masses[start:stop] = probabilities.cpu().numpy().astype(np.float32)
    return indices, masses


def _exposure_row(
    exposed: set[int],
    neighbors: np.ndarray,
    masses: np.ndarray,
) -> dict[str, float]:
    """Recall at each K and total teacher mass, for one anchor's exposed set."""
    result: dict[str, float] = {}
    for k in RECALL_KS:
        k_eff = min(k, neighbors.size)
        top = neighbors[:k_eff]
        hits = sum(1 for j in top if int(j) in exposed)
        result[f"recall_at_{k}"] = hits / k_eff
    mass = sum(
        float(m) for j, m in zip(neighbors, masses) if int(j) in exposed
    )
    result["teacher_mass_coverage"] = mass
    result["support_size"] = float(len(exposed))
    return result


def batch_exposure(
    anchors: np.ndarray,
    n_items: int,
    batch_size: int,
    epochs: int,
    rng: np.random.Generator,
    grouping: str,
    neighbors: np.ndarray,
) -> dict[int, set[int]]:
    """Which columns each anchor ever co-occurs with, over `epochs` reshuffles.

    `grouping` selects the batch composition being modelled -- the same three the
    training study uses. `teacher_neighbor` and `teacher_diverse` are modelled
    here rather than imported from the sampler because this script has no
    DataLoader: the construction is the same greedy neighbourhood partition, and
    the numbers it produces are the ones the trained arms are compared against.
    """
    tracked = {int(a): set() for a in anchors}
    for epoch in range(epochs):
        if grouping == "random":
            order = rng.permutation(n_items)
            batches = [
                order[start : start + batch_size]
                for start in range(0, n_items, batch_size)
            ]
        else:
            batches = _teacher_batches(neighbors, batch_size, rng, grouping)
        for batch in batches:
            members = [int(index) for index in batch]
            member_set = set(members)
            for anchor in members:
                if anchor in tracked:
                    tracked[anchor].update(member_set - {anchor})
    return tracked


def _teacher_batches(
    neighbors: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    grouping: str,
) -> list[np.ndarray]:
    from src.data_utils.batch_samplers import _diverse_batches, _neighbor_batches

    builder = _neighbor_batches if grouping == "teacher_neighbor" else _diverse_batches
    return builder(neighbors[:, : min(200, neighbors.shape[1])], batch_size, rng)


def artifact_support(
    artifact: dict, anchors: np.ndarray, quota: int | None
) -> dict[int, set[int]]:
    """The method's own support: the anchor's truncated transition row.

    With `quota` set, the deterministic teacher top-`quota` prefix of that row --
    the fixed-budget teacher arm the random-support arms are matched against.
    """
    pool = artifact["pool_indices"].numpy()
    probabilities = artifact["pool_probs"].numpy().sum(axis=0)
    exposed = {}
    for anchor in anchors:
        anchor = int(anchor)
        row = pool[anchor]
        valid = row >= 0
        columns = row[valid]
        if quota is not None:
            order = np.argsort(-probabilities[anchor][valid])
            columns = columns[order[:quota]]
        exposed[anchor] = set(int(j) for j in columns)
    return exposed


def random_support(
    anchors: np.ndarray, n_items: int, quota: int, rng: np.random.Generator
) -> dict[int, set[int]]:
    exposed = {}
    for anchor in anchors:
        anchor = int(anchor)
        draw = rng.choice(n_items - 1, size=min(quota, n_items - 1), replace=False)
        draw = draw + (draw >= anchor)
        exposed[anchor] = set(int(j) for j in draw)
    return exposed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="teacher embedding cache (.pt)")
    parser.add_argument("--artifact", default=None, help="GGPKD graph artifact (.pt)")
    parser.add_argument("--out", required=True, help="CSV to write")
    parser.add_argument("--pair", default="", help="label for the CSV's pair column")
    parser.add_argument("--eval-k", type=int, default=1000)
    parser.add_argument("--anchors", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--batch-sizes",
        default="16,32,64,128,256",
        help="comma-separated batch sizes for the in-batch rules",
    )
    parser.add_argument(
        "--quota",
        type=int,
        default=None,
        help="fixed support budget for the matched arms; default is the median "
        "transition-row width, which is the width the method actually uses",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.0,
        help="exclude these teacher edges from the evaluation universe, matching "
        "a held-out training family so coverage and geometry are read on the "
        "same relations",
    )
    parser.add_argument("--holdout-seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    embeddings = load_teacher(Path(args.cache))
    n_items = int(embeddings.size(0))
    print(f"teacher cache: {n_items} x {int(embeddings.size(1))}")

    neighbors, masses = teacher_universe(
        embeddings, args.eval_k, device=args.device
    )
    if args.holdout_frac > 0:
        rows = np.repeat(
            np.arange(n_items, dtype=np.int64)[:, None], neighbors.shape[1], axis=1
        )
        keep = ~heldout_edge_mask(
            rows, neighbors.astype(np.int64), args.holdout_seed, args.holdout_frac
        )
        # Held-out relations are not part of what any arm could expose, so they
        # must leave the universe too; otherwise every rule is charged for
        # missing columns that were withheld from all of them.
        masses = masses * keep
        neighbors = np.where(keep, neighbors, -1)
        print(f"holdout: {1 - float(keep.mean()):.1%} of universe edges removed")

    rng = np.random.default_rng(args.seed)
    anchors = rng.choice(n_items, size=min(args.anchors, n_items), replace=False)
    anchors.sort()

    artifact = None
    quota = args.quota
    if args.artifact:
        artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
        widths = (artifact["pool_indices"].numpy() >= 0).sum(axis=1)
        if quota is None:
            quota = int(np.median(widths))
        print(
            f"graph artifact: row width mean={widths.mean():.1f} "
            f"median={int(np.median(widths))} max={int(widths.max())}; quota={quota}"
        )
    if quota is None:
        quota = 64

    rules: list[tuple[str, str, int, dict[int, set[int]]]] = []
    for grouping in ("random", "teacher_neighbor", "teacher_diverse"):
        for batch_size in [int(b) for b in args.batch_sizes.split(",") if b]:
            if grouping != "random" and batch_size not in (64,):
                # The teacher-informed groupings are the batch-intervention arms
                # and are only trained at the study's batch size; sweeping them
                # here would fill the CSV with rows no run corresponds to.
                continue
            exposed = batch_exposure(
                anchors,
                n_items,
                batch_size,
                args.epochs,
                np.random.default_rng([args.seed, batch_size]),
                grouping,
                neighbors,
            )
            rules.append((f"batch_{grouping}", grouping, batch_size, exposed))

    rules.append(
        (
            "corpus_uniform",
            "-",
            0,
            random_support(anchors, n_items, quota, np.random.default_rng(args.seed + 1)),
        )
    )
    if artifact is not None:
        rules.append(("teacher_topk", "-", 0, artifact_support(artifact, anchors, quota)))
        rules.append(("teacher_full_row", "-", 0, artifact_support(artifact, anchors, None)))

    fieldnames = [
        "pair",
        "support_rule",
        "batch_grouping",
        "batch_size",
        "epochs",
        "n_items",
        "quota",
        "support_size",
        *[f"recall_at_{k}" for k in RECALL_KS],
        "teacher_mass_coverage",
        "analytic_exposure",
        "eval_k",
        "holdout_frac",
        "n_anchors",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rule, grouping, batch_size, exposed in rules:
            rows = []
            for anchor in anchors:
                anchor = int(anchor)
                row_neighbors = neighbors[anchor]
                keep = row_neighbors >= 0
                rows.append(
                    _exposure_row(
                        exposed.get(anchor, set()),
                        row_neighbors[keep],
                        masses[anchor][keep],
                    )
                )
            aggregate = {
                key: float(np.mean([row[key] for row in rows])) for key in rows[0]
            }
            analytic = ""
            if rule == "batch_random" and batch_size > 1:
                per_epoch = (batch_size - 1) / (n_items - 1)
                analytic = 1.0 - (1.0 - per_epoch) ** args.epochs
            writer.writerow(
                {
                    "pair": args.pair,
                    "support_rule": rule,
                    "batch_grouping": grouping,
                    "batch_size": batch_size or "",
                    "epochs": args.epochs,
                    "n_items": n_items,
                    "quota": quota,
                    "eval_k": args.eval_k,
                    "holdout_frac": args.holdout_frac,
                    "n_anchors": len(anchors),
                    "analytic_exposure": analytic,
                    **{key: round(value, 6) for key, value in aggregate.items()},
                }
            )
            print(
                f"  {rule:<20} B={batch_size or '-':<5} "
                f"|S|={aggregate['support_size']:<8.1f} "
                f"R@10={aggregate['recall_at_10']:.4f} "
                f"R@1000={aggregate['recall_at_1000']:.4f} "
                f"mass={aggregate['teacher_mass_coverage']:.4f}"
            )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
