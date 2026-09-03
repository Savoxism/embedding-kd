"""Support-coverage replay: what each selection policy ever shows the student.

The first link of the paper's causal chain -- support coverage -> geometry
preservation -> downstream -- is a property of the *sampler*, not of a trained
model. The candidate stream is seeded by (seed, epoch, anchor) and reads nothing
but the graph artifact, so the exact columns a policy would draw in a real run
can be replayed offline, in seconds, on a CPU. Figure 1 and the coverage columns
of the support table come from here; no training run is needed to produce either,
and none is re-run when the figure changes.

Two quantities, both defined against the same teacher mixture row
p_i = sum_r omega_r p_i^(r) that the objective's diffusion group targets:

  coverage(epoch)  the teacher mass of every column anchor i has *ever* drawn up
                   to and including this epoch: 1 - delta_T. Cumulative, because
                   supervision is cumulative -- a column drawn in epoch 1 has
                   already moved the student when epoch 3 comes round.
  epsilon(epoch)   the per-epoch restriction distortion. Renormalizing a
                   probability row onto a subset that holds mass m costs exactly
                   KL(ptilde || p) = -log m nats, so this is the target
                   perturbation the policy imposes, in the units of the loss.

Top-K should maximize teacher-mass coverage under the per-epoch budget and then
stay flat because it draws the same support every epoch. Proportional and uniform
sampling expose new columns across epochs, while paying higher restriction
distortion on each individual update.

Usage:
    python scripts/ablation/replay_coverage.py \
        --artifact cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/graph_base.pt \
        --out runs/ablation/analysis/coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ggpkd.candidate_sampler import GGPKDCandidateSampler  # noqa: E402
from src.ggpkd.policy import (  # noqa: E402
    SUPPORT_POLICIES,
    derive_diffusion_quota,
    normalized_diffusion_weights,
)


def replay(
    artifact: dict,
    policy: str,
    *,
    quota: int,
    hard_neg_k: int,
    random_neg_k: int,
    seed: int,
    epochs: int,
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative coverage and per-epoch epsilon, both (epochs, n_anchors)."""
    sampler = GGPKDCandidateSampler(
        artifact=artifact,
        diffusion_quota=quota,
        hard_neg_k=hard_neg_k,
        random_neg_k=random_neg_k,
        seed=seed,
        support_policy=policy,
    )
    pool_indices = artifact["pool_indices"].numpy()
    pool_probs = artifact["pool_probs"].numpy()
    scales = artifact["metadata"]["diffusion_scales"]
    weights = normalized_diffusion_weights(scales).astype(np.float64)
    # The mixture the diffusion group actually optimizes, so coverage is measured
    # against the target rather than against any single scale.
    mixture = np.einsum("s,sij->ij", weights, pool_probs.astype(np.float64))
    mixture /= np.maximum(mixture.sum(axis=1, keepdims=True), 1e-12)

    coverage = np.zeros((epochs, anchors.size), dtype=np.float64)
    epsilon = np.zeros((epochs, anchors.size), dtype=np.float64)
    # Position sets, not corpus ids: mixture is indexed by pool position.
    ever: list[set[int]] = [set() for _ in range(anchors.size)]

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for slot, anchor in enumerate(anchors):
            idx = int(anchor)
            rng = sampler._rng(idx, GGPKDCandidateSampler._STREAM_CANDIDATES)
            _, positions = sampler._select_support_impl(idx, rng)
            row = mixture[idx]
            drawn = float(row[positions].sum()) if positions.size else 0.0
            # Clipped, not asserted: a row whose whole pool mass is below 1 after
            # truncation can leave -log slightly negative from float error, and a
            # figure should not die of 1e-16.
            epsilon[epoch, slot] = -np.log(np.clip(drawn, 1e-12, 1.0))
            ever[slot].update(int(p) for p in positions)
            seen = np.fromiter(ever[slot], dtype=np.int64, count=len(ever[slot]))
            coverage[epoch, slot] = float(row[seen].sum()) if seen.size else 0.0

    # Duplicate positions across scales are already collapsed by the draw itself,
    # so coverage is monotone by construction; assert it rather than trust it.
    if epochs > 1 and (np.diff(coverage, axis=0) < -1e-9).any():
        raise RuntimeError(f"{policy}: cumulative coverage decreased between epochs")
    return coverage, epsilon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="GGPKD graph artifact .pt")
    parser.add_argument("--out", required=True, help="output CSV")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="seeds to replay; the spread across them is the figure's error band",
    )
    parser.add_argument(
        "--n-anchors",
        type=int,
        default=2048,
        help="anchors sampled for the replay; 0 replays the whole corpus",
    )
    parser.add_argument("--quota", type=int, default=None, help="default: derived")
    parser.add_argument("--hard-neg-k", type=int, default=40)
    parser.add_argument("--random-neg-k", type=int, default=26)
    parser.add_argument(
        "--policies", default=",".join(SUPPORT_POLICIES), help="comma-separated"
    )
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    scales = artifact["metadata"]["diffusion_scales"]
    quota = args.quota or derive_diffusion_quota(
        artifact["pool_probs"].numpy(), scales
    )
    n_items = int(artifact["pool_indices"].shape[0])
    # A fixed anchor sample, seeded independently of the candidate stream, so
    # every policy is measured on exactly the same anchors.
    if args.n_anchors and args.n_anchors < n_items:
        anchors = np.random.default_rng(0).choice(
            n_items, size=args.n_anchors, replace=False
        )
        anchors.sort()
    else:
        anchors = np.arange(n_items)

    print(
        f"artifact={args.artifact} scales={scales} quota={quota} "
        f"anchors={anchors.size}/{n_items} epochs={args.epochs}"
    )

    rows = []
    for policy in args.policies.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            coverage, epsilon = replay(
                artifact,
                policy,
                quota=quota,
                hard_neg_k=args.hard_neg_k,
                random_neg_k=args.random_neg_k,
                seed=seed,
                epochs=args.epochs,
                anchors=anchors,
            )
            for epoch in range(args.epochs):
                rows.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "epoch": epoch + 1,
                        "quota": quota,
                        "coverage_mean": coverage[epoch].mean(),
                        "coverage_p10": np.percentile(coverage[epoch], 10),
                        "coverage_p90": np.percentile(coverage[epoch], 90),
                        "epsilon_mean": epsilon[epoch].mean(),
                        "epsilon_p90": np.percentile(epsilon[epoch], 90),
                        "n_anchors": anchors.size,
                    }
                )
            print(
                f"  {policy:13s} seed {seed}: coverage "
                + " -> ".join(f"{coverage[e].mean():.3f}" for e in range(args.epochs))
                + f" | final epsilon {epsilon[-1].mean():.4f} nats"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
