"""Replay Table 2 support coverage without training a student.

Coverage is cumulative teacher mass exposed through the graph-support slots.
Restriction error is ``-log(mass selected this epoch)``. Both quantities use
the same scale-weighted teacher row optimized by the relation objective.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ggpkd.candidate_sampler import GGPKDCandidateSampler
from src.ggpkd.policy import (
    derive_diffusion_quota,
    normalized_diffusion_weights,
)

TABLE_POLICIES = ("topk", "proportional", "uniform")


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
    """Return cumulative coverage and per-epoch restriction error."""
    sampler = GGPKDCandidateSampler(
        artifact=artifact,
        diffusion_quota=quota,
        hard_neg_k=hard_neg_k,
        random_neg_k=random_neg_k,
        seed=seed,
        support_policy=policy,
    )
    pool_probs = artifact["pool_probs"].numpy().astype(np.float64)
    scales = tuple(artifact["metadata"]["diffusion_scales"])
    weights = normalized_diffusion_weights(scales).astype(np.float64)
    # Do not renormalize over the cached pool. Its missing residual is teacher
    # mass discarded by graph truncation, so renormalizing here would inflate
    # corpus-level coverage and understate the restriction error. The reported
    # maximum can therefore be below one, which is the intended global-mass
    # interpretation.
    mixture = np.einsum("s,sij->ij", weights, pool_probs)

    coverage = np.zeros((epochs, anchors.size), dtype=np.float64)
    epsilon = np.zeros_like(coverage)
    ever: list[set[int]] = [set() for _ in range(anchors.size)]

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for slot, anchor in enumerate(anchors):
            idx = int(anchor)
            rng = sampler._rng(idx, GGPKDCandidateSampler._STREAM_CANDIDATES)
            _, positions = sampler._select_support_impl(idx, rng)
            row = mixture[idx]
            selected_mass = float(row[positions].sum()) if positions.size else 0.0
            epsilon[epoch, slot] = -np.log(np.clip(selected_mass, 1e-12, 1.0))
            ever[slot].update(int(position) for position in positions)
            seen = np.fromiter(ever[slot], dtype=np.int64, count=len(ever[slot]))
            coverage[epoch, slot] = float(row[seen].sum()) if seen.size else 0.0

    if epochs > 1 and (np.diff(coverage, axis=0) < -1e-9).any():
        raise RuntimeError(f"{policy}: cumulative coverage decreased")
    return coverage, epsilon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-anchors", type=int, default=2048)
    parser.add_argument("--quota", type=int)
    parser.add_argument("--hard-neg-k", type=int, default=40)
    parser.add_argument("--random-neg-k", type=int, default=26)
    parser.add_argument("--policies", default=",".join(TABLE_POLICIES))
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    scales = tuple(artifact["metadata"]["diffusion_scales"])
    quota = args.quota or derive_diffusion_quota(artifact["pool_probs"].numpy(), scales)
    policies = tuple(part.strip() for part in args.policies.split(",") if part.strip())
    invalid = sorted(set(policies) - set(TABLE_POLICIES))
    if invalid:
        parser.error(f"Table 2 policies must be one of {TABLE_POLICIES}: {invalid}")

    n_items = int(artifact["pool_indices"].shape[0])
    if 0 < args.n_anchors < n_items:
        anchors = np.random.default_rng(0).choice(
            n_items, size=args.n_anchors, replace=False
        )
        anchors.sort()
    else:
        anchors = np.arange(n_items)

    rows: list[dict[str, float | int | str]] = []
    seeds = tuple(int(seed) for seed in args.seeds.split(","))
    for policy in policies:
        for seed in seeds:
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
                        "coverage_mean": float(coverage[epoch].mean()),
                        "coverage_p10": float(np.percentile(coverage[epoch], 10)),
                        "coverage_p90": float(np.percentile(coverage[epoch], 90)),
                        "epsilon_mean": float(epsilon[epoch].mean()),
                        "epsilon_p90": float(np.percentile(epsilon[epoch], 90)),
                        "n_anchors": int(anchors.size),
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows, R={scales}, quota={quota})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
