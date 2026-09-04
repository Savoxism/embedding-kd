"""Resolve the derived Top-K quota and fixed-width sensitivity allocations.

The canonical diffusion quota is derived from a GGPKD graph artifact. For a
Top-K sensitivity arm, ``--multiplier`` changes that quota while reallocating
the remaining hard/random negatives in their canonical ratio, keeping the total
candidate width constant.

Examples:
    python scripts/ablation/budget.py --artifact graph_base.pt --format quota
    python scripts/ablation/budget.py --artifact graph_base.pt --format width
    python scripts/ablation/budget.py --artifact graph_base.pt --multiplier .5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ggpkd.policy import derive_diffusion_quota  # noqa: E402


def _round_positive(value: float) -> int:
    return max(1, int(value + 0.5))


def allocate_fixed_width(
    base_quota: int,
    hard: int,
    random: int,
    multiplier: float,
) -> tuple[int, int, int]:
    if base_quota < 1:
        raise ValueError("base quota must be positive")
    if hard < 0 or random < 0:
        raise ValueError("negative quotas must be non-negative")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")

    width = base_quota + hard + random
    quota = _round_positive(base_quota * multiplier)
    if quota > width:
        raise ValueError(
            f"multiplier {multiplier:g} requests Top-K={quota}, larger than "
            f"the fixed candidate width {width}"
        )

    remaining = width - quota
    negative_total = hard + random
    if negative_total == 0:
        return quota, remaining, 0
    new_hard = int(remaining * hard / negative_total + 0.5)
    new_random = remaining - new_hard
    return quota, new_hard, new_random


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", type=Path)
    source.add_argument("--quota", type=int, help="explicit canonical quota")
    parser.add_argument("--hard", type=int, default=40)
    parser.add_argument("--random", type=int, default=26)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument(
        "--format",
        choices=("allocation", "quota", "width"),
        default="allocation",
    )
    args = parser.parse_args()

    if args.artifact is not None:
        artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
        base_quota = derive_diffusion_quota(
            artifact["pool_probs"].numpy(),
            artifact["metadata"]["diffusion_scales"],
        )
    else:
        base_quota = args.quota

    quota, hard, random = allocate_fixed_width(
        base_quota, args.hard, args.random, args.multiplier
    )
    if args.format == "quota":
        print(base_quota)
    elif args.format == "width":
        print(base_quota + args.hard + args.random)
    else:
        print(quota, hard, random)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
