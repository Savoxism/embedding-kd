"""Report the graph-scale contribution to the floor of L_rel from an artifact.

Because cross-entropy is linear in the target, a weighted sum of forward KLs that
all share one student distribution satisfies

    sum_r w_r KL(p_r || q) = -sum_r w_r H(p_r) + CE(pbar, q),    pbar = sum_r w_r p_r,

so its minimum over every possible q is

    JS_w(p_1..p_R) = H(pbar) - sum_r w_r H(p_r)  >= 0.

The artifact contains the neighbor scale r=1 and multi-hop diffusion scales r>1;
the ambient r=0 profile is built online from the teacher bank. The reported floor
therefore covers the graph-scale portion of L_rel and is multiplied by that group's
weight in the full objective. If an observed loss_rel sits near this lower bound,
the objective is close to exhausted. Distinct per-scale temperatures break the
identity, and then JS_w is a diagnostic bound rather than the exact minimum.

Usage:
    python scripts/ggpkd/floor.py cache/ggpkd/qwen3_4b_bert_base_graph.pt
    python scripts/ggpkd/floor.py <artifact.pt> --observed-loss 0.83
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ggpkd.policy import diffusion_weights


def _entropy(probs: np.ndarray) -> np.ndarray:
    mask = probs > 0
    safe = np.where(mask, probs, 1.0)
    return -(np.where(mask, probs * np.log(safe), 0.0)).sum(axis=-1)


def _percentiles(values: np.ndarray, label: str, unit: str = "nats") -> None:
    qs = [1, 10, 25, 50, 75, 90, 99]
    cuts = np.percentile(values, qs)
    body = "  ".join(f"p{q}={cut:.4f}" for q, cut in zip(qs, cuts))
    print(f"  {label}: mean={values.mean():.4f} {unit}   {body}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=str, help="path to the GGPKD artifact .pt")
    parser.add_argument(
        "--observed-loss",
        type=float,
        default=None,
        help="a loss_rel value from training, to compare against the floor",
    )
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    metadata = artifact.get("metadata", {})
    if "pool_probs" not in artifact:
        print(
            f"{args.artifact} is a pre-v3 artifact (keys: {sorted(artifact)}). "
            f"Delete it and rerun training to rebuild."
        )
        return 1

    scales = tuple(metadata.get("diffusion_scales", ()))
    pool_indices = artifact["pool_indices"].numpy()
    probs = artifact["pool_probs"].numpy().astype(np.float64)
    n_scales, n_items, _ = probs.shape
    if len(scales) != n_scales:
        print(
            f"artifact scale metadata {scales} does not match pool_probs axis {n_scales}"
        )
        return 1
    raw_weights = np.asarray(diffusion_weights(scales), dtype=np.float64)
    # The ambient raw weight equals the r=1 raw weight. Normalize graph scales
    # internally to form their mixture, then multiply their JS divergence by the
    # graph group's actual weight in the full relational objective.
    weights = raw_weights / raw_weights.sum()
    graph_group_weight = raw_weights.sum() / (raw_weights[0] + raw_weights.sum())

    valid = pool_indices >= 0
    probs = np.where(valid[None, :, :], probs, 0.0)
    totals = probs.sum(axis=-1, keepdims=True)
    probs = np.divide(probs, np.maximum(totals, 1e-12))

    print(f"artifact          : {args.artifact}")
    print(f"n_items           : {n_items}")
    print(f"scales            : {scales}")
    print(f"graph mix weights : {np.round(weights, 4).tolist()}")
    print(f"graph weight L_rel: {graph_group_weight:.4f}")
    print(f"teacher print     : {metadata.get('teacher_fingerprint', 'n/a')[:16]}")
    print()

    per_scale_entropy = np.stack([_entropy(probs[r]) for r in range(n_scales)])
    mixture = (probs * weights.reshape(-1, 1, 1)).sum(axis=0)
    mixture_entropy = _entropy(mixture)
    js = graph_group_weight * np.clip(
        mixture_entropy - (per_scale_entropy * weights.reshape(-1, 1)).sum(0),
        0,
        None,
    )

    print("Per-scale targets")
    for r in range(n_scales):
        support = (probs[r] > 0).sum(axis=-1).astype(np.float64)
        kl_uniform = np.log(np.maximum(support, 1.0)) - per_scale_entropy[r]
        print(
            f"  r={scales[r] if r < len(scales) else r}: "
            f"support={support.mean():6.2f}  H={per_scale_entropy[r].mean():.4f}  "
            f"top1={probs[r].max(axis=-1).mean():.4f}  "
            f"KL(p||U_supp)={kl_uniform.mean():.4f}"
        )
    print()

    print("Cross-scale divergence (are the scales actually different?)")
    clamped = np.clip(probs, 1e-12, None)
    clamped = clamped / clamped.sum(axis=-1, keepdims=True)
    for r in range(n_scales - 1):
        cross = (clamped[r] * (np.log(clamped[r]) - np.log(clamped[r + 1]))).sum(-1)
        left = scales[r] if r < len(scales) else r
        right = scales[r + 1] if r + 1 < len(scales) else r + 1
        print(f"  KL(p_r{left} || p_r{right}) = {cross.mean():.4f} nats")
    print()

    print("Graph-scale contribution to the L_rel floor (tied temperature)")
    _percentiles(js, "W_graph * JS_w(p_1..p_R)")
    print(f"  mixture entropy: mean={mixture_entropy.mean():.4f} nats")
    print(f"  mixture top1   : mean={mixture.max(axis=-1).mean():.4f}")
    print()

    floor = float(js.mean())
    if args.observed_loss is not None:
        excess = args.observed_loss - floor
        share = 100.0 * floor / max(args.observed_loss, 1e-12)
        print(f"observed loss_rel  = {args.observed_loss:.4f}")
        print(f"floor              = {floor:.4f}  ({share:.1f}% of the observed value)")
        print(f"excess KL(pbar||q) = {excess:.4f}")
        if excess < 0.05:
            print(
                "\nVERDICT: the student is sitting on the floor. The objective has no "
                "gradient left to give -- this is target design, not optimization. "
                "Untie the per-scale temperatures and/or widen the candidate pool."
            )
        elif excess < 0.2:
            print(
                "\nVERDICT: close to the floor. Most of the remaining loss is the "
                "scale disagreement, not fittable structure."
            )
        else:
            print(
                "\nVERDICT: well above the floor -- the objective still carries "
                "gradient, so saturation would point at optimization or capacity."
            )
    else:
        print(f"floor              = {floor:.4f} nats")
        print("Pass --observed-loss <value> to compare a training loss against it.")

    if floor < 0.01:
        print(
            "\nWARNING: JS_w < 0.01 nats. The scales carry essentially the same target, "
            "so with a tied temperature the multi-scale objective is numerically "
            "identical to a single-scale one."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
