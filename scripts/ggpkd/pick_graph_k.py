#!/usr/bin/env python3
"""Pick graph_k by the sharpness it induces, before spending a training run.

graph_k is no longer free headroom. Bandwidth is tau_i = (s(1) - s(k)) / log(k),
so k sets how sharp every transition row is, and the row is the target the student
is trained against. Too large a k and the row goes uniform: the span it measures
stops being the local cosine decay and becomes the distance out of the anchor's
own neighbourhood, so L_rel degenerates into a binary neighbour/non-neighbour
objective that carries no ranking information.

The quantity to read is KL(p || uniform on support) = log|supp| - H(p), which is
what the build already reports as `target_kl_uniform_r1`:

    < 0.05   degenerate; the build warns and the run is not worth starting
    ~ 0.76   what the previous entropic-affinity graph (perplexity 30) trained at

This builds the graph at each k and prints that column, so the choice costs one
teacher-embedding pass rather than one training run per candidate.

Usage:
    python scripts/ggpkd/pick_graph_k.py --cache cache/ggpkd/<pair>/teacher_train.pt
    python scripts/ggpkd/pick_graph_k.py --cache <path> --k 20,40,80,150,200
"""

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ggpkd.graph_builder import build_or_load_ggpkd_artifact  # noqa: E402

DEGENERATE = 0.05
REFERENCE = 0.757  # the perplexity-30 graph the earlier runs trained on


def sharpness(artifact) -> dict:
    probs = artifact["pool_probs"].numpy()[0].astype(np.float64)
    support = (probs > 0).sum(axis=1)
    entropy = -np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-300)), 0.0).sum(
        axis=1
    )
    kl = np.log(np.maximum(support, 1)) - entropy
    temps = artifact["row_temps"].numpy()
    fill = (artifact["pool_indices"].numpy() >= 0).sum(axis=1)
    return {
        "kl_p50": float(np.median(kl)),
        "kl_p10": float(np.quantile(kl, 0.1)),
        "degenerate_share": float((kl < DEGENERATE).mean()),
        "eff_perplexity": float(np.median(np.exp(entropy))),
        "tau_p50": float(np.median(temps)),
        "fill_mean": float(fill.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="cached teacher embeddings (.pt)")
    parser.add_argument("--k", default="20,40,80,150,200")
    parser.add_argument("--scales", default="1")
    args = parser.parse_args()

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    embeddings = blob["embeddings"] if isinstance(blob, dict) else blob
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(np.asarray(embeddings))
    scales = tuple(int(s) for s in args.scales.split(","))
    print(f"corpus {tuple(embeddings.shape)}, scales={scales}\n")

    header = f"{'graph_k':>8} {'KL p50':>8} {'KL p10':>8} {'<0.05':>7} {'perp_hd':>8} {'tau p50':>9} {'fill':>7}"
    print(header)
    print("-" * len(header))
    rows = []
    for k in (int(part) for part in args.k.split(",")):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            artifact = build_or_load_ggpkd_artifact(
                teacher_embeddings=embeddings,
                cache_path=str(Path(tmp) / "g.pt"),
                log_dir=tmp,
                graph_k=k,
            )
        stats = sharpness(artifact)
        rows.append((k, stats))
        print(
            f"{k:8d} {stats['kl_p50']:8.3f} {stats['kl_p10']:8.3f} "
            f"{stats['degenerate_share']:6.1%} {stats['eff_perplexity']:8.1f} "
            f"{stats['tau_p50']:9.4f} {stats['fill_mean']:7.1f}"
        )

    print(f"\ndegenerate below {DEGENERATE}; the perplexity-30 graph read {REFERENCE}")
    usable = [(k, s) for k, s in rows if s["kl_p50"] >= DEGENERATE]
    if not usable:
        print("every k tried is degenerate -- extend the sweep downward")
        return
    best = min(usable, key=lambda item: abs(item[1]["kl_p50"] - REFERENCE))
    print(
        f"closest to the sharpness the earlier runs trained at: graph_k={best[0]} "
        f"(KL p50={best[1]['kl_p50']:.3f}, {best[1]['fill_mean']:.1f} columns per anchor)"
    )


if __name__ == "__main__":
    main()
