#!/usr/bin/env python3
"""Measure the candidate-encoder step cost against chunk size and pad alignment.

The GGPKD step spends essentially all of its time in one place. A measured run
(Qwen3-0.6B -> MiniLMv2-H384, batch 64) encoded **4,445 unique texts per step to
serve 64 anchors** -- a 70x multiplier -- for ~73k real tokens, at 470 MB peak
and ~0.2 s/step. That was candidate width 89. With the negatives removed the
width is the diffusion quota alone and the pool is ~1,400 texts, so ``--pool``
defaults to that; pass ``--pool 4445`` to reproduce the old draw. Everything else in the
step, the relational loss included, is rounding error next to that.

Two knobs decide how those texts are cut into forward calls, and neither can
change a student embedding: with correct attention masks, padding does not enter
the encoder's output. They trade padded FLOPs against kernel launches:

* ``ENCODE_CHUNK_SIZE`` -- wider chunks mean fewer launches and strictly more
  padding, because a length-sorted chunk pads to its own longest member.
* ``PAD_TO_MULTIPLE_OF`` -- at a median of 13 tokens, rounding each chunk's width
  up to a multiple of 8 costs ~17% of all tokens by itself. It buys tensor-core
  alignment on the sequence axis, which may or may not be worth that.

Which side wins is a property of the GPU, so this script measures it instead of
guessing. It runs the real student on the real corpus length distribution and
times forward + backward, which is what training actually pays.

Usage:
    python scripts/ggpkd/bench_encode.py
    python scripts/ggpkd/bench_encode.py --student <hf-name> --pool 4445
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ggpkd.policy import ENCODE_CHUNK_SIZE, PAD_TO_MULTIPLE_OF


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--student", default="nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base"
    )
    p.add_argument("--train_data", default="data/train_set/merged_3_data_5k_each.csv")
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_length", type=int, default=256)
    # Defaults are the measured production numbers, so a bare run reproduces the
    # real step rather than a synthetic one.
    p.add_argument("--pool", type=int, default=1400, help="unique candidates per step")
    p.add_argument("--batch", type=int, default=64, help="anchors per step")
    p.add_argument("--chunks", default="128,256,512,1024,2048")
    p.add_argument("--pads", default="1,8,16")
    p.add_argument("--steps", type=int, default=12, help="timed steps per setting")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--compile",
        action="store_true",
        help="Also measure torch.compile(dynamic=True). The step is launch bound, "
        "which is what compile exists to fix, but each distinct chunk width is a "
        "new shape -- raise --warmup well above the default when using this.",
    )
    return p.parse_args()


def pad_block(rows: list[np.ndarray], pad_id: int, multiple: int):
    width = max(len(r) for r in rows)
    if multiple > 1:
        width = ((width + multiple - 1) // multiple) * multiple
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, : r.size] = r
        mask[i, : r.size] = 1
    return torch.from_numpy(ids), torch.from_numpy(mask)


def time_setting(model, corpus_ids, lengths, pad_id, device, chunk, multiple, args):
    """One timed measurement of `steps` full forward+backward passes."""
    rng = np.random.default_rng(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)
    padded_tokens = 0
    durations = []

    for step in range(args.warmup + args.steps):
        nodes = rng.choice(len(corpus_ids), size=args.pool, replace=False)
        order = np.argsort(lengths[nodes], kind="stable")
        nodes = nodes[order]
        anchors = rng.choice(len(corpus_ids), size=args.batch, replace=False)

        blocks = [
            pad_block([corpus_ids[i] for i in nodes[s : s + chunk]], pad_id, multiple)
            for s in range(0, nodes.size, chunk)
        ]
        blocks.append(
            pad_block([corpus_ids[i] for i in anchors], pad_id, multiple)
        )
        blocks = [(i.to(device), m.to(device)) for i, m in blocks]

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pooled = [
                model(input_ids=i, attention_mask=m).last_hidden_state[:, 0, :]
                for i, m in blocks
            ]
            # A scalar with a real dependence on every encoded row, so the
            # backward pass has the same shape as the training one.
            loss = torch.cat(pooled, dim=0).float().pow(2).mean()
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= args.warmup:
            durations.append(time.perf_counter() - start)
            padded_tokens += sum(int(i.numel()) for i, _ in blocks)

    return float(np.median(durations)), padded_tokens / max(1, len(durations))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "WARNING: no CUDA device. The launch-vs-FLOP trade-off this script "
            "exists to measure does not reproduce on CPU; the numbers below will "
            "not tell you what to set.\n"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.student, use_fast=True)
    model = AutoModel.from_pretrained(args.student).to(device)
    model.train()
    if args.compile:
        # dynamic=True so the handful of distinct chunk widths share one graph
        # instead of triggering a recompile each.
        model = torch.compile(model, dynamic=True)
        print("torch.compile enabled (dynamic=True)\n")

    frame = pd.read_csv(args.train_data)
    texts = frame[args.text_column].astype(str)
    texts = texts[~texts.duplicated(keep="first")].tolist()
    encoded = tokenizer(texts, truncation=True, max_length=args.max_length)["input_ids"]
    corpus_ids = [np.asarray(e, dtype=np.int64) for e in encoded]
    lengths = np.asarray([len(e) for e in corpus_ids], dtype=np.int64)
    pad_id = tokenizer.pad_token_id or 0

    print(
        f"student={args.student}\n"
        f"corpus={len(corpus_ids)} texts, mean {lengths.mean():.1f} tokens "
        f"(p50 {np.percentile(lengths, 50):.0f}, p90 {np.percentile(lengths, 90):.0f})\n"
        f"simulating {args.pool} candidates + {args.batch} anchors per step, "
        f"{args.steps} timed steps per setting\n"
    )

    chunks = [int(c) for c in args.chunks.split(",")]
    pads = [int(p) for p in args.pads.split(",")]
    results = {}
    print(f"{'chunk':>7} {'pad':>4} {'fwd':>5} {'padded tok':>12} {'ms/step':>10} {'vs current':>11}")
    print("-" * 56)
    for chunk in chunks:
        for multiple in pads:
            ms, tokens = time_setting(
                model, corpus_ids, lengths, pad_id, device, chunk, multiple, args
            )
            calls = -(-args.pool // chunk) + 1
            results[(chunk, multiple)] = ms
            print(f"{chunk:>7} {multiple:>4} {calls:>5} {tokens:>12,.0f} {ms * 1000:>10.1f}", end="")
            base = results.get((ENCODE_CHUNK_SIZE, PAD_TO_MULTIPLE_OF))
            print(f" {ms / base:>10.2f}x" if base else f" {'--':>11}")

    best = min(results, key=results.get)
    base = results.get((ENCODE_CHUNK_SIZE, PAD_TO_MULTIPLE_OF))
    print(
        f"\nfastest: ENCODE_CHUNK_SIZE={best[0]}, PAD_TO_MULTIPLE_OF={best[1]}"
        + (
            f"  ({1 - results[best] / base:.1%} faster than the current "
            f"{ENCODE_CHUNK_SIZE}/{PAD_TO_MULTIPLE_OF})"
            if base and results[best] < base
            else "  (the current setting is already fastest)"
        )
    )
    print("Set both in src/ggpkd/policy.py.")
    if not args.compile:
        print(
            "\nThe measured step is launch bound (470 MB peak, a few percent of "
            "arithmetic throughput on a 22.7M-parameter student over 13-token "
            "sequences). Re-run with --compile --warmup 20 to see whether "
            "torch.compile closes that gap before tuning these two knobs further."
        )


if __name__ == "__main__":
    main()
