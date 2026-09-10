#!/usr/bin/env python3
"""A kNN graph built from the *base student's* embeddings, for the student-kNN arm.

The arm it serves asks a specific question: is teacher relevance what matters, or
merely that the support looks semantic? The base student already has a usable
notion of similarity, so its own nearest neighbours are semantic support that the
teacher did not choose. If the student-kNN arm lands with the random controls, the
answer is teacher relevance; if it lands with the teacher arms, the paper's claim
has to weaken to "any reasonable semantic support".

Two caveats, both of which belong in the paper rather than in a footnote here:

*The bandwidths are the student's.* The artifact stores one temperature per row,
derived from that encoder's own cosine spread, and the criterion scores each
anchor at the bandwidth stored with its row. So this arm differs from the teacher
arms in temperature as well as in support -- one factor more than the other arms
do. That is why it is opt-in and why it is described as indicative.

*The support is frozen.* It comes from the base student, before any distillation,
and does not follow the student as it trains. Recomputing it during training would
make the support a moving target and the arm would no longer be a controlled
comparison; it would also cost a corpus encode per epoch.

The teacher targets are unaffected: the training arm runs at
`--relation_target direct`, which reads the teacher bank, so what changes is which
columns get scored, not what they are scored against.

Usage:
    python scripts/exp/student_graph.py \
        --pair qwen3_0_6b_to_minilmv2_h384 \
        --train-data data/train_set/merged_3_data_5k_each.csv \
        --out cache/exp/<pair>/<corpus>/graph_student_knn.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ggpkd.graph_builder import build_or_load_ggpkd_artifact  # noqa: E402
from scripts.exp.heldout_geometry import corpus_texts  # noqa: E402

# The same pair table scripts/ggpkd/train.sh carries. Kept in sync by being read
# only for the student name: a mismatch would build the graph with the wrong
# encoder, which the artifact's own teacher fingerprint would then happily accept
# because it fingerprints whatever it is given.
PAIR_STUDENTS = {
    "qwen3_0_6b_to_minilmv2_h384": "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base",
    "bge_m3_to_minilmv2_h768": "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base",
    "qwen3_4b_to_bert_base": "google-bert/bert-base-uncased",
}


@torch.no_grad()
def encode(
    model_name: str, texts: list[str], max_length: int, batch_size: int, device: str
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    outputs = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        result = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        # CLS pooling, matching the student path in training and evaluation.
        outputs.append(result.last_hidden_state[:, 0, :].float().cpu())
    return torch.cat(outputs, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, choices=sorted(PAIR_STUDENTS))
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--graph-k", type=int, default=200)
    parser.add_argument("--holdout-frac", type=float, default=0.0)
    parser.add_argument("--holdout-seed", type=int, default=12345)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    texts = corpus_texts(Path(args.train_data))
    student_name = PAIR_STUDENTS[args.pair]
    print(f"encoding {len(texts)} texts with the base student {student_name}")
    embeddings = encode(
        student_name, texts, args.max_length, args.batch_size, args.device
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_or_load_ggpkd_artifact(
        teacher_embeddings=embeddings,
        cache_path=str(out_path),
        log_dir=str(out_path.parent / "student_graph_logs"),
        graph_k=args.graph_k,
        diffusion_scales=(1,),
        knn_mode="mutual",
        holdout_edge_frac=args.holdout_frac,
        holdout_seed=args.holdout_seed,
    )
    print(f"student kNN graph written: {out_path}")
    print(
        "NOTE: this artifact's row bandwidths are the student's, so the arm that "
        "trains on it differs from the teacher arms in temperature as well as in "
        "support. Report it as indicative."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
