#!/usr/bin/env bash
# E3 -- Held-out geometry: does teacher-selected exposure generalize past its own
# supervised edges?
#
# No training. This scores the checkpoints E2 already produced, on relations no
# arm was ever supervised on, which is the objection E2 alone cannot answer: of
# course teacher-selected relations fit teacher-selected relations better.
#
# Three relation sets, three metrics each. The sets are built in
# scripts/exp/heldout_geometry.py and every one of them excludes the union of each
# anchor's diffusion pool and transition row, so no arm is scored on anything it
# could have been trained on:
#
#   heldout_edges  the 20% of teacher edges E2 withheld from every arm's support.
#                  The headline: did the student recover a neighbour that was
#                  deliberately hidden from it? Present only if E2 ran with
#                  HOLDOUT_FRAC > 0.
#   nonlocal       teacher ranks graph_k+1 .. 5*graph_k -- teacher-relevant but
#                  outside every arm's retrieval width. Available even for a
#                  family trained without a holdout.
#   unsupervised   everything else. The broadest and the loosest; most of it is
#                  pairs the teacher calls unrelated, so it tracks anisotropy.
#                  Reported for completeness, not as the headline.
#
# By default it scores the most recent E2 run. Point RUN_ROOT at another one to
# score that instead, or at an exp1/exp4 run root to score those checkpoints with
# the same instrument.
#
# Usage:
#   bash scripts/exp/exp3_heldout_geometry.sh
#   RUN_ROOT=results/exp2_support/<run_id>/<pair> bash scripts/exp/exp3_heldout_geometry.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT="exp3_heldout"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-exp2_support}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache/exp}"
CORPUS="${CORPUS:-data/train_set/merged_3_data_5k_each.csv}"
CORPUS_KEY="$(basename "${CORPUS%.*}")"
ANCHORS="${ANCHORS:-512}"
KNN_K="${KNN_K:-10}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/$EXPERIMENT}"
mkdir -p "$OUT_DIR"

# Which run tree to score. The pointer file is written by run_arms on every
# successful sweep, so the common case needs no argument at all.
if [[ -z "${RUN_ROOT:-}" ]]; then
    POINTER="$REPO_ROOT/runs/$SOURCE_EXPERIMENT/last_run_root.txt"
    if [[ ! -f "$POINTER" ]]; then
        echo "No RUN_ROOT given and no $POINTER; run $SOURCE_EXPERIMENT first" >&2
        exit 2
    fi
    RUN_ROOT="$(cat "$POINTER")"
    # exp2 nests one level per pair, so the pointer may name the pair directory
    # or its parent. Score every pair directory found underneath either way.
fi

mapfile -t RUN_DIRS < <(
    if [[ -d "$RUN_ROOT/runs" ]]; then
        printf '%s\n' "$RUN_ROOT"
    else
        find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -exec test -d '{}/runs' ';' -print
    fi
)
if (( ${#RUN_DIRS[@]} == 0 )); then
    echo "No run trees with a runs/ directory under $RUN_ROOT" >&2
    exit 2
fi

echo "E3 held-out geometry over ${#RUN_DIRS[@]} run tree(s)"
parts=()
for run_dir in "${RUN_DIRS[@]}"; do
    # The pair is recorded by the runner rather than parsed out of the path.
    pair="$(awk -F'\t' '$1=="pair"{print $2}' "$run_dir/run_config.tsv" 2>/dev/null || true)"
    pair="${pair:-qwen3_0_6b_to_minilmv2_h384}"
    # The graph the arms actually trained against carries the holdout metadata,
    # so the evaluation reconstructs exactly the split the build used. The mutual
    # graph is the one every arm but teacher_topk used, and the holdout is
    # identical across keys because it is a function of the pair and the seed.
    artifact="$CACHE_ROOT/$pair/$CORPUS_KEY/graph_mutual.pt"
    if [[ ! -f "$artifact" ]]; then
        artifact="$(find "$CACHE_ROOT/$pair/$CORPUS_KEY" -name 'graph_*.pt' | head -1)"
    fi
    if [[ ! -f "$artifact" ]]; then
        echo "No graph artifact for $pair under $CACHE_ROOT/$pair/$CORPUS_KEY" >&2
        exit 2
    fi
    out="$OUT_DIR/heldout_geometry_$pair.csv"
    echo "  $pair: $run_dir"
    "$PYTHON_BIN" "$SCRIPT_DIR/heldout_geometry.py" \
        --runs "$run_dir/runs" \
        --cache "$CACHE_ROOT/$pair/$CORPUS_KEY/teacher_train.pt" \
        --artifact "$artifact" \
        --train-data "$CORPUS" \
        --pair "$pair" \
        --anchors "$ANCHORS" \
        --knn-k "$KNN_K" \
        --out "$out"
    parts+=("$out")
done

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import csv, sys
from pathlib import Path

root = Path(sys.argv[1])
parts = sorted(root.glob("heldout_geometry_*.csv"))
rows, fields = [], []
for part in parts:
    with part.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for name in reader.fieldnames or []:
            if name not in fields:
                fields.append(name)
        rows.extend(reader)
out = root / "heldout_geometry.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"merged {len(parts)} CSVs -> {out} ({len(rows)} rows)")
PY

echo
echo "E3 outputs:"
echo "  $OUT_DIR/heldout_geometry.csv"
echo "Join it with runs/$SOURCE_EXPERIMENT/results.csv on (pair, arm, seed)."
