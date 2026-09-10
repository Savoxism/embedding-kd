#!/usr/bin/env bash
# E4 -- Exposure scaling: when does batch-defined support fail?
#
# If the problem with in-batch relational KD is that batch co-occurrence decides
# the support, then it must degrade systematically in N/B, because that ratio is
# exactly what sets how often a given teacher-relevant pair ever meets. Under
# random batching the prediction is closed-form, and it is made before the runs:
#
#     p_seen = 1 - (1 - (B-1)/(N-1))^E
#
# For the repository's own corpus (N=13553, B=64, E=5) that is 2.3%: a specific
# teacher-relevant neighbour has a 2.3% chance of ever being shown to its anchor
# in a whole training run.
#
# The sweep:
#     N in {5000, 13553, 25000, 48000}     corpus subsets, nested and stratified
#     B in {16, 64, 256}                   batch size
#     x 2 objectives                       in-batch vs teacher-support
#     + 1 pointwise control per N          (at B=64)
#     x 3 seeds                            = 84 runs
#
# Read it on N/B, not on N. The corpora available here cap N at ~48.7k -- the
# deduplicated union of the three training sets -- so N spans one decade, but N/B
# spans from 5000/256 ~ 20 to 48000/16 = 3000, which is the axis the hypothesis
# actually names.
#
# Two design points, because the headline claim depends on both:
#
# *The teacher-support arm is matched to the in-batch arm at every B.* Its budget
# is B-1 columns, the same number the batch exposes, so at each point on the
# x-axis the two arms differ in which relations and not in how many. That is why
# the teacher arm's curve being flat is informative rather than a budget effect.
#
# *y must be read relative to the same N.* A larger corpus makes every method
# better, and TRE also rises with N, so plotting an absolute score against TRE
# would show a "collapse onto one trend" driven entirely by N. The pointwise arm
# is trained at every N for exactly this: the quantity to plot is each relational
# arm's gap to pointwise at its own N. The CSV carries all three so the
# normalization happens in the analysis and stays visible.
#
# What the prediction is, stated honestly: the teacher-support arm should be much
# *flatter*, not flat. Its support does not read batch membership, but its
# encoder pool and its optimizer step count still change with B and N, so some
# dependence is expected and claiming none would be overselling it.
#
# Usage:
#   bash scripts/exp/exp4_exposure_scaling.sh
#   DRY_RUN=1 bash scripts/exp/exp4_exposure_scaling.sh
#   SIZES=5000,13553 BATCHES=16,64 bash scripts/exp/exp4_exposure_scaling.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

EXPERIMENT="exp4_scaling"
PAIR="${PAIR:-qwen3_0_6b_to_minilmv2_h384}"
SIZES="${SIZES:-5000,13553,25000,48000}"
BATCHES="${BATCHES:-16,64,256}"
SUBSET_DIR="${SUBSET_DIR:-data/train_set/scaling}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache/exp}"
EPOCHS="${EPOCHS:-5}"
OUT_DIR="$REPO_ROOT/runs/$EXPERIMENT"
mkdir -p "$OUT_DIR"

IFS=',' read -r -a SIZE_LIST <<< "$SIZES"
IFS=',' read -r -a BATCH_LIST <<< "$BATCHES"

# The subsets are written once and reused. A dry run only validates and prints
# the matrix, so it must remain side-effect free even from a clean checkout.
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/make_subsets.py" --sizes "$SIZES" --out-dir "$SUBSET_DIR"
fi

GRAPH_SPEC=""
ARMS_SPEC=""
MINIMAL="--relation_target direct --no_ambient --row_weight 0"

for n in "${SIZE_LIST[@]}"; do
    corpus="$SUBSET_DIR/scaling_n$n.csv"
    GRAPH_SPEC+="n$n|$corpus|
"
    for b in "${BATCH_LIST[@]}"; do
        support=$((b - 1))
        ARMS_SPEC+="in_batch_n${n}_b${b}|n$n|ggpkd|--batch_local $MINIMAL --batch_size $b
"
        ARMS_SPEC+="teacher_support_n${n}_b${b}|n$n|ggpkd|$MINIMAL --batch_size $b --diffusion_quota $support
"
    done
    # One pointwise run per corpus size: the reference every relational arm at
    # this N is read against, so the scaling plot shows relation exposure rather
    # than "more data helps".
    ARMS_SPEC+="pointwise_n${n}_b64|n$n|pointwise|--batch_size 64
"
done

export GRAPH_SPEC ARMS_SPEC PAIR
CSV_OUT="${CSV_OUT:-$OUT_DIR/results.csv}"

# `|| status=$?` rather than `; status=$?`: under `set -e` a failing call exits
# the script before the assignment runs, and the post-processing below -- which is
# still worth having when only some arms failed -- would never happen.
status=0
run_arms "$REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit $status
fi

# The x-axis. TRE@K per (N, B) is an exposure statistic, not a training result,
# so it is computed from the teacher cache of each subset -- cheap, and it is what
# the trained points are plotted against.
: > "$OUT_DIR/exposure.csv"
first=1
for n in "${SIZE_LIST[@]}"; do
    corpus_key="scaling_n$n"
    part="$OUT_DIR/exposure_n$n.csv"
    "$PYTHON_BIN" "$SCRIPT_DIR/coverage.py" \
        --cache "$CACHE_ROOT/$PAIR/$corpus_key/teacher_train.pt" \
        --artifact "$CACHE_ROOT/$PAIR/$corpus_key/graph_n$n.pt" \
        --pair "$PAIR" \
        --batch-sizes "$BATCHES" \
        --epochs "$EPOCHS" \
        --out "$part"
    if (( first )); then
        cat "$part" >> "$OUT_DIR/exposure.csv"
        first=0
    else
        tail -n +2 "$part" >> "$OUT_DIR/exposure.csv"
    fi
done

echo
echo "E4 outputs:"
echo "  $CSV_OUT              (downstream, per arm x seed)"
echo "  $OUT_DIR/exposure.csv (TRE@K per N and B, plus the analytic curve)"
echo "Join on (n_items, batch_size); plot y = arm score - pointwise score at the same N."
exit $status
