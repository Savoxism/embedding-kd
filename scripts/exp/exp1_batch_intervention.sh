#!/usr/bin/env bash
# E1 -- Batch intervention: does batching change the learning problem?
#
# The claim under test is that a mini-batch is a computational device for some
# objectives and part of the objective for others. The intervention is batch
# composition; everything else is held fixed.
#
#   3 batch samplers   random (i.i.d.) / teacher_neighbor / teacher_diverse
#   x 3 objectives     pointwise KD / in-batch relational KD / teacher-support
#                      relational KD
#   x 3 seeds          = 27 runs
#
# Prediction, stated before the runs so the result can contradict it:
#
#   pointwise           flat across all three samplers. Its objective reads one
#                       example at a time, so whatever spread it shows is seed
#                       noise -- and that spread is the yardstick the other two
#                       are read against. This arm is a measurement floor, not a
#                       competitor.
#   teacher_support     flat. Its support is the teacher's kNN row, which does
#                       not know what else is in the batch.
#   in_batch            moves. Its support *is* the batch, so a teacher-neighbour
#                       grouping hands it far more teacher-relevant relations
#                       than a diverse grouping does.
#
# Two things this script is careful about, because without them the comparison
# is not one-factor:
#
# *Matched support size.* The in-batch arm scores each anchor against the other
# B-1 texts in its batch. The teacher-support arm is therefore given a budget of
# B-1 columns too (SUPPORT_SIZE below), rather than its natural full row, so the
# two differ in *which* relations, not in how many.
#
# *Matched temperature.* The in-batch arm runs at `--relation_target direct`, so
# its target is the teacher's cosine profile at the anchor's own tau_i -- exactly
# what the teacher-support arm uses. The older `ambient_only` form of this
# baseline scores at a single global temperature instead, which would have made
# the two arms differ in two things at once.
#
# Both relational arms drop the ambient scale and the auxiliary row loss
# (`--no_ambient --row_weight 0`), leaving one KL per anchor over that anchor's
# own columns. That is the minimal relational objective: the shared candidate
# pool still deduplicates the encoder work, but the loss is provably unaffected
# by it, because the graph softmax is masked to each anchor's own draw.
#
# Read the result from `runs/exp1_batch/results.csv` (downstream + diagnostics)
# together with `runs/exp1_batch/coverage.csv`, which measures how teacher-
# relevant each grouping's in-batch relations actually came out. The paper needs
# both: the second says the intervention did what it claims, the first says what
# that did to the student.
#
# Usage:
#   bash scripts/exp/exp1_batch_intervention.sh            # 27 runs
#   DRY_RUN=1 bash scripts/exp/exp1_batch_intervention.sh  # print the matrix
#   GPUS=0,1 SEEDS=42,43,44 bash scripts/exp/exp1_batch_intervention.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

EXPERIMENT="exp1_batch"
PAIR="${PAIR:-qwen3_0_6b_to_minilmv2_h384}"
CORPUS="${CORPUS:-data/train_set/merged_3_data_5k_each.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
# B-1: the number of relations an in-batch anchor is scored against.
SUPPORT_SIZE="${SUPPORT_SIZE:-$((BATCH_SIZE - 1))}"

# One graph, used three ways: as the teacher-support arm's candidate source, as
# the neighbourhood definition both teacher-informed samplers group by, and as
# the row bandwidths tau_i every arm's direct target is scored at.
GRAPH_SPEC="main|$CORPUS|"

# The minimal relational objective, shared by both relational arms.
MINIMAL="--relation_target direct --no_ambient --row_weight 0 --batch_size $BATCH_SIZE"

ARMS_SPEC=""
for sampler in random teacher_neighbor teacher_diverse; do
    ARMS_SPEC+="pointwise_$sampler|main|pointwise|--batch_sampler $sampler --batch_size $BATCH_SIZE
"
    ARMS_SPEC+="in_batch_$sampler|main|ggpkd|--batch_local $MINIMAL --batch_sampler $sampler
"
    ARMS_SPEC+="teacher_support_$sampler|main|ggpkd|$MINIMAL --diffusion_quota $SUPPORT_SIZE --batch_sampler $sampler
"
done

export GRAPH_SPEC ARMS_SPEC PAIR
CSV_OUT="${CSV_OUT:-$REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"

echo "E1 batch intervention: support size $SUPPORT_SIZE (= batch_size - 1)"
# `|| status=$?` rather than `; status=$?`: under `set -e` a failing call exits
# the script before the assignment runs, and the post-processing below -- which is
# still worth having when only some arms failed -- would never happen.
status=0
run_arms "$REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit $status
fi

# The exposure side of the same question, and it needs no training: how
# teacher-relevant each grouping's relations actually were. Run after the sweep
# so it reads the very graph the arms trained against.
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache/exp}"
CORPUS_KEY="$(basename "${CORPUS%.*}")"
"$PYTHON_BIN" "$SCRIPT_DIR/coverage.py" \
    --cache "$CACHE_ROOT/$PAIR/$CORPUS_KEY/teacher_train.pt" \
    --artifact "$CACHE_ROOT/$PAIR/$CORPUS_KEY/graph_main.pt" \
    --pair "$PAIR" \
    --batch-sizes "$BATCH_SIZE" \
    --quota "$SUPPORT_SIZE" \
    --out "$(dirname "$CSV_OUT")/coverage.csv"

echo
echo "E1 outputs:"
echo "  $CSV_OUT"
echo "  $(dirname "$CSV_OUT")/coverage.csv"
exit $status
