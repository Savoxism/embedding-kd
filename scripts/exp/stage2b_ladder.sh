#!/usr/bin/env bash
# Stage 2B of experiments.md -- the comparison ladder. 12 runs per pair.
#
# One axis moves: which texts an anchor is compared against. The number of
# comparisons, the loss, the temperature, the optimiser and the seed are
# identical across arms -- a like-for-like swap, not a cap. The objective is
# deliberately minimal so that no part of our method can be credited with a gap
# that belongs to the choice of compared texts.
#
#   in_batch        the rest of the batch                      (the baseline)
#   corpus_uniform  the same number, drawn from the corpus      kills "just leave the batch"
#   student_knn     the student's own top-k                     kills "any neighbourhood would do"
#   teacher         the texts the teacher retrieved             ours
#
# `rewired` is deliberately absent: with a directed graph and no truncation every
# anchor has degree exactly graph_k, so it draws the same number from the same
# distribution as corpus_uniform. Verified identical for 400/400 anchors.
#
#   GRAPH_K=200 GPUS=0,1,2,3 bash scripts/exp/stage2b_ladder.sh
#   GRAPH_K=200 ARMS=student_knn bash scripts/exp/stage2b_ladder.sh   # re-run one arm
#
# Env: PAIRS (qwen3_0_6b_to_minilmv2_h384 only; the second pair is out of scope),
#      SUPPORT_SIZE (batch_size - 1), STUDENT_KNN=0 to drop that arm, ARMS, plus
#      the usual GPUS/SEEDS/DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="stage2b_ladder"
BATCH_SIZE="${BATCH_SIZE:-64}"
# The like-for-like count: every arm scores an anchor against this many texts,
# and the baseline has exactly batch_size - 1 available.
SUPPORT_SIZE="${SUPPORT_SIZE:-$((BATCH_SIZE - 1))}"
STUDENT_KNN="${STUDENT_KNN:-1}"
PAIRS="${PAIRS:-qwen3_0_6b_to_minilmv2_h384}"

MINIMAL="$(minimal_objective)"
BUDGET="--diffusion_quota $SUPPORT_SIZE"

echo "Stage 2B: comparison ladder at graph_k=$GRAPH_K, $SUPPORT_SIZE comparisons per anchor"
note "objective: $MINIMAL"
note "pairs: ${PAIRS//,/, }"
echo

overall=0
IFS=',' read -r -a pair_list <<< "$PAIRS"
for pair in "${pair_list[@]}"; do
    echo "##############################################"
    echo "# ladder -- pair $pair"
    echo "##############################################"

    graph_spec="teacher|$CORPUS|$(method_graph_flags)"
    arms="in_batch|teacher|ggpkd|--batch_local $MINIMAL
corpus_uniform|teacher|ggpkd|--support_policy corpus_uniform $MINIMAL $BUDGET
teacher|teacher|ggpkd|$MINIMAL $BUDGET
"

    if [[ "$STUDENT_KNN" == "1" ]]; then
        # Columns from the *base student's* kNN, ranked by the student, so the
        # quota takes the student's own top-$SUPPORT_SIZE. The artifact keeps the
        # teacher's row temperatures and MINIMAL reads targets off the teacher
        # bank, so which texts are compared is the only difference.
        #
        # The graph is built by the runner's own prepare step: its cache key
        # carries the neighbour source, so it can no longer load the teacher
        # graph under this name -- which is what the 2026-09-12 sweep did.
        graph_spec+="
student_knn|$CORPUS|$(method_graph_flags) --neighbor_source student"
        arms+="student_knn|student_knn|ggpkd|$MINIMAL $BUDGET
"
    fi

    csv_out="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
    mkdir -p "$(dirname "$csv_out")"
    PAIR="$pair" GRAPH_SPEC="$graph_spec" ARMS_SPEC="$arms" CSV_OUT="$csv_out" \
        RESULT_BASE="${RESULT_BASE:-$STAGE_REPO_ROOT/results/$EXPERIMENT/$pair}" \
        run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || overall=$?
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<'EOF'

Read it as a ladder, not a pair of numbers: the arm that compares against
unrelated texts should sit *on top of* the batch-local baseline, not between it
and ours. Report train_encoded_texts_cum beside Avg -- an off-graph draw
deduplicates far worse than a teacher draw, and that gap is part of the argument.
EOF
fi
exit $overall
