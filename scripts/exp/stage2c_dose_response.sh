#!/usr/bin/env bash
# Stage 2C of experiments.md -- dose-response on the count. 27 runs.
#
# A gives the count analytically, B gives the outcome. C moves the count two
# different ways and checks the score follows, which is the strongest form the
# claim can take.
#
#   C.1 (15 runs)  raise the count by composing batches from one teacher
#                  neighbourhood. pointwise / in_batch / full GGPKD, samplers
#                  random and teacher_neighbor. `full @ random` is Stage 0's
#                  winner and is not repeated.
#
#   C.2 (12 runs)  raise the count by enlarging the batch. in_batch and full
#                  GGPKD at B in {256, 1024}; B=64 comes from Stage 2B.
#                  This is the arm that can hurt us -- in the old runs the
#                  baseline climbed with batch size and at B=256 passed the
#                  teacher arm. Run it before a reviewer asks.
#
# Both halves are run on the SHIPPED objective, not the minimal one: the claim
# that matters is about the method we publish, and C.1 is where the calibration
# term's batch-dependence would surface if it is real.
#
#   GRAPH_K=50 GPUS=0,1,2,3 bash scripts/exp/stage2c_dose_response.sh
#   HALF=c1    GRAPH_K=50 bash scripts/exp/stage2c_dose_response.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

HALF="${HALF:-both}"
BATCH_SIZES="${BATCH_SIZES:-256,1024}"
GRAPH_SPEC_BASE="main|$CORPUS|$(method_graph_flags)"
overall=0

run_half() {
    local experiment="$1" arms="$2"
    local csv="$STAGE_REPO_ROOT/runs/$experiment/results.csv"
    mkdir -p "$(dirname "$csv")"
    GRAPH_SPEC="$GRAPH_SPEC_BASE" ARMS_SPEC="$arms" CSV_OUT="$csv" \
        run_arms "$STAGE_REPO_ROOT" "$experiment" || return $?
}

# ----------------------------------------------------------------- C.1 ------
if [[ "$HALF" == "both" || "$HALF" == "c1" ]]; then
    echo "Stage 2C.1: batch composition at graph_k=$GRAPH_K"
    note "pointwise is the measurement floor: its objective cannot read batch"
    note "membership, so its movement across samplers IS the noise level the"
    note "other arms are read against -- difference-in-differences, not raw spread"
    note "'full @ random' is Stage 0's winner and is not repeated here"
    echo
    # teacher_diverse is dropped: it sat within noise of random and bought nothing.
    C1_ARMS="pointwise_random|main|pointwise|--batch_sampler random
pointwise_neighbor|main|pointwise|--batch_sampler teacher_neighbor
in_batch_random|main|ggpkd|--batch_local --relation_target direct --batch_sampler random
in_batch_neighbor|main|ggpkd|--batch_local --relation_target direct --batch_sampler teacher_neighbor
full_neighbor|main|ggpkd|--batch_sampler teacher_neighbor
"
    run_half "stage2c1_composition" "$C1_ARMS" || overall=$?
fi

# ----------------------------------------------------------------- C.2 ------
if [[ "$HALF" == "both" || "$HALF" == "c2" ]]; then
    echo
    echo "Stage 2C.2: batch size at graph_k=$GRAPH_K, B in {${BATCH_SIZES//,/, }}"
    note "A predicts the baseline's count rises linearly in B and needs"
    note "B ~ 4270 to reach ours, so it should climb and not arrive"
    warn "this is the arm that can hurt us: at N=48k the old baseline went"
    warn "  71.85 (B=16) -> 72.82 (B=64) -> 73.20 (B=256) and passed the"
    warn "  teacher arm's 72.83. Counts 0.06 -> 0.26 -> 1.06 explain the climb,"
    warn "  not the overtake. If it survives, state the claim per unit of"
    warn "  encoder work rather than per step."
    echo
    C2_ARMS=""
    IFS=',' read -r -a sizes <<< "$BATCH_SIZES"
    for b in "${sizes[@]}"; do
        [[ "$b" =~ ^[0-9]+$ ]] || { echo "batch size must be an integer, got: $b" >&2; exit 2; }
        C2_ARMS+="in_batch_b$b|main|ggpkd|--batch_local --relation_target direct --batch_size $b
full_b$b|main|ggpkd|--batch_size $b
"
    done
    run_half "stage2c2_batch_size" "$C2_ARMS" || overall=$?
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<'EOF'

Both halves must report the MEASURED count, not only the score, so the result is
a curve: Avg against informative comparisons per anchor, with both ways of buying
the count landing on it and ours at the right-hand end for a fraction of the
encoder cost. scripts/exp/coverage.py computes the count per arm.
EOF
fi
exit $overall
