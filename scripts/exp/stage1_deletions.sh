#!/usr/bin/env bash
# Stage 1 of experiments.md -- three deletion tests. 12 runs.
#
# Each asks whether a component earns its place, and each "no" removes a term, a
# hyperparameter, a paragraph of method and a later sweep. Run before anything
# expensive. The full-model reference is Stage 0's winning run, not repeated here.
#
#   1.1  --row_weight 0        does L_row earn a hyperparameter?
#   1.1  random row centers    ... and is its gain about the extra anchors being
#                              teacher-chosen, or is it just a regulariser?
#   1.2  --calibration_mode none   does the calibration term earn its place?
#   1.3  uniform target        does tau_i earn the per-row temperature machinery?
#
# Two of these need patches that do not exist yet (see the patch table in
# experiments.md). They are skipped with a message rather than failing mid-sweep.
#
#   GRAPH_K=50 GPUS=0,1,2,3 bash scripts/exp/stage1_deletions.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="stage1_deletions"
GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)"

# 1.1a and 1.2 need no new code.
ARMS_SPEC="row_weight_0|main|ggpkd|--row_weight 0
no_calibration|main|ggpkd|--calibration_mode none
"

# 1.1b -- random row centers. The row term supervises each extra anchor against
# the texts the teacher retrieved for it; this arm draws those columns at random
# instead. It is the arm that decides whether the term is an eligibility effect
# or a regulariser, which is why it matters more than --row_weight 0.
if has_flag "--row_centers"; then
    ARMS_SPEC+="row_centers_random|main|ggpkd|--row_centers random
"
else
    skip_arm "row_centers_random" "--row_centers" \
        "the row loss always uses N_j intersect P_B; add a switch that draws the
  extra anchors' columns uniformly from the pool instead."
fi

# 1.3 -- uniform target. Same texts compared, same columns, same temperature
# plumbing; only the target values change, from the teacher's similarity profile
# over the neighbourhood to treating every neighbour in it as equally similar.
if _stage_help | grep -q "relation_target.*uniform"; then
    ARMS_SPEC+="uniform_target|main|ggpkd|--relation_target uniform
"
else
    skip_arm "uniform_target" "--relation_target uniform" \
        "add a relation_target whose target is uniform over the retrieved
  neighbours. If it ties the transition row, tau_i and the whole temperature
  section come out of the paper."
fi

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Stage 1: deletion tests at graph_k=$GRAPH_K"
report_missing
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<EOF

Rules (compare against Stage 0's graph_k=$GRAPH_K run):
  row_weight_0        within noise of full            -> L_row is not worth a knob
  row_centers_random  gains as much as the real term  -> it is a regulariser;
                                                         delete L_row, the method
                                                         drops to one knob
  no_calibration      loses <= 0.2                    -> delete the term and the
                                                         story's one contradiction
  uniform_target      ties the transition row         -> delete tau_i and the
                                                         temperature section
Results: $CSV_OUT
EOF
fi
exit $status
