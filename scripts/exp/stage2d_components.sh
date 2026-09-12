#!/usr/bin/env bash
# Stage 2D of experiments.md -- component ablation on the shipped objective.
# 9 runs. The single-term arms come from Stage 1 and the full model from Stage 0;
# only these three are new.
#
#   no_both      both terms off at once
#   knn_mutual   --knn_mode mutual
#   truncated    --truncation_tolerance 0.01
#
# The last two defaults were changed on a structural argument and have never been
# measured on the full objective. The ladder put mutual 0.36 below directed on the
# minimal one, which is suggestive and not enough to change a default on.
#
# Stage 2E scores these checkpoints post-hoc, which is why the held-out split is
# applied here: set HOLDOUT_FRAC=0 for numbers comparable to the main table
# instead, at the cost of E.
#
#   GRAPH_K=50 GPUS=0,1,2,3 bash scripts/exp/stage2d_components.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="stage2d_components"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.2}"
HOLDOUT_SEED="${HOLDOUT_SEED:-12345}"
HOLDOUT_FLAGS=""
if [[ "$HOLDOUT_FRAC" != "0" && "$HOLDOUT_FRAC" != "0.0" ]]; then
    # Separate from the training seed on purpose: the split must be identical
    # across seeds and arms, or E is not scoring the same withheld pairs.
    HOLDOUT_FLAGS="--holdout_edge_frac $HOLDOUT_FRAC --holdout_seed $HOLDOUT_SEED"
fi

# knn_mode and truncation_tolerance change the artifact, so each needs its own
# graph key; the runner forwards a graph's flags to the runs that use it.
GRAPH_SPEC="main|$CORPUS|$(method_graph_flags) $HOLDOUT_FLAGS
mutual|$CORPUS|$(method_graph_flags) --knn_mode mutual $HOLDOUT_FLAGS
truncated|$CORPUS|$(method_graph_flags) --truncation_tolerance 0.01 $HOLDOUT_FLAGS"

ARMS_SPEC="no_both|main|ggpkd|--row_weight 0 --calibration_mode none
knn_mutual|mutual|ggpkd|
truncated|truncated|ggpkd|
"

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Stage 2D: component ablation at graph_k=$GRAPH_K"
if [[ -n "$HOLDOUT_FLAGS" ]]; then
    note "withholding $HOLDOUT_FRAC of the teacher's pairs so Stage 2E can score"
    note "these checkpoints; absolute numbers therefore do not line up with the"
    note "main table -- the comparison that matters is between these arms"
else
    note "HOLDOUT_FRAC=0: comparable to the main table, but Stage 2E cannot score it"
fi
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<EOF

Compare against: Stage 0's graph_k=$GRAPH_K run (full) and Stage 1's single-term
arms. If knn_mutual or truncated wins on the full objective, the default that was
changed on an argument was changed wrongly -- say so and change it back.

Then score the held-out pairs:
    bash scripts/exp/exp3_heldout_geometry.sh
Note that probe is currently broken (pair_order = 1.0000 for every arm); see
experiments.md Stage 2E before trusting it.

Results: $CSV_OUT
EOF
fi
exit $status
