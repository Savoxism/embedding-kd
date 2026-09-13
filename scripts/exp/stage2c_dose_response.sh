#!/usr/bin/env bash
# Stage 2C of experiments.md -- dose-response on the count. 15 runs.
#
# A gives the count analytically, B gives the outcome. C raises the count for a
# batch-local objective by composing batches from one teacher neighbourhood and
# checks the score follows. pointwise / in_batch / full GGPKD, samplers random and
# teacher_neighbor. `full @ random` is Stage 0's winner and is not repeated.
#
# The batch-size half (C.2) was cut: no single setting matches optimizer steps,
# passes over the data and learning rate across B at once, so it cannot isolate
# the count. See "Cut" in experiments.md.
#
# Run on the SHIPPED objective, not the minimal one: the claim that matters is
# about the method we publish, and this is where the calibration term's
# batch-dependence would surface if it is real.
#
#   GRAPH_K=200 GPUS=0,1,2,3 bash scripts/exp/stage2c_dose_response.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="stage2c1_composition"
GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)"

echo "Stage 2C: batch composition at graph_k=$GRAPH_K"
note "pointwise is the measurement floor: its objective cannot read batch"
note "membership, so its movement across samplers IS the noise level the"
note "other arms are read against -- difference-in-differences, not raw spread"
note "'full @ random' is Stage 0's winner and is not repeated here"
echo
# teacher_diverse is dropped: it sat within noise of random and bought nothing.
ARMS_SPEC="pointwise_random|main|pointwise|--batch_sampler random
pointwise_neighbor|main|pointwise|--batch_sampler teacher_neighbor
in_batch_random|main|ggpkd|--batch_local --relation_target direct --batch_sampler random
in_batch_neighbor|main|ggpkd|--batch_local --relation_target direct --batch_sampler teacher_neighbor
full_neighbor|main|ggpkd|--batch_sampler teacher_neighbor
"

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<'EOF'

Report the MEASURED count beside the score: in_batch under teacher_neighbor
batching should move exactly because its batches now hold related texts.
scripts/exp/coverage.py computes the count per arm.
EOF
fi
exit $status
