#!/usr/bin/env bash
# Stage 0 of experiments.md -- fix the operating point. 12 runs.
#
# graph_k = 200 was chosen when a row was mutual-filtered and cut to the
# neighbours holding 99% of the teacher's similarity, leaving ~67 real columns
# per anchor. An anchor now keeps all graph_k of them, so graph_k sets the encode
# cost per step directly, and through tau_i = (s(1) - s(k)) / log k it still sets
# how sharp a row is. The old sweep measured neither quantity.
#
# The winning setting is also the full-model reference for Stage 1, D and F, so
# these runs are reused rather than repeated.
#
#   GPUS=0,1,2,3 bash scripts/exp/stage0_graph_k.sh
#   DRY_RUN=1    bash scripts/exp/stage0_graph_k.sh
#
# Env: GPUS, SEEDS (42,43,44), PAIR, CORPUS, GRAPH_KS (25,50,100,200),
#      CACHE_ROOT, RESULT_BASE, CSV_OUT, DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="stage0_graph_k"
GRAPH_KS="${GRAPH_KS:-25,50,100,200}"

GRAPH_SPEC=""
ARMS_SPEC=""
IFS=',' read -r -a ks <<< "$GRAPH_KS"
for k in "${ks[@]}"; do
    [[ "$k" =~ ^[0-9]+$ ]] || { echo "graph_k must be an integer, got: $k" >&2; exit 2; }
    # graph_k belongs in GRAPH_SPEC: it changes the artifact, and the runner
    # forwards a graph's flags to every training run that uses it.
    GRAPH_SPEC+="k$k|$CORPUS|--graph_k $k
"
    ARMS_SPEC+="graph_k_$k|k$k|ggpkd|
"
done

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Stage 0: graph_k in {${GRAPH_KS//,/, }} on the shipped objective"
note "the arms carry no objective flags: this is the method as it ships"
note "read Avg, wall time, peak GPU, pool_fill_avg, train_encoded_texts_cum,"
note "and target_kl_uniform_r1 (a build warning under 0.05 is the flat-row failure)"
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "Rule: take the smallest graph_k within one seed-sd of the best Avg."
    echo "Small beats tied-and-large -- it is the whole cost argument."
    echo "Then run every later stage with GRAPH_K=<winner>."
    echo "Results: $CSV_OUT"
fi
exit $status
