#!/usr/bin/env bash
# Dispatch the whole ablation plan across GPUs, one run per GPU at a time.
#
# The warm-up is not optional. Every P0 script starts by training the full model,
# and the first such run is what builds the teacher-embedding cache and the base
# graph artifact. Fan out before either exists and several processes write the
# same two files at once. So: the full model is trained first, sequentially, and
# every later script's `run_full` then hits the .done guard and costs nothing.
#
#   bash scripts/ablation/run_all.sh                 # GPUs 0..N-1, all ablations
#   GPUS="0 1 2 3" bash scripts/ablation/run_all.sh  # pick the cards
#   PLAN="s1 s4" bash scripts/ablation/run_all.sh    # pick the ablations
#
# Each dispatched script owns one GPU and runs its arms in sequence, so the peak
# GPU count is min(#ablations, #GPUS) and nothing shares a card.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}"
GPUS="${GPUS:-0}"
read -ra GPU_LIST <<< "${GPUS}"
# P0 first, then P1: if the budget runs out, it runs out on the optional half.
PLAN="${PLAN:-s1 s2 s3 s4 k1 g1 n1 x1}"
read -ra PLAN_LIST <<< "${PLAN}"
LOG_DIR="${LOG_DIR:-runs/ablation/_dispatch}"
mkdir -p "${LOG_DIR}"

declare -A SCRIPT_OF=(
    [s1]=s1_support.sh [s2]=s2_scales.sh [s3]=s3_target.sh [s4]=s4_factorial.sh
    [k1]=k1_topk_t0.sh
    [g1]=g1_knn.sh     [n1]=n1_negatives.sh [x1]=x1_transfer.sh
)

echo "GPUs: ${GPU_LIST[*]}   plan: ${PLAN_LIST[*]}"
echo "--- warm-up: full model on GPU ${GPU_LIST[0]} (builds teacher cache + base graph) ---"
GPU="${GPU_LIST[0]}" bash "${SCRIPT_DIR}/full.sh" 2>&1 | tee "${LOG_DIR}/warmup.log"

pids=()
slot=0
for ablation in "${PLAN_LIST[@]}"; do
    script="${SCRIPT_OF[${ablation}]:-}"
    if [[ -z "${script}" ]]; then
        echo "unknown ablation '${ablation}' -- known: ${!SCRIPT_OF[*]}" >&2
        exit 2
    fi
    gpu="${GPU_LIST[$(( slot % ${#GPU_LIST[@]} ))]}"
    echo "dispatch ${ablation} -> GPU ${gpu}  (log ${LOG_DIR}/${ablation}.log)"
    GPU="${gpu}" bash "${SCRIPT_DIR}/${script}" > "${LOG_DIR}/${ablation}.log" 2>&1 &
    pids+=("$!:${ablation}")
    slot=$(( slot + 1 ))
    # More ablations than cards: wait for the current wave before starting the
    # next, so two scripts never land on the same GPU.
    if (( slot % ${#GPU_LIST[@]} == 0 )); then
        for entry in "${pids[@]}"; do
            wait "${entry%%:*}" || echo "FAILED: ${entry#*:} (see ${LOG_DIR}/${entry#*:}.log)"
        done
        pids=()
    fi
done
for entry in "${pids[@]}"; do
    wait "${entry%%:*}" || echo "FAILED: ${entry#*:} (see ${LOG_DIR}/${entry#*:}.log)"
done

echo
echo "--- analysis ---"
PY="${REPO_ROOT}/.venv/bin/python"; [[ -x "${PY}" ]] || PY=python3
"${PY}" scripts/ablation/replay_coverage.py \
    --artifact "cache/ggpkd/${PAIR_KEY:-qwen3_0_6b_to_minilmv2_h384}/graph_base.pt" \
    --out runs/ablation/analysis/coverage.csv
"${PY}" scripts/ablation/collect.py \
    --csv runs/ablation/analysis/results.csv \
    --latex latex/tables --hubness
"${PY}" scripts/ablation/figures.py --out-dir latex/figures
