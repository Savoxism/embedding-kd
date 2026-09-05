#!/usr/bin/env bash
# Run the complete paper suite across one or more GPUs.
#
# Examples:
#   GPUS="0 1 2" bash scripts/ablation/run_all.sh
#   PLAN="components radius sensitivity" GPUS="0 1" bash scripts/ablation/run_all.sh
#   SEEDS="42" GPUS="0" bash scripts/ablation/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}"
GPUS="${GPUS:-0}"
read -r -a GPU_LIST <<< "${GPUS}"
if (( ${#GPU_LIST[@]} == 0 )); then
    echo "No GPU selected. Set GPUS, for example GPUS=0." >&2
    exit 2
fi

PLAN="${PLAN:-main support components radius sensitivity}"
read -r -a PLAN_LIST <<< "${PLAN}"
DISPATCH_LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ggpkd-dispatch.XXXXXX")"

declare -A SCRIPT_OF=(
    [main]=main_results.sh
    [support]=s1_support.sh
    [components]=components.sh
    [radius]=radius.sh
    [sensitivity]=sensitivity.sh
)

for group in "${PLAN_LIST[@]}"; do
    if [[ -z "${SCRIPT_OF[${group}]:-}" ]]; then
        echo "Unknown group '${group}'. Choose from: main support components radius sensitivity" >&2
        exit 2
    fi
done

if [[ " ${PLAN_LIST[*]} " == *" main "* ]] && \
   [[ -n "${ABL_ROOT:-}${CACHE_ROOT:-}${LOG_ROOT:-}" ]]; then
    echo "ABL_ROOT/CACHE_ROOT/LOG_ROOT are single-pair overrides and cannot be used with PLAN containing main." >&2
    echo "Use RUNS_BASE/CACHE_BASE/LOG_BASE for the three-pair suite." >&2
    exit 2
fi

pids=()
names=()
cleanup_children() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${pids[@]:-}"; do
        wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup_children INT TERM EXIT

wait_wave() {
    local failed=0
    local slot
    for slot in "${!pids[@]}"; do
        if ! wait "${pids[${slot}]}"; then
            echo "[fail] ${names[${slot}]} (temporary log: ${DISPATCH_LOG_DIR}/${names[${slot}]}.log)" >&2
            failed=1
        fi
    done
    pids=()
    names=()
    return "${failed}"
}

# Warm up sequentially. All later jobs may read graph_base.pt concurrently, but
# no two processes are ever allowed to build it at the same time.
echo "[warmup] full model on GPU ${GPU_LIST[0]}"
GPU="${GPU_LIST[0]}" bash "${SCRIPT_DIR}/full.sh" 2>&1 | tee "${DISPATCH_LOG_DIR}/warmup.log"

for index in "${!PLAN_LIST[@]}"; do
    group="${PLAN_LIST[${index}]}"
    gpu="${GPU_LIST[$(( index % ${#GPU_LIST[@]} ))]}"
    script="${SCRIPT_OF[${group}]}"

    # Wait for the previous wave before reusing a GPU.
    if (( index > 0 && index % ${#GPU_LIST[@]} == 0 )); then
        wait_wave
    fi

    echo "[dispatch] ${group} -> GPU ${gpu}"
    GPU="${gpu}" bash "${SCRIPT_DIR}/${script}" \
        > "${DISPATCH_LOG_DIR}/${group}.log" 2>&1 &
    pids+=("$!")
    names+=("${group}")
done
wait_wave

echo "[done] all requested ablation groups completed"

# Successful runs have already been compacted to CSV. Dispatcher logs are also
# temporary and are retained only when the suite exits early with a failure.
rm -rf "${DISPATCH_LOG_DIR}"

trap - INT TERM EXIT
