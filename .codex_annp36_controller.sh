#!/usr/bin/env bash
set -euo pipefail

cd /home/annp36/work/heatgeo

export EXPERIMENT_KEY=paper_r1_v2
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

log_dir="runs/ablation/_dispatch/paper_r1_v2_annp36"
mkdir -p "${log_dir}"
controller_log="${log_dir}/controller.log"
exec > >(tee -a "${controller_log}") 2>&1

echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"

pids=()
names=()

stop_children() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill -- "-${pid}" 2>/dev/null || true
    done
    for pid in "${pids[@]:-}"; do
        wait "${pid}" 2>/dev/null || true
    done
}
trap stop_children INT TERM EXIT

start_worker() {
    local name="$1"
    local gpu="$2"
    local seeds="$3"
    local script="$4"
    echo "[dispatch] ${name} -> GPU ${gpu}, seeds ${seeds}"
    setsid env GPU="${gpu}" SEEDS="${seeds}" \
        bash "scripts/ablation/${script}" \
        > "${log_dir}/${name}.log" 2>&1 &
    pids+=("$!")
    names+=("${name}")
}

# Sensitivity is the largest group, so give one GPU to each seed. The other
# groups are split by seed across the remaining five GPUs.
start_worker sensitivity_s42 0 "42" sensitivity.sh
start_worker sensitivity_s43 1 "43" sensitivity.sh
start_worker sensitivity_s44 2 "44" sensitivity.sh
start_worker support_s42_s43 3 "42 43" s1_support.sh
start_worker components_s42_s43 4 "42 43" components.sh
start_worker radius_all 5 "42 43 44" radius.sh
start_worker support_s44 6 "44" s1_support.sh
start_worker components_s44 7 "44" components.sh

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[${index}]}"; then
        echo "[worker-done] ${names[${index}]}"
    else
        echo "[worker-fail] ${names[${index}]}" >&2
        failed=1
    fi
done
pids=()
names=()

if (( failed )); then
    echo "[retry] at least one worker failed; rerunning canonical coverage on GPU 0"
    for script in s1_support.sh components.sh radius.sh sensitivity.sh; do
        GPU=0 SEEDS="42 43 44" bash "scripts/ablation/${script}"
    done
fi

result_root="runs/ablation/qwen3_0_6b_to_minilmv2_h384/${EXPERIMENT_KEY}"
result_count="$(find "${result_root}" -name result.csv -type f | wc -l)"
if [[ "${result_count}" -ne 60 ]]; then
    echo "[fail] expected 60 compact results, found ${result_count}" >&2
    exit 1
fi

echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ) results=${result_count}"
trap - INT TERM EXIT
