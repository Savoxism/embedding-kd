#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-$REPO_ROOT/results/talas}"
RUN_ROOT="$RESULT_BASE/$RUN_ID"
CACHE_ROOT="$RUN_ROOT/cache"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
RUNS_DIR="$RUN_ROOT/runs"
MANIFEST="$RUN_ROOT/manifest.tsv"

PAIRS=(
    qwen3_0_6b_to_minilmv2_h384
    bge_m3_to_minilmv2_h768
    qwen3_4b_to_bert_base
)
SEEDS=(42 43 44)
GPUS=(0 1 2 3 4 5 6 7)

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite existing TALAS run: $RUN_ROOT" >&2
    exit 2
fi
mkdir -p "$STATUS_DIR" "$LOG_DIR" "$RUNS_DIR"

finish_controller() {
    local code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$RUN_ROOT/controller.exit"
    exit "$code"
}
trap finish_controller EXIT

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Project virtual-environment Python is not executable: $PYTHON_BIN" >&2
    exit 2
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if (( GPU_COUNT < 8 )); then
    echo "TALAS paper run requires 8 visible GPUs, found $GPU_COUNT" >&2
    exit 2
fi

FREE_KB="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
if (( FREE_KB < 30 * 1024 * 1024 )); then
    echo "TALAS paper run requires at least 30 GiB free disk" >&2
    exit 2
fi

"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
from config.talas_config import TALAS_PAPER_PAIRS

model_ids = {
    value[name]
    for value in TALAS_PAPER_PAIRS.values()
    for name in ("teacher", "student")
}
for model_id in sorted(model_ids):
    snapshot_download(repo_id=model_id, local_files_only=True)
    print(f"offline model ready: {model_id}")
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

printf 'phase\tpair\tseed\tgpu\tpid\tstate\n' > "$MANIFEST"

echo "Preparing three pair-specific teacher caches..."
cache_pids=()
for index in "${!PAIRS[@]}"; do
    pair="${PAIRS[$index]}"
    gpu="${GPUS[$index]}"
    log="$LOG_DIR/cache.$pair.log"
    exit_file="$STATUS_DIR/cache.$pair.exit"
    (
        set +e
        PAIR_KEY="$pair" GPU="$gpu" CACHE_ROOT="$CACHE_ROOT" \
            RUN_DIR="$RUN_ROOT/cache_setup/$pair" PYTHON_BIN="$PYTHON_BIN" \
            bash "$SCRIPT_DIR/train_talas.sh" --prepare-cache >"$log" 2>&1
        code=$?
        printf '%s\n' "$code" > "$exit_file"
        exit "$code"
    ) &
    pid=$!
    cache_pids+=("$pid")
    printf 'cache\t%s\t-\t%s\t%s\trunning\n' "$pair" "$gpu" "$pid" >> "$MANIFEST"
done

cache_failed=0
for index in "${!cache_pids[@]}"; do
    if ! wait "${cache_pids[$index]}"; then
        cache_failed=1
    fi
done
if (( cache_failed != 0 )); then
    echo "At least one TALAS teacher-cache preparation failed" >&2
    exit 1
fi

tasks_pair=()
tasks_seed=()
for pair in "${PAIRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        tasks_pair+=("$pair")
        tasks_seed+=("$seed")
    done
done

declare -A PID_GPU=()
declare -A PID_TASK=()
active_pids=()
next_task=0
failed_runs=0

launch_training() {
    local task_index=$1
    local gpu=$2
    local pair="${tasks_pair[$task_index]}"
    local seed="${tasks_seed[$task_index]}"
    local task="$pair.seed_$seed"
    local run_dir="$RUNS_DIR/$pair/seed_$seed"
    local log="$LOG_DIR/$task.log"
    local exit_file="$STATUS_DIR/$task.exit"

    mkdir -p "$run_dir"
    (
        set +e
        PAIR_KEY="$pair" SEED="$seed" GPU="$gpu" CACHE_ROOT="$CACHE_ROOT" \
            RUN_DIR="$run_dir" WEIGHTS_DIR="$run_dir/weights" \
            PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/train_talas.sh" >"$log" 2>&1
        code=$?
        printf '%s\n' "$code" > "$exit_file"
        exit "$code"
    ) &
    local pid=$!
    active_pids+=("$pid")
    PID_GPU["$pid"]="$gpu"
    PID_TASK["$pid"]="$task"
    printf 'train\t%s\t%s\t%s\t%s\trunning\n' \
        "$pair" "$seed" "$gpu" "$pid" >> "$MANIFEST"
    echo "Launched $task on GPU $gpu (pid $pid)"
}

while (( next_task < ${#tasks_pair[@]} && next_task < ${#GPUS[@]} )); do
    launch_training "$next_task" "${GPUS[$next_task]}"
    ((next_task += 1))
done

while (( ${#active_pids[@]} > 0 )); do
    completed_pid=""
    set +e
    wait -n -p completed_pid "${active_pids[@]}"
    completed_status=$?
    set -e
    if [[ -z "$completed_pid" ]]; then
        echo "Could not identify completed TALAS process" >&2
        exit 1
    fi
    completed_gpu="${PID_GPU[$completed_pid]}"
    completed_task="${PID_TASK[$completed_pid]}"
    if (( completed_status != 0 )); then
        failed_runs=1
        echo "FAILED: $completed_task exited $completed_status" >&2
    else
        echo "Completed: $completed_task"
    fi

    remaining=()
    for pid in "${active_pids[@]}"; do
        if [[ "$pid" != "$completed_pid" ]]; then
            remaining+=("$pid")
        fi
    done
    active_pids=("${remaining[@]}")
    unset 'PID_GPU[$completed_pid]' 'PID_TASK[$completed_pid]'

    if (( next_task < ${#tasks_pair[@]} )); then
        launch_training "$next_task" "$completed_gpu"
        ((next_task += 1))
    fi
done

if (( failed_runs != 0 )); then
    echo "At least one TALAS training run failed; refusing to aggregate" >&2
    exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_talas.py" "$RUN_ROOT" | tee "$RUN_ROOT/summary.txt"
echo "TALAS paper run complete: $RUN_ROOT"
