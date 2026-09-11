#!/usr/bin/env bash
# Ambient-replacement gate for the canonical Qwen-0.6B -> MiniLM-H384 pair.
#
# Default new runs (3 arms x 3 seeds = 9):
#   none_row1             Is the existing auxiliary row loss sufficient alone?
#   none_row0             Graph-row objective with no calibration.
#   fixed_reference_row0  Replace batch-pool Ambient with fixed-reference KL.
#
# The already-measured pool_row0 arm is omitted by default. To reproduce the
# complete isolated comparison, include it explicitly:
#   ARMS=pool_row0,none_row0,fixed_reference_row0 \
#     bash scripts/ggpkd/run_calibration_gate.sh
#
# Useful overrides:
#   GPUS=0,1,2 SEEDS=42,43,44 REFERENCE_SIZE=200 \
#     bash scripts/ggpkd/run_calibration_gate.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
PAIR_KEY="${PAIR_KEY:-qwen3_0_6b_to_minilmv2_h384}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-$REPO_ROOT/results/ggpkd_calibration}"
RUN_ROOT="$RESULT_BASE/$RUN_ID"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache/ggpkd}"
REFERENCE_SIZE="${REFERENCE_SIZE:-200}"
GRAPH_K="${GRAPH_K:-200}"
TRUNCATION_TOLERANCE="${TRUNCATION_TOLERANCE:-0.01}"
DIFFUSION_SCALES="${DIFFUSION_SCALES:-1}"

IFS=',' read -r -a ARMS_LIST <<< \
    "${ARMS:-none_row1,none_row0,fixed_reference_row0}"
IFS=',' read -r -a SEEDS_LIST <<< "${SEEDS:-42,43,44}"

if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "$GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(
        nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '
    )
else
    echo "No nvidia-smi and no GPUS override; set GPUS=0 to select one device" >&2
    exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python is not executable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! "$REFERENCE_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "REFERENCE_SIZE must be a positive integer, got: $REFERENCE_SIZE" >&2
    exit 2
fi
if (( ${#GPU_LIST[@]} == 0 )); then
    echo "No GPUs selected" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite existing run: $RUN_ROOT" >&2
    exit 2
fi

for arm in "${ARMS_LIST[@]}"; do
    case "$arm" in
        pool_row0|none_row1|none_row0|fixed_reference_row0|fixed_reference_row1)
            ;;
        *)
            echo "Unknown arm: $arm" >&2
            echo "Expected pool_row0, none_row1, none_row0, fixed_reference_row0, or fixed_reference_row1" >&2
            exit 2
            ;;
    esac
done

STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
RUNS_DIR="$RUN_ROOT/runs"
MANIFEST="$RUN_ROOT/manifest.tsv"
mkdir -p "$STATUS_DIR" "$LOG_DIR" "$RUNS_DIR" "$CACHE_ROOT/$PAIR_KEY"

finish_controller() {
    local code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$RUN_ROOT/controller.exit"
    exit "$code"
}
trap finish_controller EXIT

printf 'arm\tseed\tgpu\tpid\tstate\n' > "$MANIFEST"
{
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'pair\t%s\n' "$PAIR_KEY"
    printf 'arms\t%s\n' "${ARMS_LIST[*]}"
    printf 'seeds\t%s\n' "${SEEDS_LIST[*]}"
    printf 'gpus\t%s\n' "${GPU_LIST[*]}"
    printf 'reference_size\t%s\n' "$REFERENCE_SIZE"
    printf 'graph_k\t%s\n' "$GRAPH_K"
    printf 'truncation_tolerance\t%s\n' "$TRUNCATION_TOLERANCE"
    printf 'diffusion_scales\t%s\n' "$DIFFUSION_SCALES"
    printf 'commit\t%s\n' "$(git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$RUN_ROOT/run_config.tsv"

export TOKENIZERS_PARALLELISM=false
TEACHER_CACHE="$CACHE_ROOT/$PAIR_KEY/teacher_train.pt"
GRAPH_CACHE="$CACHE_ROOT/$PAIR_KEY/graph.pt"

echo "Calibration gate: $RUN_ID"
echo "  pair:      $PAIR_KEY"
echo "  arms:      ${ARMS_LIST[*]}"
echo "  seeds:     ${SEEDS_LIST[*]}"
echo "  references:$REFERENCE_SIZE"
echo "  output:    $RUN_ROOT"

# Build the seed-independent teacher cache and graph once before concurrent
# training starts. Calibration mode does not alter either artifact.
CACHE_LOG="$LOG_DIR/cache.log"
echo "Preparing shared teacher/graph cache (log: $CACHE_LOG)"
set +e
PAIR_KEY="$PAIR_KEY" GPU="${GPU_LIST[0]}" PYTHON_BIN="$PYTHON_BIN" \
    CACHE_PATH="$TEACHER_CACHE" \
    GGPKD_CACHE_PATH="$GRAPH_CACHE" \
    GGPKD_LOG_DIR="$RUN_ROOT/graph_logs" \
    SAVE_DIR="$RUN_ROOT/cache_setup" \
    bash "$SCRIPT_DIR/train.sh" \
        --prepare_cache_only \
        --calibration_mode none \
        --row_weight 0 \
        --graph_k "$GRAPH_K" \
        --truncation_tolerance "$TRUNCATION_TOLERANCE" \
        --diffusion_scales "$DIFFUSION_SCALES" >"$CACHE_LOG" 2>&1
CACHE_CODE=$?
set -e
if (( CACHE_CODE != 0 )); then
    echo "Cache preparation failed with exit $CACHE_CODE; see $CACHE_LOG" >&2
    exit 1
fi

TASK_ARMS=()
TASK_SEEDS=()
for arm in "${ARMS_LIST[@]}"; do
    for seed in "${SEEDS_LIST[@]}"; do
        TASK_ARMS+=("$arm")
        TASK_SEEDS+=("$seed")
    done
done

declare -A PID_GPU=()
declare -A PID_ARM=()
declare -A PID_SEED=()
ACTIVE_PIDS=()
NEXT_TASK=0
FAILED=0

launch_task() {
    local task_index=$1
    local gpu=$2
    local arm="${TASK_ARMS[$task_index]}"
    local seed="${TASK_SEEDS[$task_index]}"
    local run_dir="$RUNS_DIR/$arm/seed_$seed"
    local log="$LOG_DIR/$arm.seed_$seed.log"
    local exit_file="$STATUS_DIR/$arm.seed_$seed.exit"
    local -a arm_args

    case "$arm" in
        pool_row0)
            arm_args=(--calibration_mode pool --row_weight 0)
            ;;
        none_row1)
            arm_args=(--calibration_mode none --row_weight 1)
            ;;
        none_row0)
            arm_args=(--calibration_mode none --row_weight 0)
            ;;
        fixed_reference_row0)
            arm_args=(
                --calibration_mode fixed_reference
                --reference_size "$REFERENCE_SIZE"
                --row_weight 0
            )
            ;;
        fixed_reference_row1)
            arm_args=(
                --calibration_mode fixed_reference
                --reference_size "$REFERENCE_SIZE"
                --row_weight 1
            )
            ;;
    esac

    mkdir -p "$run_dir"
    (
        set +e
        PAIR_KEY="$PAIR_KEY" SEED="$seed" GPU="$gpu" PYTHON_BIN="$PYTHON_BIN" \
            CACHE_PATH="$TEACHER_CACHE" \
            GGPKD_CACHE_PATH="$GRAPH_CACHE" \
            GGPKD_LOG_DIR="$run_dir/graph_logs" \
            SAVE_DIR="$run_dir" WEIGHTS_DIR="$run_dir/weights" \
            bash "$SCRIPT_DIR/train.sh" \
                --final_weights_only \
                --graph_k "$GRAPH_K" \
                --truncation_tolerance "$TRUNCATION_TOLERANCE" \
                --diffusion_scales "$DIFFUSION_SCALES" \
                "${arm_args[@]}" >"$log" 2>&1
        code=$?
        printf '%s\n' "$code" > "$exit_file"
        exit "$code"
    ) &
    local pid=$!
    ACTIVE_PIDS+=("$pid")
    PID_GPU["$pid"]="$gpu"
    PID_ARM["$pid"]="$arm"
    PID_SEED["$pid"]="$seed"
    printf '%s\t%s\t%s\t%s\trunning\n' "$arm" "$seed" "$gpu" "$pid" >> "$MANIFEST"
    echo "Launched $arm seed=$seed on GPU $gpu (pid $pid)"
}

while (( NEXT_TASK < ${#TASK_ARMS[@]} && NEXT_TASK < ${#GPU_LIST[@]} )); do
    launch_task "$NEXT_TASK" "${GPU_LIST[$NEXT_TASK]}"
    ((NEXT_TASK += 1))
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
    COMPLETED_PID=""
    set +e
    wait -n -p COMPLETED_PID "${ACTIVE_PIDS[@]}"
    COMPLETED_CODE=$?
    set -e
    if [[ -z "$COMPLETED_PID" ]]; then
        echo "Could not identify the completed process" >&2
        exit 1
    fi

    COMPLETED_GPU="${PID_GPU[$COMPLETED_PID]}"
    COMPLETED_ARM="${PID_ARM[$COMPLETED_PID]}"
    COMPLETED_SEED="${PID_SEED[$COMPLETED_PID]}"
    printf '%s\t%s\t%s\t%s\texit_%s\n' \
        "$COMPLETED_ARM" "$COMPLETED_SEED" "$COMPLETED_GPU" \
        "$COMPLETED_PID" "$COMPLETED_CODE" >> "$MANIFEST"

    if (( COMPLETED_CODE != 0 )); then
        FAILED=1
        echo "FAILED: $COMPLETED_ARM seed=$COMPLETED_SEED; see $LOG_DIR/$COMPLETED_ARM.seed_$COMPLETED_SEED.log" >&2
    else
        echo "Completed: $COMPLETED_ARM seed=$COMPLETED_SEED"
    fi

    REMAINING=()
    for pid in "${ACTIVE_PIDS[@]}"; do
        if [[ "$pid" != "$COMPLETED_PID" ]]; then
            REMAINING+=("$pid")
        fi
    done
    ACTIVE_PIDS=("${REMAINING[@]}")
    unset 'PID_GPU[$COMPLETED_PID]' 'PID_ARM[$COMPLETED_PID]' \
        'PID_SEED[$COMPLETED_PID]'

    if (( NEXT_TASK < ${#TASK_ARMS[@]} )); then
        launch_task "$NEXT_TASK" "$COMPLETED_GPU"
        ((NEXT_TASK += 1))
    fi
done

if (( FAILED != 0 )); then
    echo "At least one calibration-gate run failed" >&2
    exit 1
fi

echo "Calibration gate complete: $RUN_ROOT"
echo "Each run contains metrics.json, run_stats.json, and final student weights."
