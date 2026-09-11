#!/usr/bin/env bash
# Full calibration x auxiliary-row factorial for the canonical
# Qwen-0.6B -> MiniLM-H384 pair.
#
# Default runs (4 calibration modes x 2 row settings x 3 seeds = 24):
#   none_row{0,1}             Graph only, optionally plus L_row.
#   pool_row{0,1}             Historical batch-pool KL calibration.
#   fixed_reference_row{0,1}  Fixed-domain KL calibration.
#   fixed_cosine_row{0,1}     Fixed-domain raw-cosine MSE calibration.
#
# Every active top-level loss has coefficient 1.0. All arms share the same
# teacher cache, graph, data, training schedule and seed set. The fixed-domain
# arms also share exactly the same deterministic reference set.
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
RESULT_BASE="${RESULT_BASE:-$REPO_ROOT/runs/ggpkd_calibration}"
RUN_ROOT="$RESULT_BASE/$RUN_ID"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache/ggpkd}"
REFERENCE_SIZE="${REFERENCE_SIZE:-200}"
GRAPH_K="${GRAPH_K:-200}"
TRUNCATION_TOLERANCE="${TRUNCATION_TOLERANCE:-0.01}"
DIFFUSION_SCALES="${DIFFUSION_SCALES:-1}"

IFS=',' read -r -a ARMS_LIST <<< \
    "${ARMS:-none_row0,none_row1,pool_row0,pool_row1,fixed_reference_row0,fixed_reference_row1,fixed_cosine_row0,fixed_cosine_row1}"
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
if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to verify every completed run's resolved config" >&2
    exit 2
fi

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_STATUS="$(git status --porcelain --untracked-files=all 2>/dev/null || true)"
if [[ -n "$GIT_STATUS" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
    echo "Refusing a non-reproducible sweep from a dirty worktree." >&2
    echo "Commit the calibration changes first, or set ALLOW_DIRTY=1 explicitly." >&2
    exit 2
fi
if [[ ! "$REFERENCE_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "REFERENCE_SIZE must be a positive integer, got: $REFERENCE_SIZE" >&2
    exit 2
fi
if [[ ! "$GRAPH_K" =~ ^[1-9][0-9]*$ ]]; then
    echo "GRAPH_K must be a positive integer, got: $GRAPH_K" >&2
    exit 2
fi
if ! jq -en --arg value "$TRUNCATION_TOLERANCE" \
    '($value | tonumber) as $number | $number >= 0' >/dev/null 2>&1; then
    echo "TRUNCATION_TOLERANCE must be a non-negative number, got: $TRUNCATION_TOLERANCE" >&2
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

declare -A SEEN_ARMS=()
for arm in "${ARMS_LIST[@]}"; do
    case "$arm" in
        none_row0|none_row1|pool_row0|pool_row1|fixed_reference_row0|fixed_reference_row1|fixed_cosine_row0|fixed_cosine_row1)
            ;;
        *)
            echo "Unknown arm: $arm" >&2
            echo "Expected CALIBRATION_rowN with CALIBRATION in {none,pool,fixed_reference,fixed_cosine} and N in {0,1}" >&2
            exit 2
            ;;
    esac
    if [[ -n "${SEEN_ARMS[$arm]:-}" ]]; then
        echo "Duplicate arm would overwrite the same run directory: $arm" >&2
        exit 2
    fi
    SEEN_ARMS["$arm"]=1
done

declare -A SEEN_SEEDS=()
for seed in "${SEEDS_LIST[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Every seed must be a non-negative integer, got: $seed" >&2
        exit 2
    fi
    if [[ -n "${SEEN_SEEDS[$seed]:-}" ]]; then
        echo "Duplicate seed would overwrite the same run directory: $seed" >&2
        exit 2
    fi
    SEEN_SEEDS["$seed"]=1
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
    printf 'graph_weight\t1.0\n'
    printf 'calibration_weight\t1.0\n'
    printf 'active_row_weight\t1.0\n'
    printf 'commit\t%s\n' "$GIT_COMMIT"
    printf 'dirty_at_start\t%s\n' "$([[ -n "$GIT_STATUS" ]] && echo true || echo false)"
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

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Running objective/domain fairness preflight tests"
    "$PYTHON_BIN" -m pytest -q \
        tests/test_fixed_reference_calibration.py \
        tests/test_new_experiment_config.py::test_ggpkd_relational_loss_reports_semantic_decomposition \
        tests/test_ablation_arms.py::test_ambient_only_gives_the_ambient_scale_the_whole_weight
fi

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
EXPECTED_REFERENCE_FINGERPRINT=""

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
        none_row0)
            arm_args=(--calibration_mode none --row_weight 0)
            ;;
        none_row1)
            arm_args=(--calibration_mode none --row_weight 1)
            ;;
        pool_row0)
            arm_args=(--calibration_mode pool --row_weight 0)
            ;;
        pool_row1)
            arm_args=(--calibration_mode pool --row_weight 1)
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
        fixed_cosine_row0)
            arm_args=(
                --calibration_mode fixed_cosine
                --reference_size "$REFERENCE_SIZE"
                --row_weight 0
            )
            ;;
        fixed_cosine_row1)
            arm_args=(
                --calibration_mode fixed_cosine
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
        EXPECTED_MODE="${COMPLETED_ARM%_row*}"
        EXPECTED_ROW="${COMPLETED_ARM##*_row}"
        EXPECTED_CAL_WEIGHT=1
        if [[ "$EXPECTED_MODE" == "none" ]]; then
            EXPECTED_CAL_WEIGHT=0
        fi
        RUN_JSON="$RUNS_DIR/$COMPLETED_ARM/seed_$COMPLETED_SEED/run.json"
        METRICS_JSONL="$RUNS_DIR/$COMPLETED_ARM/seed_$COMPLETED_SEED/metrics.jsonl"
        REFERENCE_OK=1
        if [[ "$EXPECTED_MODE" == "fixed_reference" || "$EXPECTED_MODE" == "fixed_cosine" ]]; then
            RUN_REFERENCE_FINGERPRINT="$(jq -r '.config.reference_fingerprint // empty' "$RUN_JSON")"
            if [[ -z "$RUN_REFERENCE_FINGERPRINT" ]]; then
                REFERENCE_OK=0
            elif [[ -z "$EXPECTED_REFERENCE_FINGERPRINT" ]]; then
                EXPECTED_REFERENCE_FINGERPRINT="$RUN_REFERENCE_FINGERPRINT"
            elif [[ "$RUN_REFERENCE_FINGERPRINT" != "$EXPECTED_REFERENCE_FINGERPRINT" ]]; then
                REFERENCE_OK=0
            fi
        elif ! jq -e '.config.reference_fingerprint == null' "$RUN_JSON" >/dev/null; then
            REFERENCE_OK=0
        fi

        if (( REFERENCE_OK == 1 )) && jq -e \
            --arg mode "$EXPECTED_MODE" \
            --arg commit "$GIT_COMMIT" \
            --arg scales "$DIFFUSION_SCALES" \
            --argjson tolerance "$TRUNCATION_TOLERANCE" \
            --argjson row "$EXPECTED_ROW" \
            --argjson reference_size "$REFERENCE_SIZE" \
            --argjson graph_k "$GRAPH_K" \
            '.config.calibration_mode == $mode
             and .config.row_weight == $row
             and .config.reference_size == $reference_size
             and .config.graph_k == $graph_k
             and .config.truncation_tolerance == $tolerance
             and .config.relation_target == "diffusion"
             and ((.config.diffusion_scales | map(tostring) | join(",")) == $scales)
             and .config.hard_neg_k == 0
             and .config.random_neg_k == 0
             and .git.sha == $commit' "$RUN_JSON" >/dev/null \
            && jq -e \
                --argjson row "$EXPECTED_ROW" \
                --argjson calibration "$EXPECTED_CAL_WEIGHT" \
                -s \
                '([.[] | select(.train != null)] | last | .train) as $train
                 | $train != null
                   and $train.weight_graph == 1
                   and $train.weight_cal == $calibration
                   and $train.weight_row == $row
                   and ((($train.loss_rel
                           - ($train.loss_graph + $train.loss_cal)) | fabs) < 1e-5)
                   and ((($train.loss_total
                           - ($train.loss_rel + $row * $train.loss_row)) | fabs) < 1e-5)' \
                "$METRICS_JSONL" >/dev/null; then
            echo "Completed and verified: $COMPLETED_ARM seed=$COMPLETED_SEED"
        else
            FAILED=1
            echo "CONFIG MISMATCH: $COMPLETED_ARM seed=$COMPLETED_SEED; inspect $RUN_JSON" >&2
        fi
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

FINAL_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
FINAL_GIT_STATUS="$(git status --porcelain --untracked-files=all 2>/dev/null || true)"
if [[ "$FINAL_COMMIT" != "$GIT_COMMIT" || "$FINAL_GIT_STATUS" != "$GIT_STATUS" ]]; then
    FAILED=1
    echo "Source tree changed while the sweep was running; results are not one-code comparisons" >&2
fi

if (( FAILED != 0 )); then
    echo "At least one calibration-gate run failed" >&2
    exit 1
fi

echo "Calibration gate complete: $RUN_ROOT"
echo "Each run contains metrics.jsonl, run_stats.json, and final student weights."
