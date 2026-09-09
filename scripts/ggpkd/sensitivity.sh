#!/usr/bin/env bash
# Hyperparameter sensitivity for GGPKD: sweep each declared knob one axis at a
# time, three seeds per point, and report how far the benchmark average moves
# over the swept range.
#
# method.md Sec. 4 declares exactly two hyperparameters -- graph_k (200) and
# row_weight (1.0) -- so those are the default axes. Everything else in the
# method is either derived (every temperature, the ambient weight, the
# candidate width) or a deletion arm, and a deletion belongs in the ablation
# table rather than here: this asks how sharp the optimum is, not whether a
# component is needed.
#
# graph_k is the axis that costs something. It sets retrieval width *and*, via
# tau_i, row sharpness, so each value needs its own graph artifact; the teacher
# embedding cache is shared across all of them. scripts/ggpkd/pick_graph_k.py
# reports the induced sharpness per k from one teacher pass and is the cheap
# pre-screen for extending this grid -- a k whose rows have already gone
# uniform is not worth a training run.
#
# The default point (graph_k=200, row_weight=1.0) sits on both axes and is
# trained once; the summary places it in both tables.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../common/run_stats.sh
source "$REPO_ROOT/scripts/common/run_stats.sh"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-$REPO_ROOT/results/ggpkd_sensitivity}"
RUN_ROOT="$RESULT_BASE/$RUN_ID"
# Sensitivity runs on one pair: the point is the shape of each axis, and three
# pairs would multiply 24 runs by three without changing that shape.
PAIR="${PAIR:-qwen3_0_6b_to_minilmv2_h384}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache/ggpkd_sensitivity}"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
RUNS_DIR="$RUN_ROOT/runs"
MANIFEST="$RUN_ROOT/manifest.tsv"
ARMS_TSV="$RUN_ROOT/arms.tsv"
# Per-run wall clock and peak memory, collected as each run finishes so the
# table survives a sweep that is interrupted partway through.
STATS_TSV="$RUN_ROOT/stats.tsv"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a SEEDS <<< "${SEEDS:-42,43,44}"

# graph_key -> the CLI flags that define its artifact. These are prepended to
# every arm on that key, so an arm can never train against a graph built with
# different flags: there is one place the graph is described.
GRAPH_SPEC_DEFAULT="\
graph_k_50|--graph_k 50
graph_k_100|--graph_k 100
graph_k_200|
graph_k_400|--graph_k 400"

# axis|value|label|graph_key|extra flags
# `default` appears on both axes at its own value and is trained once.
ARMS_DEFAULT="\
graph_k|50|graph_k_50|graph_k_50|
graph_k|100|graph_k_100|graph_k_100|
graph_k|200|default|graph_k_200|
graph_k|400|graph_k_400|graph_k_400|
row_weight|0|row_weight_0|graph_k_200|--row_weight 0
row_weight|0.25|row_weight_0_25|graph_k_200|--row_weight 0.25
row_weight|0.5|row_weight_0_5|graph_k_200|--row_weight 0.5
row_weight|1.0|default|graph_k_200|
row_weight|2.0|row_weight_2|graph_k_200|--row_weight 2"

# Both tables can be replaced wholesale to sweep another axis -- truncation
# tolerance, diffusion radius, a candidate budget -- without editing this file.
# A new axis that changes the graph needs a new graph_key and a GRAPH_SPEC line
# for it; one that does not (any pure objective knob) reuses graph_k_200.
if [[ -n "${GRAPH_SPEC_FILE:-}" ]]; then
    GRAPH_SPEC="$(cat "$GRAPH_SPEC_FILE")"
else
    GRAPH_SPEC="${GRAPH_SPEC:-$GRAPH_SPEC_DEFAULT}"
fi
if [[ -n "${ARMS_FILE:-}" ]]; then
    ARMS_SPEC="$(cat "$ARMS_FILE")"
else
    ARMS_SPEC="${ARMS:-$ARMS_DEFAULT}"
fi

if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "$GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
else
    echo "No nvidia-smi and no GPUS override; set GPUS=0 to run on one device" >&2
    exit 2
fi
if (( ${#GPU_LIST[@]} == 0 )); then
    echo "No GPUs available for the GGPKD sensitivity sweep" >&2
    exit 2
fi

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite existing sensitivity run: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Project virtual-environment Python is not executable: $PYTHON_BIN" >&2
    exit 2
fi

# ---- Parse the arm matrix ------------------------------------------------
declare -A GRAPH_FLAGS=()
graph_keys=()
while IFS='|' read -r key flags; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    if [[ -n "${GRAPH_FLAGS[$key]+set}" ]]; then
        echo "Duplicate graph_key in GRAPH_SPEC: $key" >&2
        exit 2
    fi
    GRAPH_FLAGS["$key"]="$flags"
    graph_keys+=("$key")
done <<< "$GRAPH_SPEC"

declare -A ARM_FLAGS=()
declare -A ARM_GRAPH=()
arm_labels=()
arm_rows=()
while IFS='|' read -r axis value label graph_key extra; do
    [[ -z "$axis" || "$axis" == \#* ]] && continue
    if [[ -z "${GRAPH_FLAGS[$graph_key]+set}" ]]; then
        echo "Arm $label references unknown graph_key: $graph_key" >&2
        exit 2
    fi
    if [[ -n "${ARM_FLAGS[$label]+set}" ]]; then
        # A label is a run directory. Two rows may share one only when they
        # describe the identical run -- that is how the default point sits on
        # both axes without being trained twice.
        if [[ "${ARM_FLAGS[$label]}" != "$extra" || "${ARM_GRAPH[$label]}" != "$graph_key" ]]; then
            echo "Arm label $label is reused with different flags" >&2
            exit 2
        fi
    else
        ARM_FLAGS["$label"]="$extra"
        ARM_GRAPH["$label"]="$graph_key"
        arm_labels+=("$label")
    fi
    arm_rows+=("$axis|$value|$label")
done <<< "$ARMS_SPEC"

if (( ${#arm_labels[@]} == 0 )); then
    echo "No arms to run" >&2
    exit 2
fi

total_runs=$(( ${#arm_labels[@]} * ${#SEEDS[@]} ))
echo "GGPKD sensitivity sweep $RUN_ID"
echo "  pair:   $PAIR"
echo "  arms:   ${#arm_labels[@]} (${arm_labels[*]})"
echo "  graphs: ${#graph_keys[@]} (${graph_keys[*]})"
echo "  seeds:  ${SEEDS[*]}"
echo "  gpus:   ${GPU_LIST[*]}"
echo "  runs:   $total_runs"
echo "  root:   $RUN_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "DRY_RUN: planned runs"
    for label in "${arm_labels[@]}"; do
        graph_key="${ARM_GRAPH[$label]}"
        printf '  %-18s graph=%-14s flags: %s %s\n' \
            "$label" "$graph_key" "${GRAPH_FLAGS[$graph_key]}" "${ARM_FLAGS[$label]}"
    done
    exit 0
fi

mkdir -p "$STATUS_DIR" "$LOG_DIR" "$RUNS_DIR" "$CACHE_ROOT"

finish_controller() {
    local code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$RUN_ROOT/controller.exit"
    exit "$code"
}
trap finish_controller EXIT

export TOKENIZERS_PARALLELISM=false

# The summary reads its axes from here, so the spec that ran and the spec that
# is reported cannot drift apart.
printf 'axis\tvalue\tlabel\n' > "$ARMS_TSV"
for row in "${arm_rows[@]}"; do
    IFS='|' read -r axis value label <<< "$row"
    printf '%s\t%s\t%s\n' "$axis" "$value" "$label" >> "$ARMS_TSV"
done

printf 'phase\tarm\tseed\tgpu\tpid\tstate\n' > "$MANIFEST"
printf 'phase\tunit\tseed\twall_seconds\tpeak_host_rss_mib\tpeak_gpu_mib\texit_code\n' > "$STATS_TSV"
{
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'pair\t%s\n' "$PAIR"
    printf 'seeds\t%s\n' "${SEEDS[*]}"
    printf 'gpus\t%s\n' "${GPU_LIST[*]}"
    printf 'cache_root\t%s\n' "$CACHE_ROOT"
    printf 'commit\t%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$RUN_ROOT/run_config.tsv"

TEACHER_CACHE="$CACHE_ROOT/$PAIR/teacher_train.pt"
graph_path() { printf '%s/%s/graph_%s.pt' "$CACHE_ROOT" "$PAIR" "$1"; }

# ---- Phase 1: one graph artifact per graph_key ---------------------------
# Serialized: each build wants the device to itself, and the first one also
# fills the teacher cache the rest read. Running them concurrently would have
# several processes writing that same teacher cache.
for key in "${graph_keys[@]}"; do
    log="$LOG_DIR/graph.$key.log"
    exit_file="$STATUS_DIR/graph.$key.exit"
    echo "Building graph $key (log: $log)"
    set +e
    # shellcheck disable=SC2086 -- graph flags are a deliberate word-split list
    PAIR_KEY="$PAIR" GPU="${GPU_LIST[0]}" PYTHON_BIN="$PYTHON_BIN" \
        CACHE_PATH="$TEACHER_CACHE" \
        GGPKD_CACHE_PATH="$(graph_path "$key")" \
        GGPKD_LOG_DIR="$RUN_ROOT/graph_logs/$key" \
        SAVE_DIR="$RUN_ROOT/graph_setup/$key" \
        bash "$SCRIPT_DIR/train.sh" --prepare_cache_only ${GRAPH_FLAGS[$key]} \
        >"$log" 2>&1
    code=$?
    set -e
    printf '%s\n' "$code" > "$exit_file"
    printf 'graph\t%s\t-\t%s\t-\t%s\n' "$key" "${GPU_LIST[0]}" "$code" >> "$MANIFEST"
    printf 'graph\t%s\t-\t%s\n' "$key" \
        "$(run_stats_row "$RUN_ROOT/graph_setup/$key/run_stats.json")" >> "$STATS_TSV"
    if (( code != 0 )); then
        echo "Graph build failed for $key (exit $code); see $log" >&2
        exit 1
    fi
done

# ---- Phase 2: arms x seeds over the GPU pool -----------------------------
tasks_label=()
tasks_seed=()
for label in "${arm_labels[@]}"; do
    for seed in "${SEEDS[@]}"; do
        tasks_label+=("$label")
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
    local label="${tasks_label[$task_index]}"
    local seed="${tasks_seed[$task_index]}"
    local task="$label.seed_$seed"
    local graph_key="${ARM_GRAPH[$label]}"
    local run_dir="$RUNS_DIR/$label/seed_$seed"
    local log="$LOG_DIR/$task.log"
    local exit_file="$STATUS_DIR/$task.exit"

    mkdir -p "$run_dir"
    (
        set +e
        # shellcheck disable=SC2086 -- both flag strings are deliberate lists
        PAIR_KEY="$PAIR" SEED="$seed" GPU="$gpu" PYTHON_BIN="$PYTHON_BIN" \
            CACHE_PATH="$TEACHER_CACHE" \
            GGPKD_CACHE_PATH="$(graph_path "$graph_key")" \
            GGPKD_LOG_DIR="$run_dir/graph_logs" \
            SAVE_DIR="$run_dir" WEIGHTS_DIR="$run_dir/weights" \
            bash "$SCRIPT_DIR/train.sh" --final_weights_only --eval_every 0 \
            ${GRAPH_FLAGS[$graph_key]} ${ARM_FLAGS[$label]} >"$log" 2>&1
        code=$?
        printf '%s\n' "$code" > "$exit_file"
        exit "$code"
    ) &
    local pid=$!
    active_pids+=("$pid")
    PID_GPU["$pid"]="$gpu"
    PID_TASK["$pid"]="$task"
    printf 'train\t%s\t%s\t%s\t%s\trunning\n' "$label" "$seed" "$gpu" "$pid" >> "$MANIFEST"
    echo "Launched $task on GPU $gpu (pid $pid)"
}

while (( next_task < ${#tasks_label[@]} && next_task < ${#GPU_LIST[@]} )); do
    launch_training "$next_task" "${GPU_LIST[$next_task]}"
    ((next_task += 1))
done

while (( ${#active_pids[@]} > 0 )); do
    completed_pid=""
    set +e
    wait -n -p completed_pid "${active_pids[@]}"
    completed_status=$?
    set -e
    if [[ -z "$completed_pid" ]]; then
        echo "Could not identify completed GGPKD process" >&2
        exit 1
    fi
    completed_gpu="${PID_GPU[$completed_pid]}"
    completed_task="${PID_TASK[$completed_pid]}"
    printf 'train\t%s\t%s\t%s\t%s\texit_%s\n' \
        "${completed_task%.seed_*}" "${completed_task##*.seed_}" \
        "$completed_gpu" "$completed_pid" "$completed_status" >> "$MANIFEST"
    completed_label="${completed_task%.seed_*}"
    completed_seed="${completed_task##*.seed_}"
    printf 'train\t%s\t%s\t%s\n' "$completed_label" "$completed_seed" \
        "$(run_stats_row "$RUNS_DIR/$completed_label/seed_$completed_seed/run_stats.json")" >> "$STATS_TSV"
    if (( completed_status != 0 )); then
        failed_runs=1
        echo "FAILED: $completed_task exited $completed_status (log: $LOG_DIR/$completed_task.log)" >&2
    else
        echo "Completed: $completed_task ($(run_stats_field "$RUNS_DIR/$completed_label/seed_$completed_seed/run_stats.json" wall_seconds)s)"
    fi

    remaining=()
    for pid in "${active_pids[@]}"; do
        if [[ "$pid" != "$completed_pid" ]]; then
            remaining+=("$pid")
        fi
    done
    active_pids=("${remaining[@]}")
    unset 'PID_GPU[$completed_pid]' 'PID_TASK[$completed_pid]'

    if (( next_task < ${#tasks_label[@]} )); then
        launch_training "$next_task" "$completed_gpu"
        ((next_task += 1))
    fi
done

echo
echo "Per-run cost (wall seconds, peak host RSS MiB, peak GPU MiB):"
column -t -s $'\t' "$STATS_TSV" 2>/dev/null || cat "$STATS_TSV"

if (( failed_runs != 0 )); then
    echo "At least one sensitivity run failed; refusing to aggregate" >&2
    exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_sensitivity.py" "$RUN_ROOT" \
    --seeds "$(IFS=,; echo "${SEEDS[*]}")" | tee "$RUN_ROOT/summary.txt"
echo "GGPKD sensitivity sweep complete: $RUN_ROOT"
