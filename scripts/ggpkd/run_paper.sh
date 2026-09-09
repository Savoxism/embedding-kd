#!/usr/bin/env bash
# Three-seed GGPKD run over the three paper pairs: 3 pairs x 3 seeds = 9 runs,
# scheduled one-per-GPU, then aggregated to mean +- std by summarize.py.
#
# The teacher cache and the kNN graph are seed-independent -- the graph is a
# function of the teacher embeddings, graph_k and the bandwidth rule, and with
# diffusion_quota=None the candidate draw is deterministic too. So both are
# built once per pair in a serialized first phase and read-only afterwards.
# Letting the nine training runs build them lazily would have three processes
# writing the same graph.pt at once.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../common/run_stats.sh
source "$REPO_ROOT/scripts/common/run_stats.sh"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-$REPO_ROOT/results/ggpkd}"
RUN_ROOT="$RESULT_BASE/$RUN_ID"
# Caches live outside the run directory by default: they are pair-scoped and
# expensive (teacher forward pass over the corpus + the kNN build), so a second
# run reuses them instead of paying for them again. Point CACHE_ROOT at the run
# directory to force a cold build.
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache/ggpkd}"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
RUNS_DIR="$RUN_ROOT/runs"
MANIFEST="$RUN_ROOT/manifest.tsv"
# Per-run wall clock and peak memory, collected as each run finishes so the
# table survives a sweep that is interrupted partway through.
STATS_TSV="$RUN_ROOT/stats.tsv"

IFS=',' read -r -a PAIRS <<< "${PAIRS:-qwen3_0_6b_to_minilmv2_h384,bge_m3_to_minilmv2_h768,qwen3_4b_to_bert_base}"
IFS=',' read -r -a SEEDS <<< "${SEEDS:-42,43,44}"

# One run per GPU. Unlike the TALAS runner this does not demand eight of them:
# with fewer GPUs the pool just holds fewer runs in flight and the same nine
# tasks are queued behind them.
if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "$GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
else
    echo "No nvidia-smi and no GPUS override; set GPUS=0 to run on one device" >&2
    exit 2
fi
if (( ${#GPU_LIST[@]} == 0 )); then
    echo "No GPUs available for the GGPKD paper run" >&2
    exit 2
fi

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite existing GGPKD run: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Project virtual-environment Python is not executable: $PYTHON_BIN" >&2
    exit 2
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

printf 'phase\tpair\tseed\tgpu\tpid\tstate\n' > "$MANIFEST"
printf 'phase\tunit\tseed\twall_seconds\tpeak_host_rss_mib\tpeak_gpu_mib\texit_code\n' > "$STATS_TSV"
{
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'pairs\t%s\n' "${PAIRS[*]}"
    printf 'seeds\t%s\n' "${SEEDS[*]}"
    printf 'gpus\t%s\n' "${GPU_LIST[*]}"
    printf 'cache_root\t%s\n' "$CACHE_ROOT"
    printf 'commit\t%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$RUN_ROOT/run_config.tsv"

echo "GGPKD paper run $RUN_ID"
echo "  pairs: ${PAIRS[*]}"
echo "  seeds: ${SEEDS[*]}"
echo "  gpus:  ${GPU_LIST[*]}"
echo "  root:  $RUN_ROOT"

# ---- Phase 1: per-pair teacher cache + kNN graph -------------------------
# Serialized across pairs on the first GPU. Each build wants the whole device
# for the block-wise cosine pass, and this phase is a small fraction of the
# nine training runs behind it.
for pair in "${PAIRS[@]}"; do
    log="$LOG_DIR/cache.$pair.log"
    exit_file="$STATUS_DIR/cache.$pair.exit"
    echo "Preparing caches for $pair (log: $log)"
    set +e
    PAIR_KEY="$pair" GPU="${GPU_LIST[0]}" PYTHON_BIN="$PYTHON_BIN" \
        CACHE_PATH="$CACHE_ROOT/$pair/teacher_train.pt" \
        GGPKD_CACHE_PATH="$CACHE_ROOT/$pair/graph.pt" \
        GGPKD_LOG_DIR="$RUN_ROOT/graph_logs/$pair" \
        SAVE_DIR="$RUN_ROOT/cache_setup/$pair" \
        bash "$SCRIPT_DIR/train.sh" --prepare_cache_only >"$log" 2>&1
    code=$?
    set -e
    printf '%s\n' "$code" > "$exit_file"
    printf 'cache\t%s\t-\t%s\t-\t%s\n' "$pair" "${GPU_LIST[0]}" "$code" >> "$MANIFEST"
    printf 'cache\t%s\t-\t%s\n' "$pair" \
        "$(run_stats_row "$RUN_ROOT/cache_setup/$pair/run_stats.json")" >> "$STATS_TSV"
    if (( code != 0 )); then
        echo "Cache preparation failed for $pair (exit $code); see $log" >&2
        exit 1
    fi
done

# ---- Phase 2: 3 pairs x 3 seeds over the GPU pool ------------------------
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
        PAIR_KEY="$pair" SEED="$seed" GPU="$gpu" PYTHON_BIN="$PYTHON_BIN" \
            CACHE_PATH="$CACHE_ROOT/$pair/teacher_train.pt" \
            GGPKD_CACHE_PATH="$CACHE_ROOT/$pair/graph.pt" \
            GGPKD_LOG_DIR="$run_dir/graph_logs" \
            SAVE_DIR="$run_dir" WEIGHTS_DIR="$run_dir/weights" \
            bash "$SCRIPT_DIR/train.sh" --final_weights_only >"$log" 2>&1
        code=$?
        printf '%s\n' "$code" > "$exit_file"
        exit "$code"
    ) &
    local pid=$!
    active_pids+=("$pid")
    PID_GPU["$pid"]="$gpu"
    PID_TASK["$pid"]="$task"
    printf 'train\t%s\t%s\t%s\t%s\trunning\n' "$pair" "$seed" "$gpu" "$pid" >> "$MANIFEST"
    echo "Launched $task on GPU $gpu (pid $pid)"
}

while (( next_task < ${#tasks_pair[@]} && next_task < ${#GPU_LIST[@]} )); do
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
    # The manifest is append-only, so a finished run gets a second row rather
    # than an edit to its "running" one; grep for the pid to pair them up.
    printf 'train\t%s\t%s\t%s\t%s\texit_%s\n' \
        "${completed_task%%.seed_*}" "${completed_task##*.seed_}" \
        "$completed_gpu" "$completed_pid" "$completed_status" >> "$MANIFEST"
    completed_pair="${completed_task%%.seed_*}"
    completed_seed="${completed_task##*.seed_}"
    printf 'train\t%s\t%s\t%s\n' "$completed_pair" "$completed_seed" \
        "$(run_stats_row "$RUNS_DIR/$completed_pair/seed_$completed_seed/run_stats.json")" >> "$STATS_TSV"
    if (( completed_status != 0 )); then
        failed_runs=1
        echo "FAILED: $completed_task exited $completed_status (log: $LOG_DIR/$completed_task.log)" >&2
    else
        echo "Completed: $completed_task ($(run_stats_field "$RUNS_DIR/$completed_pair/seed_$completed_seed/run_stats.json" wall_seconds)s)"
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

echo
echo "Per-run cost (wall seconds, peak host RSS MiB, peak GPU MiB):"
column -t -s $'\t' "$STATS_TSV" 2>/dev/null || cat "$STATS_TSV"

if (( failed_runs != 0 )); then
    echo "At least one GGPKD training run failed; refusing to aggregate" >&2
    exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/summarize.py" "$RUN_ROOT" \
    --pairs "$(IFS=,; echo "${PAIRS[*]}")" \
    --seeds "$(IFS=,; echo "${SEEDS[*]}")" | tee "$RUN_ROOT/summary.txt"
echo "GGPKD paper run complete: $RUN_ROOT"
