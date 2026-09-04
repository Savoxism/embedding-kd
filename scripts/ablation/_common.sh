#!/usr/bin/env bash
# Shared, locked protocol for the paper ablations. Source this file; do not run it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Paper setting. Override from the environment only when intentionally starting
# a separate experiment root.
PAIR_KEY="${PAIR_KEY:-qwen3_0_6b_to_minilmv2_h384}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
STUDENT_MODEL="${STUDENT_MODEL:-nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base}"
POOLING_METHOD="${POOLING_METHOD:-last_token}"
TRAIN_DATA="${TRAIN_DATA:-data/train_set/merged_3_data_5k_each.csv}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEEDS="${SEEDS:-42 43 44}"

# Pin every method-defining value so all arms stay comparable if config defaults
# later move. The canonical support quota remains derived from the graph.
GRAPH_K="${GRAPH_K:-200}"
PERPLEXITY="${PERPLEXITY:-30}"
TRUNCATION_TOLERANCE="${TRUNCATION_TOLERANCE:-0.01}"
DIFFUSION_SCALES="${DIFFUSION_SCALES:-1,2,4}"
ROW_WEIGHT="${ROW_WEIGHT:-1.0}"
ROW_START_EPOCH="${ROW_START_EPOCH:-1}"
DIRECT_TEMP="${DIRECT_TEMP:-0}"
HARD_NEG_K="${HARD_NEG_K:-40}"
RANDOM_NEG_K="${RANDOM_NEG_K:-26}"

EXPERIMENT_KEY="${EXPERIMENT_KEY:-paper_v1}"
ABL_ROOT="${ABL_ROOT:-runs/ablation/${PAIR_KEY}/${EXPERIMENT_KEY}}"
CACHE_ROOT="${CACHE_ROOT:-cache/ggpkd/${PAIR_KEY}/${EXPERIMENT_KEY}}"
LOG_ROOT="${LOG_ROOT:-logs/ggpkd/${PAIR_KEY}/${EXPERIMENT_KEY}}"
TEACHER_CACHE="${TEACHER_CACHE:-cache/ggpkd/${PAIR_KEY}/teacher_train.pt}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" && "${PYTHON_BIN}" == "python3" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

base_graph_path() {
    printf '%s\n' "${CACHE_ROOT}/graph_base.pt"
}

# run_arm <group> <arm> <seed> [main.py overrides ...]
#
# A graph-changing arm must set GRAPH_KEY immediately before this call. All
# other arms share graph_base.pt. A completed arm is safely resumable via its
# .done marker; an interrupted arm is rerun with a fresh run_id in the same
# directory, which the repository telemetry keeps distinguishable.
run_arm() {
    local group="$1"
    local arm="$2"
    local seed="$3"
    shift 3

    local graph_key="${GRAPH_KEY:-base}"
    unset GRAPH_KEY || true

    local save_dir="${ABL_ROOT}/${group}/${arm}/seed${seed}"
    local weights_dir="${save_dir}/weights"
    local graph_cache="${CACHE_ROOT}/graph_${graph_key}.pt"
    local graph_log="${LOG_ROOT}/graph_${graph_key}"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry ] ${group}/${arm}/seed${seed} | GPU=${CUDA_VISIBLE_DEVICES} graph=${graph_key}"
        echo "       overrides: $*"
        return 0
    fi

    # Different group scripts can request the same shared arm concurrently.
    # flock is process-scoped and automatically releases on exit, including a
    # failed or interrupted training process, so it cannot leave a stale lock.
    local lock_dir="${ABL_ROOT}/.locks"
    local lock_path="${lock_dir}/${group}_${arm}_seed${seed}.lock"
    local lock_fd
    mkdir -p "${lock_dir}"
    exec {lock_fd}>"${lock_path}"
    flock "${lock_fd}"

    if [[ -f "${save_dir}/.done" ]]; then
        echo "[skip] ${group}/${arm}/seed${seed} already complete"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 0
    fi

    mkdir -p "${save_dir}" "${weights_dir}" "$(dirname "${graph_cache}")" "${graph_log}"
    echo "[run ] ${group}/${arm}/seed${seed} | GPU=${CUDA_VISIBLE_DEVICES} graph=${graph_key}"
    echo "       overrides: $*"

    local started_at
    local elapsed
    local start_seconds=${SECONDS}
    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    "${PYTHON_BIN}" main.py \
        --method ggpkd \
        --train_data "${TRAIN_DATA}" \
        --student_model "${STUDENT_MODEL}" \
        --teacher_model "${TEACHER_MODEL}" \
        --pooling_method "${POOLING_METHOD}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --max_length "${MAX_LENGTH}" \
        --num_workers "${NUM_WORKERS}" \
        --seed "${seed}" \
        --graph_k "${GRAPH_K}" \
        --perplexity "${PERPLEXITY}" \
        --truncation_tolerance "${TRUNCATION_TOLERANCE}" \
        --diffusion_scales "${DIFFUSION_SCALES}" \
        --row_weight "${ROW_WEIGHT}" \
        --row_start_epoch "${ROW_START_EPOCH}" \
        --direct_temp "${DIRECT_TEMP}" \
        --hard_neg_k "${HARD_NEG_K}" \
        --random_neg_k "${RANDOM_NEG_K}" \
        --support_policy topk \
        --relation_target diffusion \
        --cache_path "${TEACHER_CACHE}" \
        --ggpkd_cache_path "${graph_cache}" \
        --ggpkd_log_dir "${graph_log}" \
        --save_dir "${save_dir}" \
        --weights_dir "${weights_dir}" \
        --final_weights_only \
        "$@" 2>&1 | tee "${save_dir}/train.log"

    elapsed=$(( SECONDS - start_seconds ))
    "${PYTHON_BIN}" - "${save_dir}" "${group}" "${arm}" "${seed}" \
        "${graph_key}" "${elapsed}" "${started_at}" "$@" <<'PYEOF'
import json
import sys

save_dir, group, arm, seed, graph_key, elapsed, started_at, *extra = sys.argv[1:]
with open(f"{save_dir}/arm.json", "w", encoding="utf-8") as handle:
    json.dump(
        {
            "ablation": group,
            "group": group,
            "arm": arm,
            "seed": int(seed),
            "graph_key": graph_key,
            "wall_clock_seconds": int(elapsed),
            "started_at": started_at,
            "extra_args": extra,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
PYEOF
    touch "${save_dir}/.done"
    flock -u "${lock_fd}"
    exec {lock_fd}>&-
    echo "[done] ${group}/${arm}/seed${seed} in ${elapsed}s"
}

run_full() {
    local seed="$1"
    GRAPH_KEY=base run_arm full full "${seed}"
}

canonical_quota() {
    if [[ -n "${CANONICAL_QUOTA:-}" ]]; then
        printf '%s\n' "${CANONICAL_QUOTA}"
        return 0
    fi
    "${PYTHON_BIN}" scripts/ablation/budget.py \
        --artifact "$(base_graph_path)" \
        --hard "${HARD_NEG_K}" \
        --random "${RANDOM_NEG_K}" \
        --format quota
}

canonical_width() {
    if [[ -n "${CANONICAL_QUOTA:-}" ]]; then
        printf '%s\n' "$(( CANONICAL_QUOTA + HARD_NEG_K + RANDOM_NEG_K ))"
        return 0
    fi
    "${PYTHON_BIN}" scripts/ablation/budget.py \
        --artifact "$(base_graph_path)" \
        --hard "${HARD_NEG_K}" \
        --random "${RANDOM_NEG_K}" \
        --format width
}

# Extreme deletion shared by the support and component tables. Candidates are
# uniform over the corpus, the graph objective is absent, and L_row is absent.
# Keeping the canonical candidate width prevents the deletion from winning or
# losing merely because it encoded a different number of columns per anchor.
run_no_graph_diffusion() {
    local seed="$1"
    local width
    width="$(canonical_width)"
    GRAPH_KEY=base run_arm shared no_graph_diffusion "${seed}" \
        --relation_target ambient_only \
        --row_weight 0 \
        --diffusion_quota 0 \
        --hard_neg_k 0 \
        --random_neg_k "${width}"
}

# Shared between the component table and row-weight sensitivity (lambda=0).
run_no_row() {
    local seed="$1"
    GRAPH_KEY=base run_arm components no_row "${seed}" --row_weight 0
}

# Clean no-diffusion control. Keep the full artifact, candidate width, ambient
# scale, row loss, and total graph-group weight. Only the information supplied
# by multi-hop diffusion is removed: support comes from P^1 and its relation
# target is the teacher's raw cosine profile over those local columns.
run_no_diffusion_clean() {
    local seed="$1"
    GRAPH_KEY=base run_arm components no_diffusion_clean "${seed}" \
        --support_policy local_topk \
        --relation_target direct
}

echo "protocol: ${TEACHER_MODEL} -> ${STUDENT_MODEL} | lr=${LR} epochs=${EPOCHS} bs=${BATCH_SIZE}"
echo "seeds: [${SEEDS}] | output: ${ABL_ROOT}"
