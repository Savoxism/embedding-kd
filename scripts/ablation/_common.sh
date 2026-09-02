# Shared protocol for the GGPKD ablations. Sourced, never executed.
#
# Everything the ablation plan calls "locked before running" lives here and
# nowhere else, so no arm can drift from another by editing one script. An arm
# script sets only what it is ablating; if a value is not in this file it is not
# part of the protocol.
#
# One run occupies one GPU. Pass GPU=<id> to pin it:
#     GPU=0 bash scripts/ablation/s1_support.sh
#     GPU=1 bash scripts/ablation/s2_scales.sh    # concurrently, different card
# Arms inside a script are sequential -- they share a teacher cache and a graph
# artifact, and two processes building the same artifact would race on it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---- Locked protocol --------------------------------------------------------
# Qwen3-0.6B -> MiniLMv2-H384 is the setting the plan fixes on: cheapest per run
# and where the reported gain is largest.
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

# The ambient-scale temperature. 0.10 is the config default (Hinton convention);
# `0` derives it as the median graph bandwidth. It is pinned explicitly rather
# than left to the config so that every arm -- and the shared `full` runs that
# S1-S4 all compare against -- provably used the same value, and so run.json
# records it. Change it here, once, or the reuse of `full` across ablations is
# no longer valid.
DIRECT_TEMP="${DIRECT_TEMP:-0.10}"

# ---- Layout -----------------------------------------------------------------
ABL_ROOT="${ABL_ROOT:-runs/ablation/${PAIR_KEY}}"
CACHE_ROOT="${CACHE_ROOT:-cache/ggpkd/${PAIR_KEY}}"
LOG_ROOT="${LOG_ROOT:-logs/ggpkd/${PAIR_KEY}}"
# Teacher embeddings depend on the pair alone, so every arm shares one cache.
TEACHER_CACHE="${TEACHER_CACHE:-${CACHE_ROOT}/teacher_train.pt}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" && "${PYTHON_BIN}" == "python3" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

# ---- run_arm ----------------------------------------------------------------
# run_arm <ablation-id> <arm-name> <seed> [extra main.py flags...]
#
# GRAPH_KEY selects which graph artifact the arm reads. Arms that change the
# graph itself (diffusion_scales, knn_mode, graph_k, perplexity) MUST set it:
# the builder validates its metadata and rebuilds on mismatch, so two arms
# pointed at one path would rebuild the graph on every alternation. Arms that
# only change the objective or the sampler leave it at "base" and share one
# build. Default reset per call so a stale value cannot leak between arms.
run_arm() {
    local ablation="$1" arm="$2" seed="$3"
    shift 3
    # Bash does not reliably restore a `VAR=x func` prefix assignment after a
    # *function* returns, so read it once and clear it. Left set, the next arm in
    # the loop would silently inherit the previous arm's graph.
    local graph_key="${GRAPH_KEY:-base}"
    unset GRAPH_KEY

    local save_dir="${ABL_ROOT}/${ablation}/${arm}/seed${seed}"
    local weights_dir="${ABL_ROOT}/${ablation}/${arm}/seed${seed}/weights"
    local graph_cache="${CACHE_ROOT}/graph_${graph_key}.pt"
    local graph_log="${LOG_ROOT}/graph_${graph_key}"

    if [[ -f "${save_dir}/.done" && "${FORCE:-0}" != "1" ]]; then
        echo "[skip] ${ablation}/${arm}/seed${seed} already complete (FORCE=1 to rerun)"
        return 0
    fi

    mkdir -p "${save_dir}" "${weights_dir}" "$(dirname "${graph_cache}")" "${graph_log}"
    echo "==================================================================="
    echo "[run ] ${ablation} / ${arm} / seed ${seed}  (GPU ${CUDA_VISIBLE_DEVICES}, graph ${graph_key})"
    echo "       extra: $*"
    echo "==================================================================="

    local started_at
    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local t0=${SECONDS}

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
        --direct_temp "${DIRECT_TEMP}" \
        --cache_path "${TEACHER_CACHE}" \
        --ggpkd_cache_path "${graph_cache}" \
        --ggpkd_log_dir "${graph_log}" \
        --save_dir "${save_dir}" \
        --weights_dir "${weights_dir}" \
        --final_weights_only \
        "$@" 2>&1 | tee "${save_dir}/train.log"

    local elapsed=$(( SECONDS - t0 ))
    # The arm's identity, written by the runner rather than parsed back out of a
    # directory name: collect.py groups on these fields, and a path is not a
    # record. wall_clock_seconds belongs to the appendix efficiency table and is
    # only observable here.
    "${PYTHON_BIN}" - "$save_dir" "$ablation" "$arm" "$seed" "$graph_key" "$elapsed" "$started_at" "$@" <<'PYEOF'
import json, sys
save_dir, ablation, arm, seed, graph_key, elapsed, started_at, *extra = sys.argv[1:]
with open(f"{save_dir}/arm.json", "w", encoding="utf-8") as fh:
    json.dump(
        {
            "ablation": ablation,
            "arm": arm,
            "seed": int(seed),
            "graph_key": graph_key,
            "wall_clock_seconds": int(elapsed),
            "started_at": started_at,
            "extra_args": extra,
        },
        fh,
        indent=1,
        sort_keys=True,
    )
PYEOF
    touch "${save_dir}/.done"
    echo "[done] ${ablation}/${arm}/seed${seed} in ${elapsed}s -> ${save_dir}"
}

# The unablated model. S1-S4 all compare against it, so it is trained once per
# seed under its own id and reused; the plan's budget arithmetic assumes exactly
# this. Any arm script may call it -- the .done guard makes repeat calls free.
run_full() {
    local seed="$1"
    GRAPH_KEY="base" run_arm full full "${seed}"
}

echo "protocol: ${TEACHER_MODEL} -> ${STUDENT_MODEL} | lr=${LR} epochs=${EPOCHS} bs=${BATCH_SIZE} seeds=[${SEEDS}] direct_temp=${DIRECT_TEMP}"
echo "outputs : ${ABL_ROOT}"
