#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../common/run_stats.sh
source "$REPO_ROOT/scripts/common/run_stats.sh"

# The pair key drives the model names *and* every per-pair path. Overriding only
# the models -- which this script used to do -- leaves the caches pointing at the
# config's default pair, so a cold run writes one teacher's embeddings under
# another teacher's path. `check_cache_provenance` only catches that on the next
# cache hit, by which point the file is already wrong.
PAIR_KEY="${PAIR_KEY:-qwen3_0_6b_to_minilmv2_h384}"

# Set by scripts/ggpkd/run_paper.sh, which pins one run per GPU and needs the
# project interpreter rather than whatever `python3` resolves to on the server.
# All three keep the previous behaviour when unset.
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED_VALUE="${SEED:-}"
GPU_VALUE="${GPU:-}"

if [[ $# -gt 0 && "$1" != -* ]]; then
    PAIR_KEY="$1"
    shift
fi

case "$PAIR_KEY" in
    qwen3_0_6b_to_minilmv2_h384)
        TEACHER_MODEL_DEFAULT="Qwen/Qwen3-Embedding-0.6B"
        STUDENT_MODEL_DEFAULT="nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base"
        POOLING_METHOD_DEFAULT="last_token"
        ;;
    bge_m3_to_minilmv2_h768)
        TEACHER_MODEL_DEFAULT="BAAI/bge-m3"
        STUDENT_MODEL_DEFAULT="nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base"
        POOLING_METHOD_DEFAULT="cls"
        ;;
    qwen3_4b_to_bert_base)
        TEACHER_MODEL_DEFAULT="Qwen/Qwen3-Embedding-4B"
        STUDENT_MODEL_DEFAULT="google-bert/bert-base-uncased"
        POOLING_METHOD_DEFAULT="last_token"
        ;;
    *)
        echo "Unknown GGPKD pair: $PAIR_KEY" >&2
        echo "Expected qwen3_0_6b_to_minilmv2_h384, bge_m3_to_minilmv2_h768, or qwen3_4b_to_bert_base" >&2
        exit 2
        ;;
esac

echo "======================================"
echo "Training with GGPKD method (pair=$PAIR_KEY)"
echo "======================================"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
if [[ -n "$GPU_VALUE" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_VALUE"
fi
if [[ -n "$SEED_VALUE" && ! "$SEED_VALUE" =~ ^[0-9]+$ ]]; then
    echo "Seed must be a non-negative integer, got: $SEED_VALUE" >&2
    exit 2
fi

TRAIN_DATA="${TRAIN_DATA:-data/train_set/merged_3_data_5k_each.csv}"
STUDENT_MODEL="${STUDENT_MODEL:-$STUDENT_MODEL_DEFAULT}"
TEACHER_MODEL="${TEACHER_MODEL:-$TEACHER_MODEL_DEFAULT}"
POOLING_METHOD="${POOLING_METHOD:-$POOLING_METHOD_DEFAULT}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CACHE_PATH="${CACHE_PATH:-cache/ggpkd/$PAIR_KEY/teacher_train.pt}"
GGPKD_CACHE_PATH="${GGPKD_CACHE_PATH:-cache/ggpkd/$PAIR_KEY/graph.pt}"
GGPKD_LOG_DIR="${GGPKD_LOG_DIR:-logs/ggpkd/$PAIR_KEY}"
SAVE_DIR="${SAVE_DIR:-models/ggpkd/$PAIR_KEY}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"
# Wall clock and peak memory land beside the run's own outputs, so a result
# and what it cost stay together even when SAVE_DIR is a per-seed directory.
STATS_FILE="${STATS_FILE:-$SAVE_DIR/run_stats.json}"

COMMAND=(
    "$PYTHON_BIN" main.py
    --method ggpkd
    --train_data "$TRAIN_DATA"
    --student_model "$STUDENT_MODEL"
    --teacher_model "$TEACHER_MODEL"
    --pooling_method "$POOLING_METHOD"
    --cache_path "$CACHE_PATH"
    --ggpkd_cache_path "$GGPKD_CACHE_PATH"
    --ggpkd_log_dir "$GGPKD_LOG_DIR"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --max_length "$MAX_LENGTH"
    --save_dir "$SAVE_DIR"
)

if [[ -n "$SEED_VALUE" ]]; then
    COMMAND+=(--seed "$SEED_VALUE")
fi

if [[ -n "$WEIGHTS_DIR" ]]; then
    COMMAND+=(--weights_dir "$WEIGHTS_DIR")
fi

COMMAND+=("$@")
run_with_stats "$STATS_FILE" "${COMMAND[@]}"
