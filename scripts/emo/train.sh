#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/run_stats.sh
source "$SCRIPT_DIR/../common/run_stats.sh"

echo "======================================"
echo "Training with EMO method"
echo "======================================"

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

METHOD="emo"
TRAIN_DATA="../../data/test_debug.csv"
STUDENT_MODEL="jim12345/MiniLMv2-L6-H384-distilled-from-BERT-Base"
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE=4
EPOCHS=5
LR=1e-5
MAX_LENGTH=256
SAVE_DIR="checkpoints/emo"

COMMAND=(
    python3 ../../main.py
    --method "$METHOD"
    --train_data "$TRAIN_DATA"
    --student_model "$STUDENT_MODEL"
    --teacher_model "$TEACHER_MODEL"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --max_length "$MAX_LENGTH"
    --save_dir "$SAVE_DIR"
)

STATS_FILE="${STATS_FILE:-$SAVE_DIR/run_stats.json}"
run_with_stats "$STATS_FILE" "${COMMAND[@]}"

echo ""
echo "Training completed!"
echo "======================================"
