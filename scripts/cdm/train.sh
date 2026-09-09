#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/run_stats.sh
source "$SCRIPT_DIR/../common/run_stats.sh"

echo "======================================"
echo "Training with CDM method"
echo "======================================"

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

METHOD="cdm"
TRAIN_DATA="../../data/test_debug.csv"
STUDENT_MODEL="jim12345/MiniLMv2-L6-H384-distilled-from-BERT-Base"
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE=32
EPOCHS=5
LR=2e-5
MAX_LENGTH=256
SAVE_DIR="checkpoints/cdm"

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
    --w_task 0.5
    --alpha_dtw 0.5
    --num_workers 2
    --debug
)

STATS_FILE="${STATS_FILE:-$SAVE_DIR/run_stats.json}"
run_with_stats "$STATS_FILE" "${COMMAND[@]}"

echo "======================================"
echo "Training completed!"
echo "======================================"
