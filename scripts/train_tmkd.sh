#!/bin/bash
set -euo pipefail

echo "======================================"
echo "Training with TMKD method"
echo "======================================"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="false"

TRAIN_DATA="${TRAIN_DATA:-../data/test_debug.csv}"
STUDENT_MODEL="${STUDENT_MODEL:-bert-base-uncased}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
SAVE_DIR="${SAVE_DIR:-checkpoints/tmkd}"
LAMBDA_TMKD="${LAMBDA_TMKD:-1.0}"
TMKD_BLOCK_SIZE="${TMKD_BLOCK_SIZE:-512}"
TMKD_MODE="${TMKD_MODE:-full}"

python3 ../main.py \
    --method tmkd \
    --train_data "$TRAIN_DATA" \
    --student_model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max_length "$MAX_LENGTH" \
    --save_dir "$SAVE_DIR" \
    --lambda_tmkd "$LAMBDA_TMKD" \
    --tmkd_block_size "$TMKD_BLOCK_SIZE" \
    --tmkd_mode "$TMKD_MODE"
