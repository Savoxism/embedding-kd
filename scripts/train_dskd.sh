#!/bin/bash

echo "======================================"
echo "Training with DSKD method"
echo "======================================"

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

METHOD="dskd"
TRAIN_DATA="..\data\test_debug.csv"
STUDENT_MODEL="..\model_hub\MiniLMv2-L6-H384-distilled-from-BERT-Base\MiniLM-L6-H384-distilled-from-BERT-Base"
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE=32
EPOCHS=10
LR=2e-5
MAX_LENGTH=256
SAVE_DIR="checkpoints/dskd"

python3 ../main.py \
    --method $METHOD \
    --train_data $TRAIN_DATA \
    --student_model $STUDENT_MODEL \
    --teacher_model $TEACHER_MODEL \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max_length $MAX_LENGTH \
    --save_dir $SAVE_DIR \
    --w_task 1.0 \
    --alpha_dtw 1.0 \
    --num_workers 2

echo "======================================"
echo "Training completed!"
echo "======================================"
