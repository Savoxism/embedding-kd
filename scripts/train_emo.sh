#!/bin/bash

echo "======================================"
echo "Training with EMO method"
echo "======================================"

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

METHOD="emo"
TRAIN_DATA="../data/test_debug.csv"
STUDENT_MODEL="..\model_hub\MiniLMv2-L6-H384-distilled-from-BERT-Base\MiniLM-L6-H384-distilled-from-BERT-Base"
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE=32
EPOCHS=5
LR=2e-5
MAX_LENGTH=256
SAVE_DIR="checkpoints/emo"

python3 ../main.py \
    --method $METHOD \
    --train_data $TRAIN_DATA \
    --student_model $STUDENT_MODEL \
    --teacher_model $TEACHER_MODEL \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max_length $MAX_LENGTH \
    --save_dir $SAVE_DIR

echo ""
echo "Training completed!"
echo "======================================"
