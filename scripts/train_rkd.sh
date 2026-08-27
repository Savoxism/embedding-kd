#!/bin/bash

echo "======================================"
echo "Training with RKD method"
echo "======================================"

export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"

METHOD="rkd"
TRAIN_DATA="../data/train_set/merged_3_data_5k_each.csv"
STUDENT_MODEL="google-bert/bert-base-uncased"
TEACHER_MODEL="Qwen/Qwen3-Embedding-4B"
BATCH_SIZE=32
EPOCHS=5
LR=2e-5
MAX_LENGTH=256
SAVE_DIR="checkpoints/rkd"

# Park et al. (2019): lambda_RKD-D = 1, lambda_RKD-A = 2 (RKD-DA).
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
    --dist_ratio 1.0 \
    --angle_ratio 2.0 \
    --num_workers 2

echo "======================================"
echo "Training completed!"
echo "======================================"
