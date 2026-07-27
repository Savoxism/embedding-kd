#!/bin/bash

# ==============================================================================
# Script to run MDD (TWMD) method on A100 GPU
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Define paths and models
METHOD="mdd"
TEACHER_MODEL="Qwen/Qwen3-Embedding-4B"
STUDENT_MODEL="google-bert/bert-base-uncased"
TRAIN_DATA="data/train_set/merged_3_data_5k_each.csv"

# Hyperparameters (Optimized for A100 40GB/80GB)
BATCH_SIZE=32       # Increase to 64 if you have A100 80GB and want faster training
EPOCHS=5
NUM_WORKERS=4       # Fast data loading
LEARNING_RATE=2e-5

# Set wandb mode to online to sync with wandb servers
export WANDB_MODE="online"

echo "============================================================"
echo "Starting MDD (TWMD) Training on A100"
echo "Teacher Model: $TEACHER_MODEL"
echo "Student Model: $STUDENT_MODEL"
echo "Dataset: $TRAIN_DATA"
echo "============================================================"

# Run the training
python3 main.py \
    --method $METHOD \
    --teacher_model $TEACHER_MODEL \
    --student_model $STUDENT_MODEL \
    --train_data $TRAIN_DATA \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --num_workers $NUM_WORKERS \
    --learning_rate $LEARNING_RATE

echo "============================================================"
echo "Training Completed successfully!"
echo "Check ./twmd_checkpoints/ cho file best_model.pt"
echo "============================================================"
