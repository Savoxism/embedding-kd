#!/bin/bash
set -euo pipefail

echo "======================================"
echo "Training with HeatGeo method"
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-iclr-mdd-heatgeo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-heatgeo_qwen3_4b_to_bert_base}"
export WANDB_MODE="${WANDB_MODE:-online}"

METHOD="heatgeo"
TRAIN_DATA="${TRAIN_DATA:-data/train_set/merged_3_data_5k_each.csv}"
STUDENT_MODEL="${STUDENT_MODEL:-google-bert/bert-base-uncased}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-Embedding-4B}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
SGC_WEIGHT="${SGC_WEIGHT:-0.05}"
SAVE_DIR="${SAVE_DIR:-models/heatgeo/qwen3_4b_to_bert_base}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"

COMMAND=(
    python3 main.py
    --method "$METHOD"
    --train_data "$TRAIN_DATA"
    --student_model "$STUDENT_MODEL"
    --teacher_model "$TEACHER_MODEL"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --max_length "$MAX_LENGTH"
    --sgc_weight "$SGC_WEIGHT"
    --save_dir "$SAVE_DIR"
)

if [[ -n "$WEIGHTS_DIR" ]]; then
    COMMAND+=(--weights_dir "$WEIGHTS_DIR")
fi

COMMAND+=("$@")
"${COMMAND[@]}"
