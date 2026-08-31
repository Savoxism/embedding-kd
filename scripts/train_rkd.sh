#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PAIR_KEY="${PAIR_KEY:-qwen3_0_6b_to_minilmv2_h384}"
SEED_VALUE="${SEED:-42}"
GPU_VALUE="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_ROOT/data/train_set/merged_3_data_5k_each.csv}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache/rkd}"
RUN_DIR="${RUN_DIR:-}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
PREPARE_CACHE_ONLY=0
DRY_RUN=0
EXTRA_ARGS=()

if [[ $# -gt 0 && "$1" != -* ]]; then
    PAIR_KEY="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            SEED_VALUE="${2:?--seed requires a value}"
            shift 2
            ;;
        --gpu)
            GPU_VALUE="${2:?--gpu requires a value}"
            shift 2
            ;;
        --prepare-cache)
            PREPARE_CACHE_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown launcher option: $1 (put main.py options after --)" >&2
            exit 2
            ;;
    esac
done

case "$PAIR_KEY" in
    qwen3_0_6b_to_minilmv2_h384)
        TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B"
        STUDENT_MODEL="nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base"
        POOLING_METHOD="last_token"
        ;;
    bge_m3_to_minilmv2_h768)
        TEACHER_MODEL="BAAI/bge-m3"
        STUDENT_MODEL="nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base"
        POOLING_METHOD="cls"
        ;;
    qwen3_4b_to_bert_base)
        TEACHER_MODEL="Qwen/Qwen3-Embedding-4B"
        STUDENT_MODEL="google-bert/bert-base-uncased"
        POOLING_METHOD="last_token"
        ;;
    *)
        echo "Unknown RKD pair: $PAIR_KEY" >&2
        echo "Expected qwen3_0_6b_to_minilmv2_h384, bge_m3_to_minilmv2_h768, or qwen3_4b_to_bert_base" >&2
        exit 2
        ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Project virtual-environment Python is not executable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "RKD training data not found: $TRAIN_DATA" >&2
    exit 2
fi
if ! [[ "$SEED_VALUE" =~ ^[0-9]+$ ]]; then
    echo "Seed must be a non-negative integer, got: $SEED_VALUE" >&2
    exit 2
fi

CACHE_PATH="$CACHE_ROOT/$PAIR_KEY/teacher_train.pt"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/checkpoints/rkd/$PAIR_KEY/seed_$SEED_VALUE}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$RUN_DIR/weights}"

export CUDA_VISIBLE_DEVICES="$GPU_VALUE"
export TOKENIZERS_PARALLELISM="false"

COMMAND=(
    "$PYTHON_BIN" "$REPO_ROOT/main.py"
    --method rkd
    --train_data "$TRAIN_DATA"
    --student_model "$STUDENT_MODEL"
    --teacher_model "$TEACHER_MODEL"
    --task_type pair_cls
    --pooling_method "$POOLING_METHOD"
    --cache_path "$CACHE_PATH"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LEARNING_RATE"
    --max_length 256
    --rkd_distance_weight 1
    --rkd_angle_weight 2
    --w_task 0
    --seed "$SEED_VALUE"
    --save_dir "$RUN_DIR"
    --no_wandb
)

if [[ "$PREPARE_CACHE_ONLY" -eq 1 ]]; then
    COMMAND+=(--prepare_cache_only)
else
    COMMAND+=(--weights_dir "$WEIGHTS_DIR" --final_weights_only)
fi
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    COMMAND+=("${EXTRA_ARGS[@]}")
fi

echo "RKD pair=$PAIR_KEY seed=$SEED_VALUE gpu=$GPU_VALUE"
echo "Teacher: $TEACHER_MODEL"
echo "Student: $STUDENT_MODEL"
echo "Teacher cache: $CACHE_PATH"
echo "Run directory: $RUN_DIR"
echo "Paper defaults: batch=$BATCH_SIZE epochs=$EPOCHS lr=$LEARNING_RATE distance=1 angle=2 task=0"

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'COMMAND'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}"
