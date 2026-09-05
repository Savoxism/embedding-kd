#!/usr/bin/env bash
# Table 1: canonical GGPKD on all three paper teacher -> student settings.
# Baseline methods keep their own launchers; this script produces the GGPKD rows
# under the same R={1} protocol used by Tables 2--5.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_KEY="${EXPERIMENT_KEY:-paper_r1_v2}"
RUNS_BASE="${RUNS_BASE:-runs/ablation}"
CACHE_BASE="${CACHE_BASE:-cache/ggpkd}"
LOG_BASE="${LOG_BASE:-logs/ggpkd}"

run_pair() {
    local pair="$1"
    local teacher="$2"
    local student="$3"
    local pooling="$4"

    PAIR_KEY="${pair}" \
    TEACHER_MODEL="${teacher}" \
    STUDENT_MODEL="${student}" \
    POOLING_METHOD="${pooling}" \
    ABL_ROOT="${RUNS_BASE}/${pair}/${EXPERIMENT_KEY}" \
    CACHE_ROOT="${CACHE_BASE}/${pair}/${EXPERIMENT_KEY}" \
    LOG_ROOT="${LOG_BASE}/${pair}/${EXPERIMENT_KEY}" \
    TEACHER_CACHE="${CACHE_BASE}/${pair}/teacher_train.pt" \
    bash "${SCRIPT_DIR}/full.sh"
}

run_pair \
    qwen3_0_6b_to_minilmv2_h384 \
    Qwen/Qwen3-Embedding-0.6B \
    nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base \
    last_token

run_pair \
    bge_m3_to_minilmv2_h768 \
    BAAI/bge-m3 \
    nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base \
    cls

run_pair \
    qwen3_4b_to_bert_base \
    Qwen/Qwen3-Embedding-4B \
    google-bert/bert-base-uncased \
    last_token
