#!/usr/bin/env bash
# Three-panel old mini-batch / graph-aware mini-batch motivation figure.

SEEDS="${SEEDS:-42}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

read -r seed _ <<< "${SEEDS}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ggpkd-matplotlib}" \
"${PYTHON_BIN}" scripts/ablation/batching_heatmap.py \
    --runs-root "${ABL_ROOT}" \
    --train-data "${TRAIN_DATA}" \
    --teacher-cache "${TEACHER_CACHE}" \
    --graph-artifact "$(base_graph_path)" \
    --student-model "${STUDENT_MODEL}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length 128 \
    --bins "${BATCHING_HEATMAP_BINS:-72}" \
    --output "${BATCHING_HEATMAP_OUTPUT:-latex/figures/fig_batching_heatmap}"
