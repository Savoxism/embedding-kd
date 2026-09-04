#!/usr/bin/env bash
# Generate the Base / Batch-relational KD / GGPKD geometry-error figure after
# the support ablation has completed for all requested seeds.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

seed_csv="$(printf '%s' "${SEEDS}" | tr ' ' ',')"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ggpkd-matplotlib}" \
"${PYTHON_BIN}" scripts/ablation/geometry_heatmap.py \
    --runs-root "${ABL_ROOT}" \
    --train-data "${TRAIN_DATA}" \
    --teacher-cache "${TEACHER_CACHE}" \
    --graph-artifact "$(base_graph_path)" \
    --student-model "${STUDENT_MODEL}" \
    --seeds "${seed_csv}" \
    --probe-size "${HEATMAP_PROBE_SIZE:-2048}" \
    --max-length 128 \
    --temperature "${HEATMAP_TEMPERATURE:-0.05}" \
    --output "${HEATMAP_OUTPUT:-latex/figures/fig_geometry_heatmap}"
