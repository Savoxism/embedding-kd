#!/usr/bin/env bash
# K1 -- notebook-matched Top-K validation suite.
#
# All arms use deterministic teacher Top-K support, direct_temp derived from the
# graph, and a fixed diffusion quota of 23. The two deletion arms isolate row
# reuse and ambient alignment. The scale ladder holds Top-K, row, ambient and the
# relational budget fixed while changing only R.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_arm k1 topk_row_ambient_t0 "${seed}" \
        --support_policy topk --row_weight 1 --row_start_epoch 1 \
        --diffusion_quota 23 --direct_temp 0

    run_arm k1 topk_no_row_t0 "${seed}" \
        --support_policy topk --row_weight 0 \
        --diffusion_quota 23 --direct_temp 0

    run_arm k1 topk_no_ambient_t0 "${seed}" \
        --support_policy topk --row_weight 1 --row_start_epoch 1 --no_ambient \
        --diffusion_quota 23 --direct_temp 0

    GRAPH_KEY="k1_r1" run_arm k1 topk_r1_t0 "${seed}" \
        --support_policy topk --row_weight 1 --row_start_epoch 1 \
        --diffusion_scales 1 --diffusion_quota 23 --direct_temp 0

    GRAPH_KEY="k1_r12" run_arm k1 topk_r12_t0 "${seed}" \
        --support_policy topk --row_weight 1 --row_start_epoch 1 \
        --diffusion_scales 1,2 --diffusion_quota 23 --direct_temp 0
done
