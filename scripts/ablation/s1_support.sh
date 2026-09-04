#!/usr/bin/env bash
# Table 2 + coverage chart: support-selection controls.
#
# Full is the teacher Top-K arm. proportional and uniform_pool change only the
# selection policy inside the same diffusion pool. batch_local is the cheap
# random-cooccurrence baseline. no_graph_diffusion is the encoder-width-matched
# uniform-corpus baseline and is shared with the component table.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"
    GRAPH_KEY=base run_arm support proportional "${seed}" \
        --support_policy proportional
    GRAPH_KEY=base run_arm support uniform_pool "${seed}" \
        --support_policy uniform
    GRAPH_KEY=base run_arm support batch_local "${seed}" \
        --batch_local --row_weight 0
    run_no_graph_diffusion "${seed}"
done
