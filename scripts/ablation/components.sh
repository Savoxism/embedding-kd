#!/usr/bin/env bash
# Table 3: component deletions. Result columns are E_hat and Avg.
#
# Rows produced/reused:
#   full                  ambient + local + multi-hop + row
#   no_multi_hop          R={1}; removes only r>1 propagation
#   direct_target         same selected nodes, raw teacher cosine target
#   no_ambient            graph relation + row only
#   no_row                anchor-centered relation objective only
#   no_ambient_no_row     graph relation only
#   no_graph_diffusion    ambient-only on matched uniform corpus candidates

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"

    # Use the full model's derived Top-K quota so R={1} changes the scale ladder,
    # not the candidate budget.
    quota="$(canonical_quota)"
    GRAPH_KEY=r1 run_arm components no_multi_hop "${seed}" \
        --diffusion_scales 1 --diffusion_quota "${quota}"

    GRAPH_KEY=base run_arm components direct_target "${seed}" \
        --relation_target direct
    GRAPH_KEY=base run_arm components no_ambient "${seed}" \
        --no_ambient
    run_no_row "${seed}"
    GRAPH_KEY=base run_arm components no_ambient_no_row "${seed}" \
        --no_ambient --row_weight 0
    run_no_graph_diffusion "${seed}"
done
