#!/usr/bin/env bash
# Table 4: graph-radius / multi-hop study.
#
# The canonical endpoint is R={1}. R={1,2} and R={1,2,4} reuse its derived
# Top-K quota, hard/random quotas, optimizer, training schedule, and the 50/50
# ambient--graph group weighting. Only the graph target radius ladder changes.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"
    quota="$(canonical_quota)"

    GRAPH_KEY=r1_2 run_arm radius r_1_2 "${seed}" \
        --diffusion_scales 1,2 \
        --diffusion_quota "${quota}"

    GRAPH_KEY=r1_2_4 run_arm radius r_1_2_4 "${seed}" \
        --diffusion_scales 1,2,4 \
        --diffusion_quota "${quota}"
done
