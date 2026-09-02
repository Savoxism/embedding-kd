#!/usr/bin/env bash
# S2 -- multi-scale diffusion vs local-only.  P0, 3 seeds, 3 new runs.
#
# Question: does diffusion buy reach beyond the immediate neighbourhood, or is
# R={1} enough? R={1} is a different graph artifact (the pools are built per
# scale), so it gets its own GRAPH_KEY -- sharing the path would rebuild the
# graph on every alternation between this arm and every other ablation.
#
# Expect: R={1,2,4} lower E_hat and higher STS Avg at the same encoded-text count.
# Check encoded_texts_cum in the collected table: if R={1} encodes materially
# fewer texts the comparison is not budget-matched and the arm needs its quota
# raised, not its result reported.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"                                                     # R = {1,2,4}
    GRAPH_KEY="r1" run_arm s2 local_only "${seed}" --diffusion_scales 1     # R = {1}
done
