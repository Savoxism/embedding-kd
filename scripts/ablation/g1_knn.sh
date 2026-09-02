#!/usr/bin/env bash
# G1 -- kNN edge rule: mutual vs directed vs symmetrized.  P1.
#
# Screen on seed 42 first (SEEDS="42", the default here); only confirm on 43/44
# if the signal is stable AND degree is matched. The claim being tested is about
# the *graph*, so the primary evidence is the hubness block the build now writes
# into graph_stats -- indegree max / p99 / gini and the edge share held by the
# top 1% of nodes -- not the downstream average. Read those with
#     python scripts/ablation/collect.py --hubness
# Directed and symmetrized have systematically larger degree than mutual at the
# same graph_k, so a downstream difference between them is partly a budget
# difference; say so in the paper rather than reading it as an edge-rule effect.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS_G1:-42}; do
    run_full "${seed}"                                                              # mutual
    GRAPH_KEY="directed"    run_arm g1 directed    "${seed}" --knn_mode directed
    GRAPH_KEY="symmetrized" run_arm g1 symmetrized "${seed}" --knn_mode symmetrized
done
