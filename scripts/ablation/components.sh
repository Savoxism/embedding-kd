#!/usr/bin/env bash
# Table 3: ambient x row decomposition under the canonical R={1} protocol.
#
# Rows produced/reused:
#   full                  graph relation + ambient + row
#   no_ambient            graph relation + row
#   no_row                graph relation + ambient
#   no_ambient_no_row     graph relation only
#   no_graph_support      ambient-only on matched uniform corpus candidates

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"

    GRAPH_KEY=base run_arm components no_ambient "${seed}" \
        --no_ambient
    run_no_row "${seed}"
    GRAPH_KEY=base run_arm components no_ambient_no_row "${seed}" \
        --no_ambient --row_weight 0
    run_no_graph_support "${seed}"
done
