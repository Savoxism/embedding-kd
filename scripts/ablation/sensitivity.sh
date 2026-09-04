#!/usr/bin/env bash
# Table 4 (appendix): Top-K quota and row-weight sensitivity.
#
# Top-K uses 0.5x / 1x / 2x the graph-derived quota. The 1x row is the shared
# full model. The 0.5x and 2x arms keep total candidate width fixed by trading
# support slots against hard/random negatives in their canonical 40:26 ratio.
# Row weight uses 0 / 0.5 / 1 / 1.5. Lambda=0 reuses components/no_row and
# lambda=1 reuses full/full.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"

    budget_source=(--artifact "$(base_graph_path)")
    if [[ -n "${CANONICAL_QUOTA:-}" ]]; then
        budget_source=(--quota "${CANONICAL_QUOTA}")
    fi

    read -r quota_half hard_half random_half < <(
        "${PYTHON_BIN}" scripts/ablation/budget.py \
            "${budget_source[@]}" \
            --hard "${HARD_NEG_K}" --random "${RANDOM_NEG_K}" \
            --multiplier 0.5
    )
    GRAPH_KEY=base run_arm sensitivity topk_half "${seed}" \
        --diffusion_quota "${quota_half}" \
        --hard_neg_k "${hard_half}" \
        --random_neg_k "${random_half}"

    read -r quota_double hard_double random_double < <(
        "${PYTHON_BIN}" scripts/ablation/budget.py \
            "${budget_source[@]}" \
            --hard "${HARD_NEG_K}" --random "${RANDOM_NEG_K}" \
            --multiplier 2
    )
    GRAPH_KEY=base run_arm sensitivity topk_double "${seed}" \
        --diffusion_quota "${quota_double}" \
        --hard_neg_k "${hard_double}" \
        --random_neg_k "${random_double}"

    run_no_row "${seed}"
    GRAPH_KEY=base run_arm sensitivity row_weight_0_5 "${seed}" --row_weight 0.5
    GRAPH_KEY=base run_arm sensitivity row_weight_1_5 "${seed}" --row_weight 1.5
done
