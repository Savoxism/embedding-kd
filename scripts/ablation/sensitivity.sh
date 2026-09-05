#!/usr/bin/env bash
# Table 5: Top-K support-quota and row-weight sensitivity.
#
# Top-K uses 0.5x / 0.75x / 1x / 1.5x / 2x / 2.5x / 3x the graph-derived
# quota. Non-default arms keep total candidate width fixed by trading support
# slots against hard/random negatives in their canonical ratio. Row weight
# uses 0 / 0.1 / 0.25 / 0.5 / 0.75 / 1. Lambda=0 and lambda=1 reuse shared arms.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_topk_multiplier() {
    local seed="$1"
    local multiplier="$2"
    local arm="$3"
    local budget_source=(--artifact "$(base_graph_path)")
    if [[ -n "${CANONICAL_QUOTA:-}" ]]; then
        budget_source=(--quota "${CANONICAL_QUOTA}")
    fi
    local quota hard random
    read -r quota hard random < <(
        "${PYTHON_BIN}" scripts/ablation/budget.py \
            "${budget_source[@]}" \
            --hard "${HARD_NEG_K}" --random "${RANDOM_NEG_K}" \
            --multiplier "${multiplier}"
    )
    GRAPH_KEY=base run_arm sensitivity "${arm}" "${seed}" \
        --diffusion_quota "${quota}" \
        --hard_neg_k "${hard}" \
        --random_neg_k "${random}"
}

for seed in ${SEEDS}; do
    run_full "${seed}"

    run_topk_multiplier "${seed}" 0.5 topk_0_5x
    run_topk_multiplier "${seed}" 0.75 topk_0_75x
    run_topk_multiplier "${seed}" 1.5 topk_1_5x
    run_topk_multiplier "${seed}" 2 topk_2x
    run_topk_multiplier "${seed}" 2.5 topk_2_5x
    run_topk_multiplier "${seed}" 3 topk_3x

    run_no_row "${seed}"
    GRAPH_KEY=base run_arm sensitivity row_0_1 "${seed}" --row_weight 0.1
    GRAPH_KEY=base run_arm sensitivity row_0_25 "${seed}" --row_weight 0.25
    GRAPH_KEY=base run_arm sensitivity row_0_5 "${seed}" --row_weight 0.5
    GRAPH_KEY=base run_arm sensitivity row_0_75 "${seed}" --row_weight 0.75
done
