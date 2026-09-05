#!/usr/bin/env bash
# Table 2: fixed-budget support-selection controls.
#
# Full is the teacher Top-K arm. proportional and uniform_pool change only the
# selection policy inside the same diffusion pool. batch_local is the cheap
# random-cooccurrence baseline. no_graph_support is the encoder-width-matched
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
    run_no_graph_support "${seed}"
done

# Coverage and restriction error are sampler properties, so generate them once
# after the graph exists rather than coupling them to a training run.
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    seed_csv="$(printf '%s' "${SEEDS}" | tr ' ' ',')"
    "${PYTHON_BIN}" scripts/ablation/replay_coverage.py \
        --artifact "$(base_graph_path)" \
        --out "${COVERAGE_OUTPUT:-${ABL_ROOT}/tables/table2_coverage.csv}" \
        --epochs "${EPOCHS}" \
        --seeds "${seed_csv}" \
        --quota "$(canonical_quota)" \
        --hard-neg-k "${HARD_NEG_K}" \
        --random-neg-k "${RANDOM_NEG_K}" \
        --policies topk,proportional,uniform
fi
