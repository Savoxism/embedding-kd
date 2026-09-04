#!/usr/bin/env bash
# Focused one-seed diffusion study.
#
# Runs/reuses four points:
#   components/no_diffusion_clean  local P^1 support + raw teacher-cosine target
#   components/no_multi_hop        R={1}
#   diffusion_scales/r_1_2         R={1,2}
#   full/full                       R={1,2,4}
#
# The clean control deliberately uses graph_base.pt. Its local_topk selector
# reads P^1 only, while retaining the full scale-weight sum so the graph/ambient
# balance is identical to the full method. The R sweep is the natural method
# family and therefore rebuilds scale-specific artifacts where required.
#
# Usage:
#   bash scripts/ablation/diffusion.sh
#   SEEDS="43" GPU=1 bash scripts/ablation/diffusion.sh
#   DRY_RUN=1 CANONICAL_QUOTA=23 bash scripts/ablation/diffusion.sh

SEEDS="${SEEDS:-42}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    # Builds/reuses graph_base.pt and supplies the R={1,2,4} endpoint.
    run_full "${seed}"
    quota="$(canonical_quota)"

    run_no_diffusion_clean "${seed}"

    # Reuse the existing component arm as the R={1} endpoint.
    GRAPH_KEY=r1 run_arm components no_multi_hop "${seed}" \
        --diffusion_scales 1 \
        --diffusion_quota "${quota}"

    GRAPH_KEY=r1_2 run_arm diffusion_scales r_1_2 "${seed}" \
        --diffusion_scales 1,2 \
        --diffusion_quota "${quota}"
done
