#!/usr/bin/env bash
# N1 -- hard/uniform negative mix at a fixed total quota.  P1.
#
# The total negative quota is held at 66 (the method's 40 hard + 26 uniform), so
# the candidate width and therefore the encoder budget are identical across arms.
# This closes the one gap the config still records as open: the hard:random split
# has never been shown flat, and until it is, 40/26 is a tuned pair of numbers
# rather than a derived budget. A flat result here is a *good* result -- it
# collapses two knobs into one.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS_N1:-42}; do
    run_full "${seed}"                                                        # 40 / 26
    run_arm n1 all_hard    "${seed}" --hard_neg_k 66 --random_neg_k 0
    run_arm n1 all_uniform "${seed}" --hard_neg_k 0  --random_neg_k 66
done
