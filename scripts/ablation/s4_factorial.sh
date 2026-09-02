#!/usr/bin/env bash
# S4 -- ambient x row, full 2x2.  P0, 3 seeds, 9 new runs.
#
# Question: what does each of the two auxiliary mechanisms actually do, and do
# they interact? Factorial rather than one-at-a-time deletion: L_row supervises
# rows the ambient scale also touches, so the marginal effect of deleting one is
# not the same with and without the other, and two separate deletions cannot show
# that.
#
# This is also the arm with the largest upside outside S1. Pair-classification is
# currently *down* against TALAS while STS is up; the ambient scale is the term
# that calibrates a single global cosine threshold, which is exactly what pair-cls
# reads. If `no_ambient` regresses pair-cls further, the trade-off has a
# mechanism; if it does not, the regression is unexplained and must be reported
# as such rather than attributed.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"                                                       # ambient + row
    run_arm s4 no_ambient "${seed}" --no_ambient                             #        + row
    run_arm s4 no_row     "${seed}" --row_weight 0                           # ambient
    run_arm s4 neither    "${seed}" --no_ambient --row_weight 0              # neither
done
