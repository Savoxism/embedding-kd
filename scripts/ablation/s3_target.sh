#!/usr/bin/env bash
# S3 -- composed graph relations vs raw teacher cosine on the SAME nodes.
# P0, 3 seeds, 3 new runs.
#
# Question: is the win the diffusion target, or merely that the selection reaches
# farther nodes? This arm holds the selected columns, the temperature, the row
# set and the group weight fixed and swaps only what those columns are supervised
# against -- softmax(cos_T / tau_i) over the anchor's own draw instead of the
# composed multi-scale transition rows. It is the one control that separates the
# two, and the answer decides whether "composed graph relations" survives as a
# contribution or becomes a selection story.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"                                                  # diffusion target
    run_arm s3 direct_target "${seed}" --relation_target direct         # teacher cosine
done
