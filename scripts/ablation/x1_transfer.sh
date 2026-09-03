#!/usr/bin/env bash
# X1 -- does the support claim replicate on a different teacher family?  P1.
#
# BGE-M3 -> MiniLMv2-H768, two arms only: the full Top-k method against the
# uniform control. Replicating the core claim on a second teacher is worth more
# than any further hyperparameter sweep on the main setting, and two arms is what
# fits the budget.
#
# Separate pair => separate teacher cache, separate graph, separate output root;
# all three follow from PAIR_KEY, which is why it is one variable.
export PAIR_KEY="bge_m3_to_minilmv2_h768"
export TEACHER_MODEL="BAAI/bge-m3"
export STUDENT_MODEL="nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base"
export POOLING_METHOD="cls"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"                                            # Top-k support
    run_arm x1 uniform "${seed}" --support_policy uniform          # uniform control
done
