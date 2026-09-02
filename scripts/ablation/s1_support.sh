#!/usr/bin/env bash
# S1 -- fixed-budget support selection, plus the batch-local baseline.
# P0, 3 seeds, 15 new runs (+ the shared full runs).
#
# Question: is the gain bought by *teacher relevance*, or by random co-occurrence
# and extra compute? Two groups of arms answer two halves of that.
#
# Fixed-budget policies (uniform / top-K / proportional / hybrid). All four spend
# the same candidate budget on the same graph, so the only thing that moves is
# which columns the diffusion quota lands on. `uniform` draws from the anchor's
# own pool rather than the whole corpus -- outside the pool the diffusion target
# is exactly zero, which would delete the objective instead of ablating the
# policy, and the comparison would prove nothing.
#
# Baselines without graph support. `batch_local` is batch-local relational KD:
# relations among the texts that happen to share a minibatch, no graph, no
# candidate draw, no auxiliary rows -- the objective collapses to one KL against
# the teacher's similarity profile over the batch. It is the form prior
# batch-relational work takes, and it is deliberately NOT budget-matched: it
# encodes ~2B texts per step against the method's B + unique candidates. That
# gap is the arm's content, not a flaw in it, and collect.py prints the encode
# counts beside every score so it cannot be read as a like-for-like win.
#
# `uniform_corpus` is the budget-matched counterpart, and the arm to quote at a
# reviewer who says GGPKD only wins by encoding more. Same candidate width, same
# encoder budget, same ambient-only objective -- but the columns are drawn
# uniformly from the corpus instead of from the teacher's neighbourhood, so its
# diffusion mass is zero by construction and the graph group is dropped with it.
# Between the two of them, "more compute" and "structured support" are separated.
#
# Expect: hybrid has coverage near top-K at epoch 1 and keeps climbing (Figure 1,
# from replay_coverage.py), and reaches lower E_hat than uniform and top-K; both
# baselines sit far below every policy on coverage and E_hat.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# The matched baseline spends the whole candidate width on uniform corpus draws,
# so it has to know that width. The diffusion quota is derived from the graph at
# startup, not written down, so it is read back from the artifact -- hard-coding
# it would silently unmatch the budget the day the corpus changes. Resolved after
# the first run_full below, when the artifact is guaranteed to exist.

for seed in ${SEEDS}; do
    run_full "${seed}"                                            # head + proportional tail
    run_arm s1 topk         "${seed}" --support_policy topk         # teacher top-K
    run_arm s1 proportional "${seed}" --support_policy proportional # teacher-proportional
    run_arm s1 uniform      "${seed}" --support_policy uniform      # uniform over the pool

    # Batch-local relational KD: no graph at all, ~30x cheaper per step.
    run_arm s1 batch_local "${seed}" --batch_local --row_weight 0

    # Same objective, same candidate width as the method, uniform corpus columns.
    # This matches the *width*; the encode count then lands slightly above the
    # method's, because uniform corpus draws dedup less than draws concentrated on
    # a neighbourhood. The error is in the baseline's favour, which is the
    # direction it should be, and collect.py prints both counts.
    width="${CANDIDATE_WIDTH:-$("${PYTHON_BIN}" scripts/ablation/candidate_width.py \
        "${CACHE_ROOT}/graph_base.pt" "${HARD_NEG_K:-40}" "${RANDOM_NEG_K:-26}" \
        2>/dev/null || true)}"
    if [[ -z "${width}" ]]; then
        echo "[skip] s1/uniform_corpus/seed${seed}: could not read ${CACHE_ROOT}/graph_base.pt"
        echo "       to derive the candidate width; set CANDIDATE_WIDTH to force it."
    else
        run_arm s1 uniform_corpus "${seed}" \
            --relation_target ambient_only --row_weight 0 \
            --diffusion_quota 0 --hard_neg_k 0 --random_neg_k "${width}"
    fi
done
