"""Canonical GGPKD invariants shared by graph, sampler, and loss code.

These values are implementation policy, not independently tunable method
hyperparameters. Keeping their derivation here prevents the graph artifact,
candidate sampler, and criterion from silently using different conventions.
"""



# Used only by the fixed-bandwidth baseline and by low-level tests without an
# entropic-affinity artifact. Canonical GGPKD uses the per-row temperatures
# stored in the graph artifact.
FIXED_BANDWIDTH_TEMP = 0.05

# How much of each transition row the artifact keeps. This is a numerical-fidelity
# constant, not a method choice: any value small enough that the discarded tail
# cannot change a ranking gives the same objective, and the build reports
# `pool_residual_mass` / `pool_capped_rows` when it binds. It sits here for the
# same reason the encode-chunk sizes do -- changing it does not define a new
# objective.
TRUNCATION_TOLERANCE = 0.01

# Diagnostics and numerical/runtime choices are deliberately outside the method
# config. Changing them does not define a new GGPKD objective.
EPS_NORM = 1e-8
DIAG_TOPK = 8

# How the candidate pool is cut into student forward calls. Runtime policy, not
# method: with correct attention masks neither value can change a student
# embedding, so they only trade padded FLOPs against kernel launches -- which is
# why they live here rather than on GGPKDConfig, next to the other choices that
# do not define the objective.
#
# The trade-off is real and it is GPU-specific. Measured on the production corpus
# (13553 texts, mean 16.4 tokens, p50 13, p90 31) for one step's pool, against
# 23,000 real tokens:
#
#     chunk    pad=1    pad=8   pad=16   forward calls
#       128   28,119   33,158   36,408        11
#       256   29,834   35,702   38,829         6   <- the values below
#       512   40,226   47,413   52,997         3
#      1024   44,901   51,881   61,486         2
#
# That pool is ~1,400 unique candidates per step. It was ~4,445 while the method
# still drew 40 hard and 26 uniform negatives per anchor, and the same table then
# read 97,091 padded tokens at chunk 256 -- removing the negatives cut the encoder
# by 63% of its padded tokens and 67% of its forward calls, which is most of the
# training step.
#
# Two things that table says. Wider chunks buy fewer launches at strictly more
# padding, because a length-sorted chunk pads to its own longest member and a
# wider chunk spans a wider length band. And at these sequence lengths, rounding
# each chunk's width up to a multiple of 8 costs ~17% of all tokens by itself --
# a median-13-token text padded to 16 is 23% padding before any batching effect.
#
# Which side wins depends on whether the step is launch bound or FLOP bound. The
# last measured full-model run reported 470 MB peak and ~0.2 s/step for ~100k
# padded tokens on a 6-layer 384-wide student, a few percent of a modern GPU's
# arithmetic throughput -- so it was launch bound, and the no-negatives draw makes
# it more so, not less: a third of the tokens spread over a third of the calls.
# Larger chunks with pad 1 are the direction to test first. Run scripts/ggpkd/bench_encode.py to settle it on
# the actual device rather than adopting these numbers.
ENCODE_CHUNK_SIZE = 256
PAD_TO_MULTIPLE_OF = 8

# Support-selection arms for the fixed-budget ablation. The method itself no
# longer selects: with `diffusion_quota=None` the candidate set is the anchor's
# whole truncated transition row, and at that width all three sampling arms return
# the identical set. These exist so an arm can be given a budget smaller than the
# row and have the selection rule mean something again.
# `topk` is the method;
# proportional and uniform remove deterministic teacher ranking. `local_topk`
# is reserved for the clean no-diffusion control: it spends the same quota on
# r=1 relations only, while retaining the full artifact so the loss can keep the
# method's graph/ambient weighting unchanged.
#
#   topk          the whole quota taken deterministically by teacher mass. High
#                 teacher-mass coverage under the fixed relational budget.
#   proportional  the whole quota drawn by Gumbel top-k, no deterministic head.
#                 Cumulative coverage keeps growing; per-epoch coverage is noisier.
#   uniform       the quota drawn uniformly without replacement from the anchor's
#                 own pool. This is the *matched* random control: same graph, same
#                 budget, same column population, teacher relevance ordering
#                 discarded. Drawing uniformly from the whole corpus instead would
#                 give every drawn column diffusion target exactly zero and delete
#                 the objective rather than ablate the policy.
#   local_topk    deterministic top-k under P^1 only. Together with the direct
#                 relation target, this removes multi-hop diffusion without also
#                 changing candidate width, ambient calibration, or row loss.
#
# The two arms below leave the graph entirely. They exist for the controlled
# support study, where the question is not "which of the teacher's columns" but
# "does teacher relevance matter at all", so their columns carry no diffusion
# mass and they are only defined against relation_target="direct" -- the
# teacher's raw cosine, which exists for every pair. The config enforces that
# pairing, along with a matched --diffusion_quota and row_weight=0.
#
#   corpus_uniform  columns drawn uniformly from the whole corpus. The
#                   beyond-batch-but-not-teacher-informed control: it answers
#                   whether simply escaping the mini-batch is what helps.
#   rewired         degree-matched rewiring. Each anchor keeps the *number* of
#                   columns its own transition row has, but the endpoints are
#                   redrawn, so the degree profile of the graph survives and its
#                   semantics do not. Distinct from corpus_uniform only when the
#                   objective is degree-sensitive; reported to show that it is
#                   not, rather than assumed away.
SUPPORT_POLICIES = (
    "topk",
    "proportional",
    "uniform",
    "local_topk",
    "corpus_uniform",
    "rewired",
)

def candidate_budget(diffusion_quota: int, hard_neg_k: int, random_neg_k: int) -> int:
    """The candidate width is exactly the sum of its three source quotas."""
    quotas = (int(diffusion_quota), int(hard_neg_k), int(random_neg_k))
    if any(quota < 0 for quota in quotas):
        raise ValueError(f"candidate quotas must be non-negative, got {quotas}")
    budget = sum(quotas)
    if budget < 1:
        raise ValueError("at least one candidate must be requested")
    return budget


def hard_negative_pool_size(graph_k: int) -> int:
    """Use graph width as the offline hard-negative storage capacity."""
    graph_k = int(graph_k)
    if graph_k < 1:
        raise ValueError(f"graph_k must be positive, got {graph_k}")
    return graph_k
