"""Canonical GGPKD invariants shared by graph, sampler, and loss code.

These values are implementation policy, not independently tunable method
hyperparameters. Keeping their derivation here prevents the graph artifact,
candidate sampler, and criterion from silently using different conventions.
"""

from collections.abc import Sequence

import numpy as np

# Used only by the fixed-bandwidth baseline and by low-level tests without an
# entropic-affinity artifact. Canonical GGPKD uses the per-row temperatures
# stored in the graph artifact.
FIXED_BANDWIDTH_TEMP = 0.05

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

# Support-selection arms for the fixed-budget ablation. `topk` is the method;
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
SUPPORT_POLICIES = ("topk", "proportional", "uniform", "local_topk")

# Coverage target for the derived diffusion quota: the support size is the
# smallest k whose top-k mixture mass reaches this fraction at the median anchor.
# 0.7 is where the measured payoff knee sits on Qwen3-0.6B -> MiniLMv2-H384
# (graph v9, seed 42): quota 14 (~tau 0.5) -> 24 (~tau 0.7) gained ~+0.3 avg,
# while 24 -> 44 (~tau 0.8) was a tie (75.29 vs 75.25) -- and the derived value
# on that graph, 23, lands on the tuned knee. The exposure ceiling there is 1.0,
# so the target is always reachable and graph_k never binds it.
ROW_COVERAGE_TAU = 0.7


def derive_diffusion_quota(pool_probs: np.ndarray, scales: Sequence[int]) -> int:
    """Support size needed for ROW_COVERAGE_TAU coverage at the median anchor.

    Replaces the hand-tuned diffusion_quota count: the answer is a deterministic
    function of the graph artifact (sorted mixture-row cumsums), so it costs one
    pass at startup instead of a training-run sweep, and it moves with the corpus
    and teacher instead of being retuned per pair.

    Args:
        pool_probs: (n_scales, n_items, width) diffusion pool rows, zero-padded.
        scales: the artifact's diffusion scales; weighted by omega_r = 1/r.
    """
    weights = normalized_diffusion_weights(scales).astype(np.float32)
    mixture = np.einsum("s,sij->ij", weights, np.asarray(pool_probs, dtype=np.float32))
    mixture = -np.sort(-mixture, axis=1)
    coverage = np.cumsum(mixture, axis=1)
    # Rows that never reach tau (tiny components whose whole pool mass is below
    # it) need their full support; argmax on an all-False row would claim k=1.
    reaches = coverage >= ROW_COVERAGE_TAU
    need = np.where(
        reaches.any(axis=1),
        reaches.argmax(axis=1) + 1,
        (mixture > 0).sum(axis=1).clip(min=1),
    )
    return int(np.median(need))


def diffusion_weights(scales: Sequence[int]) -> tuple[float, ...]:
    """Return the canonical unnormalized rule omega_r = 1 / r."""
    resolved = tuple(int(scale) for scale in scales)
    if not resolved or any(scale < 1 for scale in resolved):
        raise ValueError(f"diffusion scales must be positive, got {resolved}")
    return tuple(1.0 / scale for scale in resolved)


def normalized_diffusion_weights(scales: Sequence[int]) -> np.ndarray:
    weights = np.asarray(diffusion_weights(scales), dtype=np.float64)
    return weights / weights.sum()


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
