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

# Deterministic head of the per-scale support draw (top-m columns kept before the
# Gumbel tail). Variance control for the sampler, not a modelling choice: it was
# never swept, and the support it protects is the same mass the Gumbel draw is
# already proportional to.
DETERMINISTIC_TOPM = 2

# Support-selection arms for the fixed-budget ablation. Only `hybrid` is the
# method; the other three exist to answer what the head and the tail each buy.
#
#   hybrid        deterministic top-DETERMINISTIC_TOPM head, then a Gumbel top-k
#                 tail drawn proportionally to the remaining teacher mass. GGPKD.
#   topk          the whole quota taken deterministically by teacher mass. High
#                 coverage on step one, and the same columns every epoch, so
#                 cumulative exposed mass plateaus.
#   proportional  the whole quota drawn by Gumbel top-k, no deterministic head.
#                 Cumulative coverage keeps growing; per-epoch coverage is noisier.
#   uniform       the quota drawn uniformly without replacement from the anchor's
#                 own pool. This is the *matched* random control: same graph, same
#                 budget, same column population, teacher relevance ordering
#                 discarded. Drawing uniformly from the whole corpus instead would
#                 give every drawn column diffusion target exactly zero and delete
#                 the objective rather than ablate the policy.
SUPPORT_POLICIES = ("hybrid", "topk", "proportional", "uniform")

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
