"""Canonical RIPPLE invariants shared by graph, sampler, and loss code.

These values are implementation policy, not independently tunable method
hyperparameters. Keeping their derivation here prevents the graph artifact,
candidate sampler, and criterion from silently using different conventions.
"""

from collections.abc import Sequence

import numpy as np

# Used only by the fixed-bandwidth baseline and by low-level tests without an
# entropic-affinity artifact. Canonical RIPPLE uses the per-row temperatures
# stored in the graph artifact.
FIXED_BANDWIDTH_TEMP = 0.05

# Diagnostics and numerical/runtime choices are deliberately outside the method
# config. Changing them does not define a new RIPPLE objective.
EPS_NORM = 1e-8
DIAG_TOPK = 8


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
