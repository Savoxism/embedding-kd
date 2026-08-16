# Random Walk Trajectory Distillation

## Overview

Walk trajectory distillation augments the standard HeatGeo diffusion loss with a step-by-step path-following objective. While diffusion matching asks "where do you end up after r steps?", walk distillation asks "can you follow the same path the teacher walks?" -- a stronger constraint that captures intermediate-node ordering and local manifold curvature.

## Configuration

In `config/heatgeo_config.py`:

```python
# Random Walk Trajectory Distillation
num_walks = 4          # Walks per anchor per epoch (0 = disabled)
walk_length = 4        # Steps per walk (recommended <= max(diffusion_scales))
walk_weight = 0.5      # Loss weight relative to L_diff
walk_temp = 0.07       # Student softmax temperature for walk transitions
walk_start_epoch = 1   # Curriculum: enable walk loss from this epoch
walk_topk = 128        # Top-K masking for walk softmax denominator
```

To disable walk distillation entirely, set `num_walks = 0`.

## How It Works

### Offline: Graph Build

`src/heatgeo/graph_builder.py` stores additional fields in the graph artifact:

- `transition_neighbors`: `[N, max_degree]` -- neighbor list per node
- `transition_probs`: `[N, max_degree]` -- corresponding transition probabilities

Storage cost: ~24MB for N=15k, k=200.

### Per-Epoch Sampling

`src/heatgeo/candidate_sampler.py` via `sample_with_walks()`:

1. Samples M random walks from each anchor following the teacher transition matrix
2. Injects walk nodes into the candidate set (replacing random negatives)
3. Returns `(candidate_idx, teacher_probs, walk_paths)` where `walk_paths` has shape `[M, L+1]`

Walk nodes are injected into the candidate set so the student encoder encodes them with zero extra forward passes.

### Training-Time Walk Loss

`src/criterions/heatgeo_distillation.py` via `_compute_walk_loss()`:

For each walk step from node $j_t$ to $j_{t+1}$:

$$p^S(j_{t+1} \mid j_t) = \text{softmax}_{j' \in \text{pool}} \left( \frac{\cos(s_{j_t}, s_{j'})}{\tau_w} \right)$$

$$\mathcal{L}_{\text{walk}} = -\frac{1}{|\text{valid steps}|} \sum_{\text{valid steps}} \log p^S(j_{t+1} \mid j_t)$$

Key details:

- Softmax runs over the **shared pool** -- reuses already-encoded embeddings
- Walk steps with nodes outside the pool are skipped (`walk_valid_ratio` metric tracks this)
- Self-transitions ($j_t = j'$) are masked out
- Optional top-K masking narrows the softmax denominator to the K most similar pool nodes

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda_{\text{walk}} \cdot \mathcal{L}_{\text{walk}}$$

Walk loss is only active when `epoch >= walk_start_epoch` (curriculum learning).

## Monitoring Metrics

| Metric | Description |
|:---|:---|
| `loss_walk` | Walk NLL loss value (after weighting) |
| `walk_nll` | Raw NLL (before multiplying by `walk_weight`) |
| `walk_valid_steps` | Number of valid walk steps (both nodes present in pool) |
| `walk_valid_ratio` | Ratio of valid to total steps (target: > 0.7) |

If `walk_valid_ratio` is low, the walks are venturing too far outside the candidate set. Fix by increasing `candidate_size` or decreasing `walk_length`.

## Ablation Experiments

```
(a) HeatGeo baseline:           num_walks = 0
(b) HeatGeo + walk:             num_walks = 4, walk_length = 4, walk_weight = 0.5
(c) Walk length sweep:          walk_length in {2, 4, 6, 8}
(d) Walk weight sweep:          walk_weight in {0.1, 0.3, 0.5, 1.0}
(e) Curriculum ablation:        walk_start_epoch in {0, 1, 2}
(f) Temperature sweep:          walk_temp in {0.04, 0.05, 0.07, 0.10}
```

## Comparison with Flow-Matching Distillation

| Aspect | Flow-Matching Distillation | Walk Trajectory Distillation |
|:---|:---|:---|
| Goal | Reduce NFE at inference | Add supervision at training |
| Mechanism | Self-consistency along ODE | Teacher-forcing NLL per step |
| Space | Continuous ($t \in [0,1]$) | Discrete on graph ($r \in \mathbb{Z}$) |
| Inference change | Yes (fewer steps) | No (student still uses 1 forward pass) |

## Technical Notes

- **Backward-compatible**: `num_walks=0` runs identically to the original HeatGeo
- **Reproducible**: walk sampling is seeded by `(seed, epoch, idx)` for deterministic results
- **Memory**: walk loss creates a tensor of shape `[n_valid, pool_size]` where `n_valid ~ B * M * L ~ 512` and `pool_size ~ 1000`, totaling ~2MB float32
- **Gradient flow**: walk loss backpropagates through `pool_norm` to the student encoder, sharing the same computation path as the diffusion KL
