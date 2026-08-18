# Walk-Sampled Kernel Matching

## Overview

The diffusion loss only ever supervises *anchor* rows of the teacher operator: every KL term in `L_diff` is conditioned on a batch anchor. The rows at all other corpus nodes go unsupervised even though several hundred of them are already encoded as candidates on every step.

Walk-sampled kernel matching supervises those rows too. A random walk on the teacher transition matrix picks which rows to supervise -- so the sampling measure is the walk's occupancy around the anchor -- and at each visited node the student is trained to reproduce the teacher's one-step kernel.

This is **not** a trajectory objective. A first-order walk scored with teacher forcing factorizes into independent one-step terms, so a path likelihood would be an unbiased but high-variance estimator of exactly the dense KL used here. The transition rows are already stored, so the dense form is used and no variance is paid.

The trajectory still does real work, but at the level of *identifiability* rather than the loss: each traversed edge constrains two rows that contain each other as columns, and cosine symmetry forces their similarity offsets to agree. Chaining along the walk propagates that identification through the graph, reducing the gauge freedom of the objective from one offset per anchor to one per connected component (Proposition `prop:walkgauge`).

## Configuration

In `config/heatgeo_config.py`:

```python
num_walks = 4          # Walks per anchor per epoch (0 = disabled)
walk_length = 4        # Steps per walk
walk_weight = 0.5      # Lambda in L = L_diff + lambda * L_walk
walk_temp = 0.05       # Student temperature; must equal graph_temp
walk_start_epoch = 1   # Curriculum: enable from this epoch
```

`walk_temp` is tied to `graph_temp` on purpose. The teacher row is itself a softmax of teacher cosines at `graph_temp`, so equal temperatures make the target exactly attainable and give the term an infimum of 0. Detuning it turns the achievable solution set from a shift family into an affine one and breaks the gauge argument.

Set `num_walks = 0` to disable; the run is then bit-identical to diffusion-only HeatGeo.

## How It Works

### Offline: graph build

`src/heatgeo/graph_builder.py` stores two extra fields in the artifact:

- `transition_neighbors` -- `[N, max_degree]`, neighbor list per node, `-1` padded
- `transition_probs` -- `[N, max_degree]`, the teacher transition row

Both are handed to the criterion as non-persistent buffers (int32 / float32), which is what makes the dense target free at train time.

### Per-epoch sampling

`src/heatgeo/candidate_sampler.py`, `sample_with_walks()`:

1. Samples `num_walks` trajectories of `walk_length` steps from each anchor, following the teacher transition matrix.
2. Injects visited nodes into the candidate set, replacing random negatives, so the student encodes them with zero extra forward passes.
3. Returns `(candidate_idx, teacher_probs, walk_paths)` with `walk_paths` of shape `[M, L+1]`.

The anchor is **not** injected: the loss drops step 0, and injecting it would burn a candidate slot that every scale masks out again.

### Training-time loss

`src/criterions/heatgeo_distillation.py`, `_compute_walk_loss()`.

Sources are the nodes visited at steps `t >= 1`, deduplicated and weighted by visit count. For a source $j$ the column set is the teacher's own support intersected with the batch pool:

$$\Omega^w_j = \mathcal{P}_B \cap \mathcal{M}_k^T(j)$$

$$\mathcal{L}_{\text{walk}} = \sum_j \nu_B(j)\, D_{\mathrm{KL}}\!\left( P^T_j\big|_{\Omega^w_j} \,\Big\|\, \mathrm{softmax}_{\Omega^w_j}\!\big(\cos(s_j, s_{j'})/\tau_w\big) \right)$$

Restricting the denominator to $\mathcal{M}_k^T(j)$ is what keeps this compatible with the column-domain split in `L_diff`: every column carries real teacher mass, so nothing is pushed down for the accidental reason that it was drawn for a different anchor. Hard negatives -- high teacher cosine, excluded from the mutual-kNN graph by construction -- can never enter $\Omega^w_j$ and are never touched by this term.

Rows with fewer than two live columns are dropped: a one-column softmax has no freedom and contributes exactly 0.

### Total loss

$$\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda\,\mathcal{L}_{\text{walk}}$$

Two terms, nothing else. `lambda_sim`, `lambda_cosine`, `lambda_infonce`, `lambda_simcse` are all 0 in this config; the CoSENT and Sinkhorn code paths have been removed from the criterion.

Because `L_walk` is a KL with an infimum of 0, it shares the scale of `L_diff` and `walk_weight` is a real trade-off rather than a units conversion.

## Monitoring Metrics

| Metric | Description |
|:---|:---|
| `loss_walk` | Weighted contribution, `walk_weight * walk_kl` |
| `walk_kl` | Raw KL, occupancy-weighted mean over supervised rows |
| `walk_teacher_entropy` | Entropy of the restricted teacher rows -- the KL's own difficulty |
| `walk_eff_denom` | Mean size of the softmax denominator per supervised row |
| `walk_rows` | Number of rows supervised this step |
| `walk_valid_ratio` | Visits that produced a usable row, over total visits |
| `walk_node_hit_ratio` | Visits landing in the pool, over total visits |

Read them together. A low `walk_node_hit_ratio` means walks leave the candidate set -- raise `candidate_size` or lower `walk_length`. A healthy hit ratio with a low `walk_valid_ratio` means walk nodes arrive but their *neighbors* do not, so rows fall below the two-column minimum -- raise `diffusion_quota` or `num_walks`. A `walk_eff_denom` near 2 means the term is technically active but nearly vacuous.

`walk_kl` near `walk_teacher_entropy` at the start is expected; what matters is that it falls while `walk_eff_denom` stays put.

## Ablation Experiments

```
(a) Diffusion only:        num_walks = 0
(b) Injection control:     num_walks = 4, walk_weight = 0   <- isolates the loss from
                                                               the changed negative mix
(c) Full:                  num_walks = 4, walk_length = 4, walk_weight = 0.5
(d) Walk length sweep:     walk_length in {2, 4, 6, 8}
(e) Walk weight sweep:     walk_weight in {0.1, 0.3, 0.5, 1.0}
(f) Curriculum:            walk_start_epoch in {0, 1, 2}
(g) Source selection:      walk-sampled rows vs. rows drawn uniformly from the pool
```

(b) is not optional. Enabling walks replaces up to `num_walks * walk_length` random negatives with walk nodes, which changes the candidate composition that `L_diff` and the ambient scale see. Without (b), the (c)-(a) gap conflates the walk loss with a different negative distribution.

(g) is the honest test of the walk itself: the loss only needs a *set of rows*, so if occupancy-weighted selection does not beat uniform selection from the pool, the walk mechanism is not earning its place -- only the extra supervision is.

## Technical Notes

- **Backward compatible**: `num_walks = 0` reproduces diffusion-only HeatGeo exactly.
- **Reproducible**: walk sampling is seeded by `(seed, epoch, idx)`.
- **Vectorized**: the corpus-to-pool lookup is a `searchsorted` on the sorted pool indices, entirely on device. No host sync inside the loss.
- **Memory**: the dense target is `[n_rows, pool_size]`, with `n_rows <= B * M * L` after dedup; roughly 4 MB at the default configuration.
- **Requires `share_in_batch=True`**: without a cross-anchor pool almost no row reaches two live columns. The criterion warns once and disables the term.
- **Gradient flow**: rows and columns both come from `pool_norm`, so the term trains the candidate encoder along the same path as the diffusion KL.
