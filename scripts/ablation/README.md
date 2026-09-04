# GGPKD paper ablations

This directory runs the four-table/two-figure experiment plan. The runners here
produce the trained arms; the existing training telemetry records final benchmark
scores, held-out geometry distortion (`teacher_weighted_distortion`, reported as
`E_hat`), encoded texts/tokens, time, and peak memory.

## Experiment groups

| Script | Paper output | Arms |
|---|---|---|
| `full.sh` | Main reference | canonical Top-K GGPKD |
| `s1_support.sh` | support table + coverage inputs | Top-K, proportional, uniform pool, batch-local, matched uniform corpus |
| `components.sh` | component table | full, no multi-hop, direct target, no ambient, no row, neither, no graph/diffusion |
| `sensitivity.sh` | appendix sensitivity table | Top-K 0.5x/1x/2x; row weight 0/0.5/1/1.5 |
| `heatmap.sh` | geometry figure | Student base / batch-relational KD / GGPKD |

Shared arms are stored once and reused. With seeds 42/43/44 the complete suite
contains 14 unique configurations per seed: **42 training runs**.

## Run

```bash
# Complete suite. The full model is warmed up first to build shared caches.
GPUS="0 1 2" bash scripts/ablation/run_all.sh

# Run everything, then generate the three-panel geometry heatmap.
MAKE_HEATMAP=1 GPUS="0 1 2" bash scripts/ablation/run_all.sh

# One group on one GPU.
GPU=0 bash scripts/ablation/components.sh

# Smoke protocol with one seed and isolated outputs.
SEEDS="42" EXPERIMENT_KEY="smoke" GPU=0 bash scripts/ablation/sensitivity.sh

# Validate all arm construction without loading models or touching run outputs.
DRY_RUN=1 CANONICAL_QUOTA=23 SEEDS="42" \
  GPUS="0 1 2" bash scripts/ablation/run_all.sh
```

Useful environment overrides include `SEEDS`, `GPUS`, `PLAN`, `EXPERIMENT_KEY`,
`ABL_ROOT`, `CACHE_ROOT`, `LR`, and `EPOCHS`. Use a new `EXPERIMENT_KEY` whenever
the locked protocol changes; this prevents stale full runs from being compared
with new arms.

The runners take an atomic per-arm file lock. It is therefore safe for parallel
groups to request shared rows such as `no_row` and `no_graph_diffusion`; only one
training process runs and the others reuse its completed output.

Each completed arm writes:

```text
runs/ablation/<pair>/<experiment>/<group>/<arm>/seed<seed>/
  .done
  arm.json
  run.json
  epochs.jsonl
  metrics.jsonl
  train.log
```

The canonical diffusion quota is graph-derived. `budget.py` resolves it and
constructs fixed-total-width Top-K sensitivity arms. The extreme
`no_graph_diffusion` deletion uses ambient-only supervision over uniform corpus
candidates at the same candidate width as the full method; it does not leak
teacher-graph selection through its candidate pool.

## Geometry heatmap

After `full` and `support/batch_local` have completed for every seed:

```bash
GPU=0 bash scripts/ablation/heatmap.sh
```

The script reproduces the fixed geometry probe, uses the teacher graph for one
shared row/column ordering, averages each trained method's error map over seeds,
and writes:

```text
latex/figures/fig_geometry_heatmap.pdf
latex/figures/fig_geometry_heatmap.png
latex/figures/fig_geometry_heatmap.json
latex/figures/fig_geometry_heatmap.npz
```

Every pixel is `p_teacher(i,j) * (cos_student(i,j)-cos_teacher(i,j))^2`.
Consequently, the mean row sum is exactly `E_hat`; the JSON records the per-seed
values and checkpoint provenance, and the NPZ preserves the plotted matrices.
