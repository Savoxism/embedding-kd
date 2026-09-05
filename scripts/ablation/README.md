# GGPKD paper experiment suite (R={1})

These runners implement the six paper tables under one locked protocol. The
canonical method uses teacher Top-K support, ambient alignment, auxiliary-row
reuse, and `diffusion_scales=1`. Multi-hop is an ablation axis, not part of the
reference arm.

## Tables and runners

| Table | Runner | Arms or source |
|---|---|---|
| 1. Main results | `main_results.sh` | Canonical GGPKD on all three teacher-student pairs |
| 2. Support selection | `s1_support.sh` | Top-K, proportional, uniform within pool, uniform corpus, batch-relational KD |
| 3. Components | `components.sh` | Full, no ambient, no row, neither, no graph support |
| 4. Radius | `radius.sh` | R={1}, R={1,2}, R={1,2,4} |
| 5. Sensitivity | `sensitivity.sh` | Top-K 0.5x/0.75x/1x/1.5x/2x/2.5x/3x; row weight 0/0.1/0.25/0.5/0.75/1 |
| 6. Efficiency | run telemetry | Wall time, seconds/step, encoded texts/tokens, and peak VRAM from the runs above |

`replay_coverage.py` creates Table 2's cumulative coverage and restriction-error
columns offline. It is called automatically by `s1_support.sh` after training.

The baseline rows of Table 1 retain their method-specific launchers. The main
runner here produces only the three GGPKD rows so that every GGPKD table uses
the same canonical protocol. GGPKD is evaluated once after epoch 5 and reports
that final model; it does not evaluate each epoch or select a validation
checkpoint. Every runner explicitly passes `--eval_every 0`; this is locked for
all 66 runs rather than inherited implicitly from a config default.

## Shared arms

Runs are stored by semantic group rather than duplicated per table:

- `full/full` is Table 1's canonical Qwen-MiniLM row, Table 2's Top-K row,
  Table 3's Full row, Table 4's R={1} row, and both default points in Table 5.
- `components/no_row` is reused as row weight 0.
- `shared/no_graph_support` is reused by Tables 2 and 3.

With seeds 42/43/44 there are 20 unique configurations on the primary
Qwen3-0.6B -> MiniLMv2-H384 setting (60 runs), plus six full-model runs for the
other two Table 1 settings: 66 training runs total. Table 6 adds no training.

## Run

```bash
# All GGPKD rows needed by Tables 1--5.
GPUS="0 1 2" bash scripts/ablation/run_all.sh

# Selected tables only.
PLAN="support components radius" GPUS="0 1 2" \
  bash scripts/ablation/run_all.sh

# One table on one GPU.
GPU=0 bash scripts/ablation/radius.sh

# Validate the complete arm matrix without model loading or GPU training.
DRY_RUN=1 CANONICAL_QUOTA=15 SEEDS="42" GPUS="0 1 2" \
  bash scripts/ablation/run_all.sh

# After all 66 runs finish, validate completeness and produce Tables 1--6 CSVs.
python scripts/ablation/collect.py
```

The default experiment key is `paper_r1_v2`, intentionally distinct from the
old `paper_v1` R={1,2,4}/hybrid artifacts. Override `EXPERIMENT_KEY` only when
starting another locked protocol.

Useful suite-wide overrides are `SEEDS`, `GPUS`, `PLAN`, `EXPERIMENT_KEY`,
`RUNS_BASE`, `CACHE_BASE`, `LOG_BASE`, `LR`, and `EPOCHS`. `ABL_ROOT`,
`CACHE_ROOT`, and `LOG_ROOT` are single-pair overrides and are rejected when a
`run_all.sh` plan contains `main`. Table-specific runners use
`PAIR_KEY=qwen3_0_6b_to_minilmv2_h384` unless explicitly overridden;
`main_results.sh` supplies the three paper pairs itself.

Each completed arm is compacted to exactly one persistent file:

```text
runs/ablation/<pair>/<experiment>/<group>/<arm>/seed<seed>/
  result.csv
```

JSONL telemetry, the console log, and final weights exist only while a run is in
progress. After the final-model evaluation is validated, they are compacted into
`result.csv` and deleted. Failed/interrupted runs retain their temporary files
for debugging. Graph/teacher tensor caches remain because subsequent arms reuse
them, but verbose graph-neighbour logs are removed.

The runners use atomic per-arm locks, so concurrent tables safely reuse shared
arms. Before reusing `result.csv`, they compare its stored request fingerprint,
including every locked parameter and a training-code fingerprint; a mismatch
fails instead of silently mixing protocols. Top-K sensitivity is a fixed-width allocation sweep:
support slots are traded against hard/random negatives in their canonical
ratio. Radius arms keep the R={1} support quota and the 50/50 ambient--graph
group weighting fixed, so the graph target radius is isolated.

## Reporting contract

- Tables 2--5 report mean +/- standard deviation and paired seed deltas for
  STS Avg, pair-classification Avg, classification Avg, overall Avg, and E_hat.
- Table 2 additionally reports final cumulative global teacher-mass coverage,
  restriction error, and encoded texts. Cached-pool residual mass is not
  renormalized away.
- Table 4 additionally reports wall time and encoded texts so the radius tradeoff
  is explicit.
- Table 6 reports mean training-step time, encoded texts/tokens, peak VRAM, and
  launcher end-to-end time. The latter is tagged by whether both teacher and
  graph caches existed before launch; the collector reports warm-cache
  end-to-end time separately. It is not labeled as pure training time because
  model initialization and final evaluation are included. Do not compare
  literature timing numbers measured on different hardware.

`collect.py` reads only compact `paper_r1_v2/result.csv` files, requires the
exact 66-run matrix by default, and writes `runs.csv` plus one CSV
per table under `runs/ablation/tables/paper_r1_v2/`. Pass `--allow-incomplete`
only for smoke-test inspection.

`figures.py` reads the compact CSVs. Weight-dependent geometry heatmaps are not
part of this suite because successful runs intentionally do not retain model
weights.
