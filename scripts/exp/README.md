# Experiments

Two generations live here.

## The staged sweeps (current — `experiments.md`)

One script per stage, run in order. The orchestrator stops after Stage 0 on
purpose: Stage 0 chooses the operating point, choosing it is a judgement call,
and no script may guess it.

```bash
GPUS=0,1,2,3            bash scripts/exp/run_all.sh   # Stage 0, then stop
column -s, -t runs/stage0_graph_k/results.csv | less -S
GRAPH_K=<winner> GPUS=0,1,2,3 bash scripts/exp/run_all.sh   # Stages 1-3
DRY_RUN=1 GRAPH_K=50    bash scripts/exp/run_all.sh   # print the plan only
```

| stage | script | question | runs |
|---|---|---|---|
| 0 | `stage0_graph_k.sh` | what operating point? also the full-model reference | 12 |
| 1 | `stage1_deletions.sh` | does each component earn its place? | 12 |
| 2B | `stage2b_ladder.sh` | which texts should an anchor be compared against? | 12/pair |
| 2C | `stage2c_dose_response.sh` | does the score follow the count, however it is bought? | 27 |
| 2D | `stage2d_components.sh` | ablation on the shipped objective | 9 |
| 3 | `stage3_main_table.sh` | the deliverable | 6–15 |

Two stages need no GPU: **A** is the formula $(B-1)k/(N-1)$ plus one exposure
curve from `coverage.py`, and **E** scores Stage 2D's checkpoints post-hoc with
`exp3_heldout_geometry.sh`.

`FROM` / `TO` restrict the range (`0`, `1`, `2b`, `2c`, `2d`, `3`). Everything
else — `GPUS`, `SEEDS`, `PAIR`/`PAIRS`, `CORPUS`, `CACHE_ROOT`, `DRY_RUN` — is
forwarded to the stage scripts.

### Re-running part of a stage

`ARMS=a,b` runs only those arms of a stage and builds only the graphs they use.
Export merges by `(experiment, pair, arm, seed)`, so the new rows replace the old
ones in the same `results.csv` instead of overwriting the file. The pieces the
2026-09-12 sweep got wrong, on `qwen3_0_6b_to_minilmv2_h384`:

```bash
export GRAPH_K=200 GPUS=0,1,2,3,4,5,6,7 JOBS_PER_GPU=2
ARMS=row_centers_random,uniform_target bash scripts/exp/stage1_deletions.sh
ARMS=student_knn                       bash scripts/exp/stage2b_ladder.sh
HALF=c2                                bash scripts/exp/stage2c_dose_response.sh
ARMS=full                              bash scripts/exp/stage2d_components.sh
bash scripts/exp/exp3_heldout_geometry.sh   # needs Stage 2D's full run tree
```

Stage 2D's graph keys are now `*_holdout`, so its old run tree scores against
`graph_main_holdout.pt` too (`RUN_ROOT=<old 2D root>`).

---

## The original motivation study (E1–E4)

Superseded by the stages above, and kept because `exp3_heldout_geometry.sh` is
still the held-out probe and `coverage.py` is still the exposure measurement.
Their numbers were all measured before the mutual filter, row truncation,
multi-hop diffusion and fixed-reference calibration were removed, so they
describe a different method — read them for relative structure only.

One `.sh` per experiment, three seeds each, every result exported to CSV under
`runs/<experiment>/`. The four answer four different questions, in order:

| | Question | Script | Runs | Output |
|---|---|---|---|---|
| E1 | Is batching deciding the learning signal? | `exp1_batch_intervention.sh` | 27 | `runs/exp1_batch/{results,coverage}.csv` |
| E2 | Does teacher relevance matter, or just leaving the batch? | `exp2_support_intervention.sh` | 15/pair | `runs/exp2_support/results.csv` |
| E3 | Does it generalize past the supervised edges? | `exp3_heldout_geometry.sh` | 0 (post-hoc) | `runs/exp3_heldout/heldout_geometry.csv` |
| E4 | Does it scale the way the hypothesis predicts? | `exp4_exposure_scaling.sh` | 84 | `runs/exp4_scaling/{results,exposure}.csv` |

Run order matters: **E1 is the gate**. If an in-batch relational objective does
not move with batch composition by more than the pointwise arm's seed noise, the
motivation is not real and E2–E4 are not worth their compute.

```bash
DRY_RUN=1 bash scripts/exp/exp1_batch_intervention.sh   # print the matrix, run nothing
GPUS=0,1,2,3 bash scripts/exp/exp1_batch_intervention.sh
bash scripts/exp/exp2_support_intervention.sh
bash scripts/exp/exp3_heldout_geometry.sh               # scores E2's checkpoints
bash scripts/exp/exp4_exposure_scaling.sh
```

Common environment variables: `GPUS` (default: every GPU `nvidia-smi` reports),
`SEEDS` (default `42,43,44`), `PAIR`, `DRY_RUN=1`, `CACHE_ROOT`, `RESULT_BASE`.
Each script's header documents its own.

## The shared harness

Every relational arm in E1, E2 and E4 trains the **same minimal objective**:

```
L = (1/|B|) sum_i KL( p_T(. | S_i) || p_S(. | S_i) )
```

one KL per anchor, over that anchor's own columns `S_i`, at that anchor's own
bandwidth `tau_i`. In flags: `--relation_target direct --no_ambient
--row_weight 0`. Only `S_i` changes between arms.

Two things about this are worth knowing before reading any result.

**Candidate sharing does not contaminate it.** The graph-scale softmax is masked
to each anchor's own draw (`diffusion_mask = self_mask | ~own_mask`,
`src/criterions/ggpkd_distillation.py:962`), so deduplicating the batch's
candidates across anchors changes the encoder cost and cannot change the loss.
The study gets an isolated per-anchor objective *and* the ~3x cheaper encode.

**The in-batch arm is temperature-matched.** `--batch_local --relation_target
direct` scores the batch columns at `tau_i`, the same temperature the
teacher-support arms use. The older `ambient_only` form of that baseline scores
at one global temperature, which would have made the arms differ in two things at
once.

Full GGPKD (`L_{r=0} + L_{r=1} + lambda L_row`) is deliberately **not** used
here. If the study's arms carried the ambient scale and the auxiliary rows, no
reader could tell whether a gain came from support selection or from those.

## What each CSV carries

`export_runs.py` writes one row per (arm, seed):

* `score_*` — nine benchmarks plus Avg-In / Avg-Out / Avg-All, in points.
* `cfg_*` — read from each run's own `run.json`, not from the arm label, so a
  mislabelled arm is visible.
* `cost_*`, `train_encoded_texts_cum` — arms are matched on **support size**, not
  on compute; a corpus-uniform draw deduplicates far worse than a teacher draw
  and encodes more texts per step. That difference belongs in the table.
* `train_*`, `geom_*` — final-epoch losses and the geometry probe, so a strange
  downstream number can be traced without re-running.
* `status` — `ok`, or why the row is incomplete. A partially finished sweep still
  exports.

Join keys across files: `(pair, arm, seed)` for E2/E3, `(n_items, batch_size)`
for E4's exposure file.

## Things to check in the output before trusting it

**`train_candidates_per_anchor` vs the requested budget.** The teacher arms are
given `SUPPORT_SIZE` (default `batch_size - 1` = 63) columns, but the real
transition rows average 66.8 with a minimum of 1, so anchors with shorter rows
get less. If this column sits well below 63, the teacher arms are carrying *less*
supervision than the in-batch arm, which biases the comparison against the
paper's claim — conservative, but it must be reported. Lower `SUPPORT_SIZE` if
the gap is large.

**`holdout_starved_rows` in the graph build log.** A row whose every edge landed
in the held-out 20% keeps its nearest neighbour anyway, so the "never supervised"
claim is false for that one edge. Expected to be 0 at the production corpus
(degree ~67); it was 1 of 300 on a smoke corpus with degree 8.

**`pairs_per_anchor` vs `--knn-k` in E3.** The recall column needs each anchor to
have comfortably more positives than `k`, or most anchors are dropped and the
survivors score near 1 regardless of arm. At the production corpus (degree ~67,
20% withheld) that is ~13 held-out edges per anchor, so the default `knn-k 10`
is at the edge; the script warns when the ratio is below 2x. Read `spearman` and
`pair_order` alongside it — neither depends on the set's width.

**The E4 y-axis must be relative.** A larger corpus makes everything better and
also raises TRE, so plotting an absolute score against TRE shows a collapse
driven by N alone. Plot each relational arm's gap to the pointwise arm at the
same N; that is why a pointwise run exists at every corpus size.

**E1's `pointwise_*` arms are a measurement floor, not a competitor.** Their
objective cannot read batch membership, so their spread across the three samplers
*is* the noise level the other arms are read against.

## The one arm that is not one-factor

`student_knn` (E2, opt-in via `WITH_STUDENT_KNN=1`) builds its graph from the
base student's embeddings, so the row bandwidths `tau_i` are the student's rather
than the teacher's. It differs from the teacher arms in temperature as well as in
support. Report it as indicative, and say so.

## Known scope limits

* **E4's corpus sizes cap at ~48.7k** — the deduplicated union of the three
  training CSVs. `make_subsets.py` refuses a larger N rather than truncating
  silently. N spans one decade; N/B spans from ~20 to ~3000, which is the axis
  the hypothesis names. Reaching 500k needs an external corpus and roughly 2
  GPU-hours per training run at that size.
* **E3 evaluates held-out *edges*, not held-out *texts*.** Both relation sets are
  computed from the cached teacher, so it runs post-hoc on any checkpoint with no
  teacher forward pass. A text-level held-out set (`data/val_set/`) would be a
  stronger generalization claim and needs a teacher pass to build.
* **E2's default family withholds 20% of teacher edges** (`HOLDOUT_FRAC=0.2`) so
  that E3 can score the same checkpoints. Its absolute numbers therefore do not
  line up with the paper's main table — it is a controlled study, and the
  comparison that matters is between its own arms. `HOLDOUT_FRAC=0` gives a
  family comparable to the main table, at the cost of E3's masked-neighbour
  metric.
