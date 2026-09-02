# GGPKD ablations

Runners and analysis for `analysis_ablation_plan.md`. One script per ablation,
one GPU per run.

```
_common.sh          locked protocol + run_arm(); sourced, never executed
full.sh             the unablated model, all seeds -- also the warm-up
s1_support.sh       P0  support selection + the two no-graph baselines
s2_scales.sh        P0  R={1,2,4} vs R={1}
s3_target.sh        P0  diffusion target vs teacher cosine on the same nodes
s4_factorial.sh     P0  ambient x row, full 2x2
g1_knn.sh           P1  mutual / directed / symmetrized kNN
n1_negatives.sh     P1  hard:uniform negative mix at fixed total quota
x1_transfer.sh      P1  BGE-M3 -> MiniLMv2-H768 replication of the support claim
run_all.sh          dispatch everything across GPUs, then run the analysis

replay_coverage.py  offline sampler replay -> coverage + epsilon (no GPU)
candidate_width.py  the method's candidate width, so a baseline can match it
collect.py          runs -> tidy CSV, console tables, booktabs LaTeX
figures.py          -> latex/figures/fig1_support_coverage.pdf, fig2_causal_chain.pdf
paperstyle.py       ICLR figure style: 5.5 in text width, Times, embedded fonts
```

## Running

```bash
# one ablation, one GPU
GPU=0 bash scripts/ablation/s1_support.sh

# several at once, one GPU each (warms up the shared caches first)
GPUS="0 1 2 3" bash scripts/ablation/run_all.sh

# a subset
PLAN="s1 s4" GPUS="0 1" bash scripts/ablation/run_all.sh
```

Every arm writes to `runs/ablation/<pair>/<ablation>/<arm>/seed<seed>/` and drops
a `.done` marker when it finishes, so re-running a script resumes rather than
repeats. `FORCE=1` re-runs anyway.

**Run `full.sh` (or `run_all.sh`) before launching scripts in parallel.** The
first run builds the teacher-embedding cache and the base graph artifact; two
processes building them at once will corrupt both. Once they exist, the scripts
are independent.

## Budget

At 3 seeds, P0 is 30 new training runs plus the 3 shared `full` runs:

| | arms x seeds | new runs |
|---|---|---|
| full (shared by S1-S4) | 1 x 3 | 3 |
| S1 | 5 x 3 | 15 |
| S2 | 1 x 3 | 3 |
| S3 | 1 x 3 | 3 |
| S4 | 3 x 3 | 9 |
| G1, N1 (seed 42 screen) | 2 + 2 | 4 |
| X1 (second pair) | 2 x 3 | 6 |

The `full` arm is trained once and reused by S1-S4, which is only valid because
every arm goes through `run_arm` with the same pinned protocol. If you change
anything in `_common.sh` -- learning rate, epochs, `DIRECT_TEMP` -- delete the
`full` runs and retrain them, or the reuse silently compares against a different
model.

## Analysis

```bash
# Figure 1 + the coverage/epsilon columns. No GPU, no training run.
python scripts/ablation/replay_coverage.py \
    --artifact cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/graph_base.pt \
    --out runs/ablation/analysis/coverage.csv

# tables: console, tidy CSV, booktabs
python scripts/ablation/collect.py --csv runs/ablation/analysis/results.csv \
    --latex latex/tables --hubness

# figures
python scripts/ablation/figures.py --out-dir latex/figures
```

`--hubness` prints the indegree tail per graph build (max, p99, Gini, the edge
share held by the top 1% of nodes). That block, not the downstream average, is
G1's primary evidence: the mutual-kNN claim is about the graph.

Figure 2 uses the single model pair represented by S1 and only that pair's
`full` runs; X1's full runs must not enter its hybrid mean or seed error bars.
Use coverage replay from the same pair. Missing or multiple S1 pair identities
are rejected instead of silently averaging across settings.

Figures are drawn at their final printed size, so include them **without** a
width key:

```latex
\begin{figure}[t]\centering
  \includegraphics{figures/fig2_causal_chain.pdf}
  \caption{...}
\end{figure}
```

`width=\textwidth` would rescale them and their 8 pt labels along with them.

## What each arm needs, and what it does not

The flags exist on `main.py`; none of them is a code branch you have to maintain:

| Arm | Flag | Touches |
|---|---|---|
| S1 | `--support_policy {hybrid,topk,proportional,uniform}` | sampler only |
| S1 | `--batch_local` | data path: candidates are the batch |
| S1 | `--relation_target ambient_only` + `--diffusion_quota 0` | objective + quotas |
| S2 | `--diffusion_scales 1` | graph artifact (own `GRAPH_KEY`) |
| S3 | `--relation_target direct` | criterion target, same columns |
| S4 | `--no_ambient`, `--row_weight 0` | criterion terms |
| G1 | `--knn_mode {mutual,directed,symmetrized}` | graph artifact (own `GRAPH_KEY`) |
| N1 | `--hard_neg_k`, `--random_neg_k` | sampler quotas |

`uniform` draws from the anchor's own diffusion pool, not from the whole corpus.
Drawing outside the pool gives every selected column diffusion target exactly
zero, which deletes the objective instead of ablating the policy -- that arm
would measure nothing.

### The two S1 baselines

`batch_local` is batch-local relational KD: no graph, no candidate draw, no
auxiliary rows. Every anchor is scored against the texts that happen to share its
minibatch, and the objective collapses to a single KL against the teacher's
similarity profile over those columns. It is the form batch-relational
distillation normally takes, and it is the plan's "random co-occurrence" control.

It is **not encoder-budget-matched**, deliberately: it encodes ~2B texts per step
against the method's B + unique candidates, roughly 30x cheaper. That gap is the
arm's content. `collect.py` prints `enc.texts` beside every score so it cannot be
read as a like-for-like comparison.

`uniform_corpus` is the budget-matched counterpart, and the arm to quote at a
reviewer who says GGPKD only wins by encoding more. Same candidate width (read
back from the graph by `candidate_width.py`, since the diffusion quota is
derived rather than configured), same ambient-only objective -- but the columns
come from a uniform corpus draw instead of the teacher's neighbourhood. Its
diffusion mass is zero by construction, so the graph group is dropped with it.
Between the two, "more compute" and "structured support" are separated.

The graph group is *removed* under `ambient_only` rather than handed a zero
target. A zero-target scale still holds its weight in the loss normalization, so
leaving it in would scale the baseline's whole loss by the ambient share alone
(0.36 at `R={1,2,4}`) -- a different effective learning rate, not a different
objective.

## Metrics the plan asks for, and where they come from

| Quantity | Source | Needs training? |
|---|---|---|
| coverage `1 - delta_T`, cumulative | `replay_coverage.py` | no |
| selected-support distortion `epsilon` | `replay_coverage.py` | no |
| `E_hat`, teacher-weighted distortion | geometry probe, per epoch | yes |
| teacher-student Spearman | geometry probe, per epoch | yes |
| unique texts / tokens encoded | `encoded_*_cum` in `epochs.jsonl` | yes |
| wall-clock | `arm.json` | yes |
| peak VRAM | `peak_memory_mb` in `epochs.jsonl` | yes |
| hubness (indegree tail) | `graph_stats` in `run.json` | no (build only) |

`epsilon` is `-log` of the mass the draw retains: renormalizing a probability row
onto a subset holding mass `m` costs exactly `KL(ptilde || p) = -log m` nats, so
it is the target perturbation in the units of the loss, not a proxy for it.

## Reading the results

The plan's decision criteria, restated as what to check in the collected table:

- The causal-chain claim survives only if higher coverage comes with lower
  `E_hat` across policies and seeds, **and** `E_hat` orders the STS average the
  same way. If downstream improves while `E_hat` does not, report the empirical
  gain and drop the mechanism claim.
- A component stays a contribution only if deleting it costs more than seed
  noise. If `no_ambient` or `no_row` is flat in S4, it is an efficiency or design
  detail, not a contribution.
- Both S1 baselines have exposed graph-relation mass of exactly zero, by
  construction rather than by measurement. That is why they appear at coverage 0
  in Figure 2 and are absent from Figure 1: neither has a curve to draw. The
  guide line in Figure 2 joins only the four fixed-budget policies -- the
  baselines differ in more than the support policy, so connecting them would
  assert a comparability they do not have.
- `STS Avg` and `Pair-cls Avg` are always reported next to `Avg`. The gain over
  TALAS is concentrated in STS while pair-cls is down; a single `Avg` column
  hides that trade-off, and `collect.py` will not print one on its own.
