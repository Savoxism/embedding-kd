# Experiments

The set that supports `story.md`, and nothing else. Ordered by information per
GPU-hour rather than by section number: the runs that can **delete** later runs
come first.

**Every number recorded before now belongs to a different method.** The mutual
filter, row truncation, multi-hop diffusion and fixed-reference calibration have
all been removed, and three of the four sit in the artifact cache key. Old
results are quoted below only where their *relative* structure says which arms
are worth paying for. None of them may appear in the paper.

---

## Budget

Counted after reuse — several cells are the same run serving two purposes, at 3
seeds. **The scripts currently default to one seed (`SEEDS=42`)**, a third of
every count below; pass `SEEDS=42,43,44` for the numbers that go into the paper.
With one seed, read gaps against the ~0.08 seed sd the 2026-09-12 sweep measured.

| | runs | reused from |
|---|---|---|
| 0.1 `graph_k` sweep | 12 | — |
| 1.1 row term (2 arms) | 6 | full reference = 0.1 winner |
| 1.2 calibration term | 3 | " |
| 1.3 uniform target | 3 | " |
| A the count | **0** | analytic |
| B comparison ladder (4 arms × 2 pairs) | 24 | — |
| C batch composition | 15 | `full @ random` = 0.1 winner |
| D component ablation | 9 | −row, −calib from Stage 1 |
| E held-out relations | **0** | post-hoc on D |
| F main table | 6 | pair 1 = 0.1 winner |
| G sensitivity | 0–9 | `graph_k` from 0.1 |
| **total** | **90–99** | |

Most are the 22M student at ~5 min; the 4B→BERT pair in B and F is the expensive
part.

---

## Code patches (done 2026-09-13)

| experiment | switch | what it changes, and nothing else |
|---|---|---|
| 1.1 random row centers | `--row_centers random` | each L_row row keeps its width $|\Omega_j|$; its columns are drawn uniformly from the pool and its target is the teacher softmax at $\tau_j$ over them |
| 1.3 uniform target | `--relation_target uniform` | the transition row's columns and $\tau_i$, equal mass on every retrieved neighbour (L_rel and L_row) |
| B `student_knn` | `--neighbor_source student` | columns from the base student's kNN; teacher row temperatures; the neighbour source is part of the artifact's cache key |

**The 2026-09-12 sweep has three invalid tables.** `student_knn` trained on the
teacher graph: the old helper wrote `graph_student_knn_k200.pt`, the runner looked
for `graph_student_knn.pt`, found nothing, and built the teacher graph under that
name — its scores equal the teacher arm's to two decimals. The batch-size half
of C ran every batch size for 5 epochs, so B = 1024 took 65 optimizer steps
against 1055 at B = 64; it has been cut (see Cut). D had no full-objective arm
under its own holdout. The other two are fixed in the scripts; re-run them with
`ARMS=` (see `scripts/exp/README.md`). The study now runs on
`qwen3_0_6b_to_minilmv2_h384` only.

---

## Stage 0 — fix the operating point (12 runs)

Nothing else is interpretable until this is settled, because every other run
inherits it. The winning setting, at 3 seeds, is also **the full-model reference**
that Stage 1, D and F compare against.

### 0.1 `graph_k`

`graph_k = 200` was chosen when a row was mutual-filtered and cut down to the
neighbours holding 99 % of the teacher's similarity, leaving ~67 real columns per
anchor. An anchor now keeps all `graph_k` of them, so `graph_k` sets the encode
cost per step directly — and through $\tau_i = (s^{(1)}-s^{(k)})/\log k$ it still
sets how sharp a row is. The old sweep measured neither quantity.

`graph_k ∈ {25, 50, 100, 200}`, 3 seeds, one pair. Record Avg, wall time, peak
GPU, `pool_fill_avg`, `train_encoded_texts_cum`, and `target_kl_uniform_r1` (the
build warns under 0.05 — that is the flat-row failure).

**Rule.** Take the smallest `graph_k` within one seed-sd of the best Avg. Small
beats tied-and-large: it is the whole cost argument.

---

## Stage 1 — three deletion tests (12 runs)

Each asks whether a component earns its place. Each "no" removes a term, a
hyperparameter, a paragraph of method, and a later sweep. Run these before
anything expensive.

### 1.1 Does $\mathcal L_{\text{row}}$ earn a hyperparameter?

Two arms: `--row_weight 0`, and **random row centers** — supervise the extra
anchor $j$ against random pool columns instead of the ones the teacher retrieved
for it. The second is the one that matters. The paper claims the extra anchors
help *because* both ends of each comparison are teacher-chosen, and nothing
currently tests that.

Old measurement: `row_weight` 0 → 75.45, 1 → 75.60, 2 → 75.63, seed sd ~0.05.

**Rule.** If random centers also gain ~0.15, the term is a regulariser and not an
eligibility effect. Delete it: the method drops to **one hyperparameter**, the
eligibility paragraph goes, the `row_weight` sweep in G goes, and so does the
`row_exposed_mass = 0.44` caveat — the fraction of a neighbourhood a pool
actually exposes. That is a better outcome for the paper than 0.15 points.

### 1.2 Does the calibration term earn its place?

`--calibration_mode none`. It is the only part of the objective whose compared
texts are decided by the batch, which is the one internal contradiction in the
story.

**Rule.** Loses ≤ 0.2 → delete it, and the contradiction with it. Loses ≥ 0.3 →
keep it, and C.1 decides whether its batch-dependence is real or only structural.

### 1.3 Does $\tau_i$ earn the per-row temperature machinery?

Compare the teacher's similarity profile over a neighbourhood against treating
every neighbour in it as equally similar. Same texts compared, same columns, same
everything — only the target values move.

The sharpest experiment in the file. If uniform matches, then *how* similar the
teacher says each neighbour is carries nothing and only *which texts it
retrieved* does: $\tau_i$, the affine-invariance argument and the whole
temperature section disappear, and the method becomes "compare against the
teacher's top-$k$". If uniform loses, $\tau_i$ is justified by measurement rather
than by argument, which is strictly better than how it is justified today.

**Rule.** Both outcomes are publishable and one of them is a large
simplification. Run it early.

---

## Stage 2 — the evidence (60 runs; A and E need no GPU)

### A. The count — the premise, computed (0 runs)

The premise is a quantity, so compute it instead of arguing it. Under random
batching, the number of an anchor's $k$ nearest neighbours that land in its own
batch is

$$\mathbb E[\text{informative comparisons per anchor per step}] = (B-1)\frac{k}{N-1}.$$

Three readings, each answering a question a reviewer would otherwise ask.

**How bad is it here.** $N = 13{,}553$, $B = 64$, $k = 200$: **0.93** out of 63
comparisons. At $k = 20$: 0.09, one every eleven batches.

**How bad does it get.** The count falls as $1/N$ at fixed batch size — the same
setting at $N = 10^6$ gives 0.013, one informative comparison every 79 steps.
This is the scaling claim, proved rather than measured. The earlier attempt to
measure it by training at four corpus sizes varied $N$ and the number of updates
together and came out *against* the hypothesis; a formula has no confound.

**What batch size would fix it.** Solving for 63 informative comparisons at
$N = 13{,}553$, $k = 200$ gives $B \approx 4{,}270$. That is the answer to "why
not just use a bigger batch": you can buy the count with batch size, linearly, at
two orders of magnitude of encoder work. It is stated from the formula, not
trained (see Cut).

The figure: teacher similarity on the $x$-axis, probability of being compared on
the $y$-axis, one curve per method. Batch-local is flat at $(B-1)/(N-1)$
whatever the similarity; ours is a step at the top-$k$ boundary.
`scripts/exp/coverage.py` already computes the exposure side.

### B. The comparison ladder — the centrepiece (24 runs)

One axis moves: which texts an anchor is compared against. The number of
comparisons, the loss, the temperature, the optimiser and the seed are identical
across arms — a like-for-like swap, not a cap. The objective is deliberately
minimal (`--relation_target direct --no_ambient --row_weight 0`) so that no part
of our method can be credited with the gap.

| arm | compared against | kills the explanation |
|---|---|---|
| `in_batch` | the rest of the batch | — (the baseline) |
| `corpus_uniform` | the same number, drawn from the corpus | "you just need to leave the batch" |
| `student_knn` | the **student's** own top-$k$ | "any semantic neighbourhood would do" |
| `teacher` | the texts the teacher retrieved | — (ours) |

Old numbers, minimal objective, one pair: teacher 74.29, corpus_uniform 71.61,
in_batch 71.25, sd ~0.1. The structure is the finding: the arm that compares
against unrelated texts sits *on top of* the batch-local baseline, not between it
and ours.

`student_knn` is the most interesting arm in the paper — the ANCE comparison made
empirical, and the one a reviewer thinks of unprompted. `--neighbor_source
student` builds the neighbour sets from the base student's embeddings, ranked by
the student so the quota takes the student's own top-63, keeps the teacher's row
temperatures, and reads the targets from the teacher (`--relation_target
direct`), so that *which texts are compared* is the only difference.

**`rewired` has been dropped: it is now the same arm as `corpus_uniform`.** It
drew each anchor's *own degree* of random texts, which differed from a fixed
quota only while degrees were ragged — mean 67, min 1, max 181 under the mutual
filter and the 99 % cut. A directed graph with no truncation gives every anchor
degree exactly `graph_k`, so the two arms draw the same number from the same
distribution; verified identical for 400/400 anchors. The "row shape" control it
used to provide has moved to 1.3, which varies target values over a fixed set of
texts — the only sense in which shape is still free.

4 arms × 3 seeds on `qwen3_0_6b_to_minilmv2_h384`. The 2026-09-12 sweep also ran
`qwen3_4b_to_bert_base` (teacher 77.32, corpus_uniform 75.53, in_batch 75.08);
that pair is out of scope from here on.

Cost columns (`cost_*`, `train_encoded_texts_cum`) fall out of the same runs and
carry C3 for free. Report `corpus_uniform`'s encode count beside ours: an
off-graph draw deduplicates far worse, and that gap is part of the argument.

### C. Dose–response on the count (15 runs)

A and B say a number matters. C moves that number for a batch-local objective
and checks the score follows, which turns a previously awkward result into a
supporting one.

**Raise the count by composing batches (15 runs).** `random` vs
`teacher_neighbor` batching, arms `pointwise` / `in_batch` / **full GGPKD**,
3 seeds; `full @ random` is the Stage 0 winner. Filling a batch from one teacher
neighbourhood raises the count for a batch-local objective and changes nothing
else, so the prediction is that it improves *for that reason*: the old run has
`in_batch` gaining +2.25 under teacher-neighbour batching while ours moved 0.07.

Run it on the **shipped objective**, not the minimal one. The claim that matters
is about the method we publish, and this is where the calibration term's
batch-dependence would surface if it is real.

Read it as difference-in-differences against the pointwise floor: `pointwise`
moved −2.54 across the same samplers, so batch composition shifts gradient
statistics for every objective and a raw spread proves nothing. The signature is
*sign and target* — batch-local improves exactly when the batch happens to supply
related texts; ours should not move.

**Report the measured count**, not only the score.

### D. Component ablation on the shipped objective (12 runs)

The full objective, minus both terms at once, plus `--knn_mode mutual` and
`--truncation_tolerance 0.01`, all under the same 20 % holdout; the single-term
arms come from Stage 1. The `full` arm is the reference: Stage 0's full run
withheld nothing, so reading these arms against it confounds every gap with the
holdout. Those last two defaults were changed on a structural
argument and have never been measured on the full objective — the ladder put
mutual 0.36 below directed on the minimal one, which is suggestive and not
enough.

### E. Held-out relations (0 runs, post-hoc on D's checkpoints)

Withhold 20 % of the teacher's pairs from every arm's comparison sets, then ask
whether the student reproduces them anyway. This is the coverage half of the
story and the only evidence that the method generalises past the comparisons it
was shown.

**The probe was broken; it is fixed.** `pair_order` drew its columns from
`range(n_anchors)` on an `[anchors, corpus]` matrix, so a sparse held-out mask
left zero or one triplet and read exactly 1.0 or 0.0. Columns are now drawn from
each anchor's own admissible set. Spearman puts the teacher arms (0.70–0.73)
*below* the arms that never saw related pairs (0.79–0.85) on the pooled number;
`spearman_anchor` now reports the mean per-anchor correlation beside it — the
hypothesis is that teacher arms sharpen local structure and compress the global
cosine scale, and a single pooled Spearman rewards the flat arms for exactly
that. The probe scores against `graph_main_holdout.pt` by name.

If the split does not rescue it, report the negative and drop "global geometry"
from the title. A negative we report costs far less than one a reviewer finds.

---

## Stage 3 — the deliverable (6–15 runs)

### F. Main table (6 runs)

Three teacher→student pairs, 3 seeds, at the Stage 0 operating point with
whatever Stage 1 left standing; pair 1 is the Stage 0 winner.

### G. Sensitivity (0–9 runs)

`graph_k` is already swept in Stage 0 — reuse it. `row_weight` only if 1.1 kept
the term. This exists to show the defaults are not sitting on a peak, which is
C4. It is not a search.

---

## Cut

- **Training at four corpus sizes.** Replaced by the formula in A, which gives
  the $1/N$ statement with no confound. The old runs varied $N$ and the number of
  updates together and came out against the hypothesis.
- **Training at several batch sizes (was C.2).** No single setting holds the
  other factors fixed across $B$: matching epochs gives B = 1024 65 optimizer
  steps against 1055 at B = 64 (the 2026-09-12 sweep, where both arms fell with
  $B$ for that reason), matching steps gives it 80 passes over the corpus against
  5, and the learning rate is unscaled either way. The result could not be read
  as a statement about the count. "Why not a bigger batch" is answered by A's
  formula and the cost columns; very large batches go under Limitations. The old
  observation that the baseline passed the teacher arm at B = 256 (N = 48k) was
  measured on a different method and is not carried forward.
- **`rewired`.** Identical to `corpus_uniform` under the current graph.
- **Multi-hop radius ablation.** Removed from the codebase; reporting it would
  mean restoring the pipeline first.
- **`teacher_diverse` batch sampler.** Within noise of `random`.
- **More points on either sensitivity sweep.** Both surfaces are flat; extra
  points buy nothing and invite "tuned on test".

---

## Order

```
patches  random-centers · uniform-target · one-factor student_knn
Stage 0  0.1 graph_k                     12   ── fixes the operating point,
                                              ── and is the full-model reference
Stage 1  1.1 row term                     6   ── each "no" deletes later work
         1.2 calibration term             3
         1.3 uniform target               3
Stage 2  A   the count                    0   ── formula + one figure
         B   ladder (4 arms, 2 pairs)     24
         C   batch composition            15
         D   component ablation            9
         E   held-out relations           0   ── post-hoc on D
Stage 3  F   main table                    6
         G   sensitivity                0–9
```

Stage 0 is strictly first. Stage 1 before Stage 2, because a deletion there
removes arms from D and G. Everything inside Stage 2 runs in parallel across
GPUs.
