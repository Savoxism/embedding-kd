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

Counted after reuse — several cells are the same run serving two purposes.

| | runs | reused from |
|---|---|---|
| 0.1 `graph_k` sweep | 12 | — |
| 1.1 row term (2 arms) | 6 | full reference = 0.1 winner |
| 1.2 calibration term | 3 | " |
| 1.3 uniform target | 3 | " |
| A the count | **0** | analytic |
| B comparison ladder (4 arms × 2 pairs) | 24 | — |
| C.1 batch composition | 15 | `full @ random` = 0.1 winner |
| C.2 batch size | 12 | $B=64$ from B |
| D component ablation | 9 | −row, −calib from Stage 1 |
| E held-out relations | **0** | post-hoc on D |
| F main table | 6 | pair 1 = 0.1 winner |
| G sensitivity | 0–9 | `graph_k` from 0.1 |
| **total** | **90–99** | |

Most are the 22M student at ~5 min; the 4B→BERT pair in B and F is the expensive
part.

---

## Code that must be patched first

Three experiments cannot run as the code stands. All three are small and all
three are load-bearing, so they are the first thing to write.

| experiment | what is missing |
|---|---|
| 1.1 random row centers | the row loss always uses $\mathcal N_j \cap \mathcal P_B$; there is no switch to draw the extra anchors' columns at random |
| 1.3 uniform target | `relation_target` has no value that treats every retrieved neighbour as equally similar |
| B `student_knn` | the existing arm builds *both* the neighbour set and the temperatures from the student, so it is not one-factor |

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
two orders of magnitude of encoder work. C.2 checks the prediction.

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
empirical, and the one a reviewer thinks of unprompted. It needs the patch above:
build the neighbour sets from the base student's embeddings but read the targets
from the teacher (`--relation_target direct`), so that *which texts are compared*
is the only difference.

**`rewired` has been dropped: it is now the same arm as `corpus_uniform`.** It
drew each anchor's *own degree* of random texts, which differed from a fixed
quota only while degrees were ragged — mean 67, min 1, max 181 under the mutual
filter and the 99 % cut. A directed graph with no truncation gives every anchor
degree exactly `graph_k`, so the two arms draw the same number from the same
distribution; verified identical for 400/400 anchors. The "row shape" control it
used to provide has moved to 1.3, which varies target values over a fixed set of
texts — the only sense in which shape is still free.

4 arms × 3 seeds × 2 teacher→student pairs. The second pair is what turns C1
from "holds in our setting" into a claim.

Cost columns (`cost_*`, `train_encoded_texts_cum`) fall out of the same runs and
carry C3 for free. Report `corpus_uniform`'s encode count beside ours: an
off-graph draw deduplicates far worse, and that gap is part of the argument.

### C. Dose–response on the count (27 runs)

A and B say a number matters. C moves that number two different ways and checks
the score follows. This is the strongest form the claim can take, and it turns
two previously awkward results into supporting ones.

**C.1 — raise the count by composing batches (15 runs).** `random` vs
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

**C.2 — raise the count by enlarging the batch (12 runs).** `in_batch` and full
GGPKD at $B \in \{256, 1024\}$, 3 seeds ($B = 64$ comes from B). A predicts the
baseline's count rises linearly in $B$ and needs $B \approx 4{,}270$ to reach
ours, so it should climb and not arrive.

**This is the arm that can hurt us, which is exactly why it is here.** In the old
scaling runs the baseline did climb with batch size — at $N = 48$k: 71.85 (B=16)
→ 72.82 (B=64) → **73.20** (B=256) — and at $B = 256$ it *passed* the teacher
arm's 72.83. Counts of 0.06 → 0.26 → 1.06 explain the climb and do not explain
the overtake. Either the anomaly does not survive the new operating point, or the
claim is stated per unit of encoder work rather than per step — which the cost
columns support, since a 1024-text batch encodes an order of magnitude more per
update. Both outcomes are reportable; being surprised by this in review is not.

**Both halves report the measured count**, not only the score, so the result is a
curve: score against informative comparisons per anchor, with both ways of buying
the count landing on it and ours at the right-hand end for a fraction of the
encoder cost.

### D. Component ablation on the shipped objective (9 runs)

Minus both terms at once, plus `--knn_mode mutual` and
`--truncation_tolerance 0.01`; the single-term arms come from Stage 1 and the
full model from Stage 0. Those last two defaults were changed on a structural
argument and have never been measured on the full objective — the ladder put
mutual 0.36 below directed on the minimal one, which is suggestive and not
enough.

### E. Held-out relations (0 runs, post-hoc on D's checkpoints)

Withhold 20 % of the teacher's pairs from every arm's comparison sets, then ask
whether the student reproduces them anyway. This is the coverage half of the
story and the only evidence that the method generalises past the comparisons it
was shown.

**It currently fails**, and the probe is broken: `pair_order` returns exactly
1.0000 for every arm, and Spearman puts the teacher arms (0.70–0.73) *below* the
arms that never saw related pairs (0.79–0.85). Fix the metric, then split
Spearman into within-neighbourhood and across-neighbourhood — the hypothesis is
that teacher arms sharpen local structure and compress the global cosine scale,
and a single mixed Spearman rewards the flat arms for exactly that.

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
  updates together and came out against the hypothesis; their one genuinely
  informative signal — the baseline climbing with batch size — is now C.2.
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
         C.1 batch composition            15
         C.2 batch size                   12   ── the arm that can hurt us
         D   component ablation            9
         E   held-out relations           0   ── post-hoc on D
Stage 3  F   main table                    6
         G   sensitivity                0–9
```

Stage 0 is strictly first. Stage 1 before Stage 2, because a deletion there
removes arms from D and G. Everything inside Stage 2 runs in parallel across
GPUs.
