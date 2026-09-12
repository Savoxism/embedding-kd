# Experiments needed to support `story.md`

Every claim in `story.md` §3 has to land on a row here. Anything else is
optional and should be cut before it costs GPU time.

**Read this first.** Every number below was measured before two method changes
landed: the mutual filter was removed (`knn_mode=directed`) and row truncation
was removed (`truncation_tolerance=0.0`). Both are in the artifact cache key, so
**every cached graph is invalid and every recorded score belongs to a different
method.** The numbers are kept here because their *relative* structure is what
the story rests on, and because they say which arms are worth re-running. None
of them may be quoted in the paper as they stand.

---

## Status of what already exists

`runs/exp{1,2,3,4}/*.csv`, pair `qwen3_0_6b_to_minilmv2_h384`, 3 seeds, Avg-All.

**E1 — is batching deciding the learning signal?** Yes for batch-local, no for
teacher support.

| arm | random | teacher_neighbor | teacher_diverse | spread |
|---|---|---|---|---|
| in_batch | 71.23 | 73.48 | 71.21 | **2.27** |
| teacher_support | 73.95 | 73.91 | 73.88 | **0.07** |
| pointwise (floor) | 68.72 | 66.18 | 68.58 | 2.54 |

Caveat that must be handled in the write-up: the pointwise floor is *not* flat,
so E1 has to be read as a difference-in-differences against it, not as a raw
spread. Batch composition changes gradient statistics for every objective; what
is specific to `in_batch` is the *direction* (+2.25 when batches are teacher
neighbourhoods, i.e. exactly when the batch accidentally approximates the
teacher's support).

**E2 — teacher relevance, or just leaving the batch?** Relevance.

| arm | Avg | reading |
|---|---|---|
| teacher_topk | 74.29 ± 0.05 | teacher support, no edge filter |
| teacher_mutual | 73.93 ± 0.02 | teacher support, mutual filter |
| corpus_uniform | 71.61 ± 0.12 | off-graph, batch-independent |
| in_batch | 71.25 ± 0.17 | baseline |
| rewired | 71.02 ± 0.11 | teacher degree, wrong endpoints |

This is the paper's central table. `corpus_uniform ≈ in_batch` kills "leaving
the batch is enough"; `rewired` below both kills "the row shape is enough".

**E3 — does it generalise past the supervised edges?** Currently says **no**,
and the metric is broken.

| arm | spearman (held-out edges) | pair_order |
|---|---|---|
| corpus_uniform | 0.796 | 1.0000 |
| in_batch | 0.789 | 1.0000 |
| teacher_mutual | 0.726 | 1.0000 |
| teacher_topk | 0.700 | 1.0000 |

`pair_order = 1.0000` for every arm is degenerate — it measures nothing. And the
Spearman column contradicts the abstract's geometry claim.

**E4 — does the advantage scale with N/B?** Currently says the opposite.
Gap (teacher_support − in_batch) at bs=64: +0.94 (N=5k), +0.85 (13.5k), +0.47
(25k), +0.58 (48k). At bs=256 it vanishes and inverts (−0.37 at N=48k). The
hypothesis predicted the gap should *widen* with N. Confounded by steps: larger
N at fixed epochs means more updates.

**Sensitivity** (`docs/latex/tables/`): `graph_k` 50→400 moves Avg by 0.34 and
400 costs +36.8 % time / +54.2 % GPU over 200; `row_weight` 0→2 moves Avg by
0.18 (0 → 75.45, 1 → 75.60, 2 → 75.63).

---

## P0 — blocking. The paper cannot be submitted without these.

### P0-1. Re-sweep `graph_k`, then rebuild every graph

**Why.** `graph_k = 200` was chosen when a row was mutual-filtered and truncated
at 99 % mass, leaving ~67 real columns per anchor. An anchor now keeps all 200,
so the shared pool and the encode cost that dominates a step grow with it. The
old sweep measured a different quantity.

**Arms.** `graph_k ∈ {25, 50, 100, 200}`, 3 seeds, one pair. Record Avg, wall
time, peak GPU, and `pool_fill_avg` / `train_encoded_texts_cum`.

**Decision rule.** Pick the smallest `graph_k` within one seed-sd of the best
Avg. Everything downstream uses it. Do not run anything else in this file until
this is settled — every other run would be at a stale operating point.

### P0-2. E1 on the **full** objective, plus `--calibration_mode none`

**Why.** Two questions in one sweep. (i) Does batch composition reach the full
objective, or only the minimal one? That is the direct test of the paper's
central claim on the actual method. (ii) What is $\mathcal L_{r=0}$ worth?

**Arms.** 12 runs, 3 seeds:

```
full_random | full_teacher_neighbor | full_teacher_diverse   (--batch_sampler ...)
full_no_ambient                                             (--calibration_mode none)
```

**Decision rule.**
- spread ≤ 0.1 → the ambient term's batch-determined support is empirically
  inert; keep the method, report the number, handle it in prose.
- spread > 0.1 **and** `no_ambient` loses ≥ 0.3 → change the ambient target so
  its estimand stops depending on the pool: per-pair cosine matching, or
  Horvitz–Thompson weights $1/\pi_j$ with $\pi_j$ from in-degree. (A fixed
  corpus reference set would also do it and has been ruled out; the code for it
  is gone.)
- `no_ambient` loses ≤ 0.2 → delete $\mathcal L_{r=0}$ and the problem with it.

### P0-3. Component ablation on the full objective

**Why.** C2 ("the gain is not from the loss") is currently an assertion. This is
the table that makes it checkable, and reviewers ask for it unconditionally.

**Arms.** Full; `--row_weight 0`; `--calibration_mode none`; both; plus
`--knn_mode mutual` and `--truncation_tolerance 0.01`, the two arms whose
defaults were changed on a structural argument rather than a measurement.
3 seeds each.

**Why the last two matter.** E2 measured `teacher_topk` 0.36 above
`teacher_mutual` on the minimal objective, and the method was changed on that
basis. If mutual wins on the full objective the default is wrong. Same for
truncation: it was removed on a structural argument, not a measured one.

### P0-4. Re-run the main table

**Why.** Three settings × 9 datasets, currently reporting the old method.
Nothing can be submitted with those numbers.

**Arms.** All three teacher→student pairs, 3 seeds, at the `graph_k` from P0-1.

### P0-5. Repair E3, then re-score

**Why.** It is the only evidence for the geometry claim and it currently
contradicts it, with one metric visibly degenerate.

**Work.** (i) Debug `pair_order` in `scripts/exp/heldout_geometry.py` — a metric
that returns exactly 1.0000 for every arm is not measuring the arms. (ii) Split
the Spearman into within-neighbourhood and across-neighbourhood; the hypothesis
is that teacher arms sharpen local structure and compress the global cosine
scale, which a single mixed Spearman rewards the flat arms for. (iii) Re-score
the P0-3 checkpoints. Post-hoc, no training.

**If the split does not rescue it:** report it as a negative result and weaken
the title from "global geometry" to "teacher-selected relational supervision".
A reported negative costs far less than one a reviewer finds.

---

## P1 — needed for the claims to generalise.

### P1-1. E2 on a second teacher→student pair

**Why.** The central table rests on one pair. If the +2.7 gap reproduces on
Qwen3-4B→BERT-base, C1 is general; if not, C1 must be scoped, and it is better
to scope it ourselves.

**Arms.** The five E2 arms, 3 seeds, `PAIR=qwen3_4b_to_bert_base`.

### P1-2. Cost table for the controlled study

**Why.** C3. The numbers are already exported per arm (`cost_*`,
`train_encoded_texts_cum`); this is a table, not a sweep. Include
`corpus_uniform`'s encode count — an off-graph draw deduplicates far worse than
a teacher draw, and that difference is part of the argument.

---

## P2 — makes the story tighter; cut if time runs out.

### P2-1. Is $\mathcal L_{\text{row}}$ actually about eligibility?

**Why.** The paper says auxiliary rows work because both endpoints are
teacher-approved. Nothing tests that, and the term is worth +0.15.

**Arms.** (a) random row centers — supervise $j$ against random pool columns
instead of $\mathcal N_j \cap \mathcal P_B$; (b) `--row_weight 0` at matched
compute (larger batch). 3 seeds.

**Decision rule.** If (a) also gains ~0.15, the term is a regulariser, not an
eligibility effect: delete it, drop the eligibility paragraph, and the method
falls to **one** hyperparameter. That is a better outcome for the paper than
keeping 0.15 points.

### P2-2. Coverage curve from `row_exposed_mass`

**Why.** Turns the 0.44 exposed-mass caveat into a measurement, in the same
alignment/coverage language the paper uses.

**Work.** Sweep batch size {16, 64, 256}, plot (`row_exposed_mass`, ΔAvg). The
metric is already logged per step; this rides along on runs that exist.

### P2-3. E4, step-matched — or cut it

**Why.** As it stands E4 argues against the hypothesis and confounds N with the
number of updates.

**Work.** Re-run with matched optimisation steps rather than matched epochs, or
drop E4 from the paper. Do not publish it in its current form.

---

## Not worth running

- More `graph_k` / `row_weight` points beyond P0-1. Both surfaces are flat; more
  points buy nothing and invite "tuned on test".
- Multi-hop diffusion scales. Removed from the codebase, not merely defaulted
  off: the method supervises one hop, so a pool *is* a transition row. Reporting
  a radius ablation would mean restoring the pipeline first.
- `student_knn` (E2, opt-in). It differs from the teacher arms in temperature as
  well as support, so it is not one-factor. Report as indicative or not at all.

---

## Order of execution

```
P0-1 (graph_k)  ─► everything else
                   ├─ P0-2 (12 runs)  ──► decides the L_r0 question
                   ├─ P0-3 (18 runs)  ──► component table  ──► P0-5 re-scores these
                   ├─ P0-4 (main table, 9 runs)
                   └─ P1-1 (15 runs)
```

P0-1 first, always. Everything after it is parallel across GPUs.
