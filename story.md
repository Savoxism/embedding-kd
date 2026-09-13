# GGPKD — the story

What the paper claims, what it deliberately does not claim, and how each claim
is allowed to be phrased. Written after the survey in which the objective turned
out to be prior art; the conclusion of that survey is that the objective
*should* stay prior art, and the paper should sell the finding instead.

A note on wording, because the whole framing turns on it. Avoid "support of the
target distribution", "teacher mass", "relational budget" and "teacher
relevance". They are either jargon or they smuggle in a claim of authority we do
not need and cannot defend. Say instead: *which texts are actually similar to
this one*, *which comparisons a student is shown*, *how many comparisons per
anchor*. The argument is stronger in plain words, and it stops a reader from
hearing "the teacher decides what matters" — which invites the question *matters
for what?*, a question we would then have to answer.

---

## 1. Thesis

> What a teacher knows about a text is which few texts in the corpus are similar
> to it, and in what order. Mini-batch training almost never shows a student
> those pairs: it compares each text against whatever else happens to share its
> batch, which in a corpus of any size is almost always a set of unrelated
> texts. The student spends training being told, over and over, that unrelated
> things are unrelated.

Everything else in the paper is in service of that paragraph.

---

## 2. The gap, and the number that carries it

Take the corpus this paper trains on: 13,553 texts, batch size 64. Count a
text's genuine neighbours generously — say its 200 nearest under the teacher.
The expected number of them landing in the same batch is

$$63 \times \frac{200}{13{,}552} \approx 0.93 .$$

So of the **63 comparisons** an anchor receives per step, fewer than **one** is
with a text it is actually related to. Count neighbours at 20 instead of 200 and
it falls to 0.09: one informative comparison every eleven batches. Everything
else is *unrelated vs unrelated* — something the student already gets right and
would keep getting right without being told.

This is not a subtle inefficiency. It is where almost all of the supervision
goes.

Every relational KD method has to choose which comparisons to make, and the
literature chooses them by **what is cheap to have on hand**:

| method | what an anchor is compared against | chosen by | labels |
|---|---|---|---|
| SP, RKD, PKT, IRG | the other examples in the batch | batch composition | no |
| CompRess, SEED, ISD | a memory bank / momentum queue | availability | no |
| XBM | cross-batch feature memory | availability | no |
| ANCE, RocketQA | top-$k$ of an ANN index | the student, negatives only | yes |
| LSP (GCN KD) | graph neighbours | the task's own graph | yes |
| **GGPKD** | the texts the teacher retrieves for it | **the teacher's own retrieval** | **no** |

Nobody has made the set of compared texts a per-anchor quantity read off the
teacher, over a corpus with no graph and no labels — and, more to the point,
nobody has measured what that choice is worth.

The reason is not oversight, it is cost: a per-anchor comparison set means
encoding a different group of texts for every anchor. Queues and memory banks
exist precisely to avoid that. Section 5 is the answer to that cost.

---

## 3. What we claim

**C1 (the finding).** Replacing an anchor's comparisons with the texts the
teacher retrieves for it — **the same number of comparisons, chosen
differently** — improves the student. What pays is that the comparisons are with
genuinely related texts: neither leaving the batch, nor using another model's
notion of similarity, reproduces the gain. And the effect tracks *how many*
related texts an objective is shown: composing batches from one neighbourhood
improves the baseline. Enlarging the batch buys the same count only linearly, at
far more encoder work — a consequence of the count formula, not a trained result.

*The like-for-like count is experimental control, not a constraint we impose on
the method.* It is there so the difference cannot be attributed to one arm
simply making more comparisons, exactly as we fix the seed and the learning
rate. Practical cost is a separate question with its own table.

**C2 (the mechanism is not the point).** The objective is a standard softmax-KL
over a set of compared texts — the CompRess/SEED form, restricted to the
teacher's neighbours as in LSP. We change which texts are compared, and nothing
else. The ablations are designed so this is checkable rather than asserted.

**C3 (it is affordable).** Deduplicating the comparison sets of a batch into one
shared pool, and reusing already-encoded pool members as extra anchors, makes a
per-anchor comparison set cost roughly what a batch-local objective costs.

**C4 (nothing is tuned).** Two hyperparameters. The per-row temperature, the
calibration temperature, the balance between the two terms, the comparison set
and the set of supervised rows are all read off the graph. The per-row
temperature $\tau_i$ comes in closed form from the retrieval span, which makes
each row invariant to the scale of a given teacher's cosines.

---

## 4. What we explicitly do NOT claim

Stating these is what keeps C1–C4 credible.

- **Not a new loss.** Softmax over a comparison set, matched by KL, is CompRess
  (NeurIPS 2020) and SEED (ICLR 2021); the same operation restricted to graph
  neighbours is LSP (CVPR 2020). We use them as-is and say so.
- **Not the first to argue that in-batch sampling is a bad sampler.** ANCE
  (ICLR 2021) makes that argument for dense-retrieval negatives and fixes it
  with a global ANN index. Ours is the unlabelled, frozen-teacher case, where
  the chosen texts carry the *targets* as well, not just the contrast.
- **Not a new temperature rule.** Per-point bandwidths from the $k$-th neighbour
  are self-tuning spectral clustering (Zelnik-Manor & Perona, 2004); the
  $\log k$ normalisation is UMAP's. Ours is a closed form instead of a solve —
  an engineering simplification, not a concept.
- **Not a claim that geometric distortion controls downstream score.** The
  alignment/coverage split motivates the design; it does not predict benchmark
  movement.
- **Not global geometry preservation** until the held-out probe is repaired
  (`experiments.md`, Stage 2E). Until then the honest phrasing is that the
  student is supervised on the comparisons that carry information, and the title
  should say that.
- **Not a claim about where the student is wrong.** We choose comparisons by
  what the teacher finds similar, not by where the student currently errs. Those
  are different criteria, and the second is ANCE's. See §7.

---

## 5. The method, as the cheapest realisation

Offline, once per (teacher, corpus): encode, $\ell_2$-normalise, retrieve the
top $k$ for every text. No reciprocity filter — every text the teacher retrieved
for $i$ is one $i$ gets compared against, and every node has degree exactly $k$.
Read the temperature off the retrieval span, $\tau_i =
(s_i^{(1)}-s_i^{(k)})/\log k$, so the $k$-th neighbour is $k$ times less likely
than the nearest, for every text, with no constant to choose. The teacher row is
the softmax over those neighbours at $\tau_i$, kept whole — no truncation.

During training the comparison set of an anchor **is** its retrieved
neighbourhood: no sampling, no negatives, no RNG, identical in every epoch. A
batch encodes the deduplicated union of those sets once, and that single
encoding pass serves three terms:

$$\mathcal L = \underbrace{\mathcal L_{r=1}}_{\text{order within the neighbourhood}}
 + \underbrace{\mathcal L_{r=0}}_{\text{calibrate levels across the pool}}
 + \lambda_{\text{row}}\underbrace{\mathcal L_{\text{row}}}_{\text{reuse pool members as extra anchors}}$$

The design principle in one sentence, for the intro:

> Which comparisons are worth making is a property of the teacher's geometry;
> mini-batches should only decide when it is convenient to compute them.

---

## 6. Evidence map

Derived from the claims, not from what is already in `runs/`. Each entry states
what a reader has to accept, what would falsify it, and the one experiment that
settles it. A claim whose falsifier cannot be stated is not a claim; an entry
whose experiment has not been run is marked as such rather than quietly replaced
by the nearest thing on disk.

**Premise P.** *Batch composition shows a student almost no informative
comparisons, and the shortfall grows with the corpus.*
Falsified if the count $(B-1)k/(N-1)$ is not small at realistic $N$, or if
related pairs are over-represented in batches relative to unrelated ones.
→ **A. The count**: 0.93 per step here, $1/N$ thereafter, and $B \approx 4{,}270$
to fix it by batch size alone. A formula and one figure; no training. (The
per-pair version — every pair has the same probability
$\binom{N-2}{B-2}/\binom{N}{B}$ — belongs in a footnote.)

**C1a.** *Comparisons with the teacher's neighbours beat comparisons with batch
co-occupants, at the same count.*
Falsified if the two tie when the number of comparisons, the loss, the
temperature and the encoder budget are equal.
→ **B. Comparison ladder**, arms `in_batch` vs `teacher`.

**C1b.** *What pays is that the texts are related, not that they come from
outside the batch.*
Falsified if drawing comparisons uniformly from the whole corpus — batch-free
but unrelated — matches the teacher's neighbours.
→ **B**, arm `corpus_uniform`.

**C1c.** *The effect tracks the number of related texts an objective sees, not
the mechanism that supplies them.*
Falsified if raising that number for a batch-local objective — by composing
batches from one neighbourhood — leaves its score unmoved.
→ **C. Dose–response**. Buying the count with batch size is not trained: no
setting matches steps, data passes and learning rate across $B$ at once. It is
stated from A's formula and listed as a limitation.

**C1d.** *It has to be this teacher's notion of similarity, not any model's.*
Falsified if comparisons chosen by the student's own kNN match. This is the arm
that separates us from ANCE empirically rather than rhetorically.
→ **B**, arm `student_knn`. **Not yet run in one-factor form.**

**C1e.** *Batch composition changes what a batch-local objective learns and does
not change what ours learns.*
Falsified if ours moves with batch composition by more than the pointwise floor
moves.
→ **C. Composition intervention**, read as difference-in-differences.

**C2.** *The gain comes from which texts are compared, not from the extra terms
of our loss.*
Falsified if removing our non-standard parts removes the gap in B — or if the
gap in B, which runs a deliberately minimal objective, does not survive on the
shipped one.
→ **D. Component ablation**, on the full objective.

**C3.** *A per-anchor comparison set costs about what a batch-local one costs.*
Falsified if the encoder budget per step is materially higher at the same number
of comparisons.
→ **B**'s cost columns. Already exported per arm; no new runs.

**C4.** *Nothing is tuned.*
Falsified if a derived quantity turns out to need tuning, or if the defaults sit
on a peak.
→ **G. Sensitivity**, plus the three deletion tests in `experiments.md` Stage 1,
which ask whether each derived quantity earns its place at all.

**Coverage.** *The student reproduces teacher relations it was never shown.*
Falsified if held-out teacher pairs are reproduced no better than by an arm that
never saw related pairs at all. Currently **failing** on the existing probe,
whose `pair_order` metric is degenerate.
→ **E. Held-out relations**.

---

## 7. Objections, and the prepared answer

**"This is CompRess with a kNN comparison set."** Correct, and that is the
point: the loss is held fixed at theirs so the choice of compared texts is the
only moving part. The ladder quantifies what moving it is worth (+2.7 on the old
method, at the same count of comparisons). Said first, in related work, with the
number.

**"ANCE already argued this."** For negatives in a labelled discriminative
objective, with an index rebuilt from the student during training. Here there
are no labels, the teacher is frozen, the graph is built once, and the chosen
texts supply the targets rather than only the contrast.

**"You should compare where the student is wrong, not where the teacher says
things are similar."** A fair objection, and a different axis: choosing by the
teacher's geometry versus choosing by the student's current error. We do not
claim the second is unnecessary — we do not test it, except insofar as
`student_knn` touches it. Named as a limitation, not argued away.

**"$\mathcal L_{r=0}$ compares against whatever the batch supplies, which is what
you criticise."** Two answers, decided by measurement (Stage 1.2 and 2C): either
the full objective is measurably unmoved by batch composition, in which case
that term calibrates levels rather than choosing comparisons and we say so with
the number; or it is not, in which case the term changes so that every
supervised comparison is one the teacher chose.

**"Only +0.15 from $\mathcal L_{\text{row}}$."** Reported as what it is: a cheap
add-on that reuses encoded texts, not a pillar. If Stage 1.1 shows its gain does
not depend on the extra anchors being teacher-chosen, it is removed and the
method drops to one hyperparameter.

**"The graph costs $O(N^2d)$."** Stated in the cost paragraph, once, with the
ANN alternative named. It is paid once per (teacher, corpus) and never at
inference.

---

## 8. Paper outline

1. **Introduction** — what a teacher knows about a text; the 0.93 count; the two
   decisions mini-batch training conflates; contributions as *finding first,
   method second*.
2. **Related work** — three blocks: (i) embedding/relational KD and where each
   family gets its comparisons, with the table from §2; (ii) beyond-batch
   supervision (queues, memory banks, full-kernel, ANCE) and why none of them
   compares against related texts specifically; (iii) neighbourhood graphs and
   per-point bandwidths, crediting self-tuning/UMAP/LSP.
3. **Which comparisons** — formalise the choice; alignment and coverage; the
   exposure statement.
4. **GGPKD** — §5 above, presented as the cheapest realisation, not as novelty.
5. **Controlled study** — the ladder and the composition intervention as the
   centrepiece, with cost columns.
6. **Downstream results** — main table, component ablation, sensitivity.
7. **Limitations** — the fraction of a neighbourhood a pool exposes, graph cost,
   the calibration term's comparison set, the held-out result, the student-error
   axis we do not test, what happens at very large batch (not trained), and
   pair-classification being less uniform.
