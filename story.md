# GGPKD — the story

What the paper claims, what it deliberately does not claim, and which sentence
each claim is allowed to appear in. Written after the survey in which the
objective turned out to be prior art; the conclusion of that survey is that the
objective *should* stay prior art, and the paper should sell the finding instead.

---

## 1. Thesis

> Relational knowledge distillation makes two decisions — which relations are
> supervised, and how they are computed — and mini-batch training conflates
> them. Separating the two is worth more than any change to the loss.

Everything else in the paper is in service of that sentence.

---

## 2. The gap

At corpus level a teacher induces $N(N-1)$ relations and no method can supervise
them all. Every relational KD method therefore picks a support, and the
literature picks it by **what is cheap to have on hand**:

| method | support of one anchor | chosen by | labels |
|---|---|---|---|
| SP, RKD, PKT, IRG | the other examples in the batch | batch composition | no |
| CompRess, SEED, ISD | a memory bank / momentum queue | availability | no |
| XBM | cross-batch feature memory | availability | no |
| ANCE, RocketQA | top-k of an ANN index | the student, negatives only | yes |
| LSP (GCN KD) | graph neighbours | the task's own graph | yes |
| **GGPKD** | $\mathcal N_i = \text{top-}k_T(i)$ | **the teacher, whole support** | **no** |

Nobody has made the supervision support of a relational objective a
teacher-selected quantity over a corpus that has no graph and no labels — and,
more to the point, nobody has measured what that choice is worth.

The reason is not oversight, it is cost: a per-anchor support means encoding a
different set of texts for every anchor. Queues and memory banks exist precisely
to avoid that. Section 5 of the paper is the answer to that cost.

---

## 3. What we claim

**C1 (the finding).** Under a fixed relational budget, teacher-selected support
beats batch-local support, and the gain is attributable to teacher *relevance*
rather than to merely leaving the batch or to matching the teacher's row shape.

**C2 (the mechanism is not the point).** The objective is a standard softmax-KL
over a candidate set — the CompRess/SEED form, restricted to $\mathcal N_i$ as
in LSP. We change what the candidate set *is*, and nothing else. The ablations
are designed so that this is checkable rather than asserted.

**C3 (it is affordable).** Deduplicating candidates into a shared pool and
reusing already-encoded pool members as auxiliary row centers makes a
per-anchor teacher-selected support cost roughly what a batch-local objective
costs.

**C4 (nothing is tuned).** Two hyperparameters. Bandwidths, all three
temperatures, the group balance, the candidate set and the set of supervised
rows are derived from the graph. Per-row bandwidth $\tau_i$ is read in closed
form from the retrieval span, which makes each row invariant to the affine scale
of a given teacher's cosines.

---

## 4. What we explicitly do NOT claim

Stating these is what keeps C1–C4 credible.

- **Not a new loss.** Softmax-over-anchors + KL is CompRess (NeurIPS 2020) and
  SEED (ICLR 2021); the same operation restricted to graph neighbours is LSP
  (CVPR 2020). We use them as-is and say so.
- **Not the first to argue that in-batch sampling is a bad sampler.** ANCE
  (ICLR 2021) makes that argument for dense-retrieval negatives and fixes it
  with a global ANN index. Our contribution is to carry it to the *supervision
  support of a distributional objective*, where the target — not just the
  contrast set — changes with the choice.
- **Not a new bandwidth rule.** Per-point bandwidths from the $k$-th neighbour
  are self-tuning spectral clustering (Zelnik-Manor & Perona, 2004); the
  $\log k$ normalisation is UMAP's. Ours is a closed form instead of a solve,
  with affine invariance. That is an engineering simplification, not a concept.
- **Not a proof that geometric distortion controls downstream score.** The
  decomposition into alignment and coverage motivates the design; it does not
  predict benchmark movement.
- **Not global geometry preservation in the strong sense** until E3 is repaired
  (see `experiments.md`, P0-1). Until then the honest claim is
  *teacher-selected relational supervision*, and the title should say that.

---

## 5. The method, as the minimal realisation of the thesis

Offline, once per (teacher, corpus): encode, L2-normalise, retrieve top-$k$ per
node. No reciprocity filter — every relation the teacher retrieved is eligible,
and every node has degree exactly $k$. Read the bandwidth off the retrieval
span, $\tau_i = (s_i^{(1)}-s_i^{(k)})/\log k$, so the $k$-th neighbour is $k$
times less likely than the nearest, for every node, with no constant to choose.
The teacher transition row is the softmax over $\mathcal N_i$ at $\tau_i$, kept
whole — no truncation.

During training the candidate set of an anchor **is** $\mathcal N_i$: no
sampling, no negatives, no RNG, identical in every epoch. A batch encodes the
deduplicated union $\mathcal P_B$ once, and that single encoding pass serves
three terms:

$$\mathcal L = \underbrace{\mathcal L_{r=1}}_{\text{rank within } \mathcal N_i}
 + \underbrace{\mathcal L_{r=0}}_{\text{calibrate across } \mathcal P_B}
 + \lambda_{\text{row}}\underbrace{\mathcal L_{\text{row}}}_{\text{reuse pool members as row centers}}$$

The design principle in one sentence, for the intro:

> Teacher geometry determines which relations are **eligible** for supervision;
> mini-batches serve only to **realise** eligible relations efficiently.

---

## 6. Evidence map — one experiment per claim

A claim with no row here does not go in the paper.

| claim | experiment | status |
|---|---|---|
| C1 teacher relevance is what pays | E2 support intervention, 5 arms at matched support | run (old method), must re-run |
| C1 it is relevance, not "leaving the batch" | E2 arm `corpus_uniform` | run |
| C1 it is relevance, not row shape | E2 arm `rewired` | run |
| C1 batch-local supervision *is* batch-determined | E1 batch intervention | run |
| C1 teacher-selected supervision is *not* | E1 arm `teacher_support` across samplers | run |
| C2 the gain is not from the loss | component ablation on the full objective | **missing** |
| C3 cost is comparable | cost columns already exported per arm | run |
| C4 defaults are not tuned on test | `graph_k` and `row_weight` sweeps | run (old method) |
| method beats strong baselines | main table, 3 settings × 9 datasets | run (old method) |

---

## 7. Objections, and the prepared answer

**"This is CompRess with a kNN anchor set."** Correct, and that is the point:
the loss is held fixed at theirs so that the support is the only moving part.
E2 quantifies what moving it is worth (+2.7 at matched support size). Said
first, in related work, with the number.

**"ANCE already argued this."** For negatives in a labelled discriminative
objective, with an index rebuilt from the student during training. Here the
support is the *whole* supervision — the target distribution itself is defined
on it — the teacher is frozen, the graph is built once, and there are no labels.

**"L_{r=0} selects its support by availability, which is what you criticise."**
Two answers, and which one we use is decided by measurement (P0-2): either the
full objective is measurably invariant to batch composition, in which case the
term is calibration rather than selection and we say so with the number; or it
is not, in which case the term changes (per-pair target, or HT reweighting) so
that every supervised relation is teacher-eligible.

**"Only +0.15 from $\mathcal L_{\text{row}}$."** Reported as what it is: a cheap
add-on that reuses encoded texts, not a pillar. If P2-1 shows its gain does not
depend on the centers being teacher-approved, it is removed and the method drops
to one hyperparameter.

**"The graph costs $O(N^2d)$."** Stated in the cost paragraph, once, with the
ANN alternative named. It is paid once per (teacher, corpus) and never at
inference.

---

## 8. Paper outline

1. **Introduction** — two decisions, mini-batch conflates them, eligibility vs
   realisation, contributions as *finding first, method second*.
2. **Related work** — three blocks: (i) embedding/relational KD and where each
   family gets its support, with the table from §2; (ii) beyond-batch
   supervision (queues, memory banks, full-kernel, ANCE) and why none of them
   selects by teacher relevance; (iii) neighbourhood graphs and per-point
   bandwidths, crediting self-tuning/UMAP/LSP.
3. **Support selection** — formalise the support-selection view; alignment and
   coverage; the statement that batch exposure is independent of teacher
   relevance.
4. **GGPKD** — §5 above, presented as the cheapest realisation, not as novelty.
5. **Controlled study** — E1 and E2 as the centrepiece, with cost columns.
6. **Downstream results** — main table, component ablation, sensitivity.
7. **Limitations** — exposed mass, graph cost, the ambient term's support,
   E3's finding, pair-classification results being less uniform.
