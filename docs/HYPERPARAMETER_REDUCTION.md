# HeatGeo Hyperparameter Reduction — Survey and Proposals

Status: 2026-08-20. Written against `heatgeo-walk-distillation` at `22213f0` + the non-backtracking
walk change.

The goal is not "fewer numbers in the config". It is that every constant in the method can be
answered for, in one sentence, with one of four evidence classes — and that the answer is not
"we tuned it". A method whose constants are *derived from the teacher graph* transfers to a new
teacher-student pair without retuning; a method whose constants are tuned does not, and a reviewer
is right to ask how many runs the tuning cost.

## 0. Where things stand

24 knobs currently change HeatGeo results (excluding paths, workers, wandb). By group:

| group | knobs | count |
|---|---|---|
| teacher graph (build-time, forces cache rebuild) | `graph_k`, `graph_temp`, `diffusion_scales`, `scale_weights`, `pool_size`, `hard_neg_pool`, `walk_keep_topk`, `dedup_corpus` | 8 |
| candidate sampling | `candidate_size`, `diffusion_quota`, `hard_neg_k`, `random_neg_k`, `deterministic_topm`, `stochastic_candidates`, `resample_candidates_per_epoch`, `seed` | 8 |
| loss | `broad_scale_temps`, `student_temp`, `direct_temp`, `direct_weight`, `share_in_batch` | 5 |
| walk term | `num_walks`, `walk_length`, `walk_weight`, `walk_start_epoch`, `walk_topk`, `walk_non_backtracking` | 6 |

`walk_temp`, `scale_temps` and `direct_student_temp` are already gone: the criterion raises if they
are passed, because each is provably tied to `graph_temp` (Group A, Proposition `prop:temptie`).

**The exponent-law direction (α, γ) is closed.** The two commits implementing `temp_exponent` and
`weight_exponent` were deleted from the remote on 2026-08-20 by decision. `student_temp`,
`broad_scale_temps`, `direct_temp`, `direct_weight` and `scale_weights` are therefore live free
knobs again, and nothing below re-proposes a fitted exponent over the scale ladder.

### 0.1 Three defects that inflate the count before any science

Found while taking the inventory; all are cheap to fix and none require an argument.

1. **Eight `getattr` fallbacks in `distiller.py` disagree with `HeatGeoConfig`.** `graph_k` 200 vs
   50, `pool_size` 256 vs 128, `candidate_size` 66 vs 32, `diffusion_quota` 14 vs 20, `hard_neg_k`
   26 vs 8, `random_neg_k` 26 vs 4, `deterministic_topm` 2 vs 4, and — worst — `num_walks` 4 vs
   **0** at five separate call sites. Any config object that omits one of these silently gets a
   different method, and for `num_walks` that means the walk term is off with no error. There is no
   such thing as a documented default while two defaults exist.
2. **Seven dead knobs.** `cache_teacher`, `lambda_heatgeo`, `lambda_cosine`, `lambda_infonce`,
   `lambda_sim`, `lambda_simcse`/`simcse_temp`/`simcse_start_epoch` (multi-layer branch only; the
   shipped config takes the single-layer branch), plus `w_task`/`alpha_dtw` inherited from
   `BaseConfig` and printed in the run banner. They are in the config, in the banner, and in a
   reviewer's mental count of the method's knobs, and none of them do anything.
3. **Three knobs read but never declared**: `eps_norm`, `heatgeo_anchor_column`,
   `heatgeo_source_column` are read via `getattr` but absent from `HeatGeoConfig`, so
   `HeatGeoConfig(**kwargs)` cannot set them — its `__init__` assigns only `if hasattr(self, k)`,
   which also means **any misspelled override is silently discarded**.

Fixing these takes the *stated* knob count from 24 to 17 and costs no evidence at all.

## 1. Evidence tiers

Unchanged from the framework agreed on 2026-08-18:

- **T1 — proof.** The value is forced; any other value makes the objective's zero set empty, or
  provably discards information. Example: `walk_temp = graph_temp`.
- **T2 — citation / established convention.** A standard criterion from the literature this method
  sits in, applied without modification.
- **T3 — measured.** Derived from the teacher graph or from logged training statistics against a
  **stated tolerance**. The tolerance is reported; the knob is not.
- **T4 — admitted free.** Kept as a knob, ablated honestly, with the search budget reported.

A fitted formula with a free exponent is none of these. That is why α/γ is closed.

## 2. What the literature actually licenses

### 2.1 Bandwidth selection: entropic affinities

Setting one global softmax temperature over neighbour similarities is the problem
*entropic affinities* were introduced to solve. Hinton and Roweis (SNE, NIPS 2002) set the scale
individually per point so that each point's neighbour distribution has a prescribed **perplexity**
(effective number of neighbours); t-SNE (van der Maaten and Hinton, JMLR 2008) inherits it;
Vladymyrov and Carreira-Perpiñán (ICML 2013) establish the mathematical properties and the
efficient numerical computation. Zelnik-Manor and Perona (NIPS 2004) reach the same conclusion from
the spectral-clustering side with local scaling by the distance to the k-th neighbour (they use
k = 7).

The knowledge-distillation literature arrives at the same place independently: Sun et al.,
*Logit Standardization in Knowledge Distillation* (CVPR 2024 Highlight), show that the shared-
temperature assumption forces an unnecessary match of logit **range and variance** between teacher
and student, and replace the tuned temperature with the weighted standard deviation of the logits
(a z-score pre-process). Both lines say the same thing: a temperature should be **derived from the
scale of the scores it is applied to**, not tuned.

### 2.2 Diffusion time: the entropy knee

This is the literature the method is named after, and it already has an automatic criterion.

- Huguet, Tong, De Brouwer, Zhang, Wolf, Adelstein and Krishnaswamy, *A Heat Diffusion Perspective
  on Geodesic Preserving Dimensionality Reduction* (NeurIPS 2023) — §"Optimal diffusion time":
  *"we select t with the knee-point method on the function t ↦ H(H_t)"*, with
  H(H_t) := −Σ_{i,j} (H_t)_{ij} log (H_t)_{ij}. They build on a kNN graph with adaptive bandwidth.
- PHATE (Moon et al., *Nature Biotechnology* 2019) selects the diffusion time as the knee point of
  the **von Neumann entropy** of the diffusion operator.

Our `diffusion_scales = (1, 2, 4)` is a hand-picked ladder in a family whose own papers pick the
time automatically.

### 2.3 Graph connectivity: how large k has to be

Brito, Chávez, Quiroz and Yukich, *Connectivity of the mutual k-nearest-neighbor graph in
clustering and outlier detection* (Statistics & Probability Letters 35(1):33–42, 1997) is the
reference result for mutual-kNN connectivity; von Luxburg's spectral-clustering tutorial (2007)
gives the k ∼ log n rule of thumb. Our corpus is n ≈ 13.5k after dedup, so log n ≈ 9.5 and
`graph_k = 200` is roughly 20× the connectivity requirement.

### 2.4 Loss weighting

The standard answers — Kendall, Gal and Cipolla (CVPR 2018, uncertainty weighting) and Chen et al.
(GradNorm, ICML 2018) — both **learn** the weights, which means adding trainable parameters to the
criterion. That is the specific thing this criterion refuses to do: the pointwise anchor term with
its free map `W_a` was removed rather than left at weight 0, on the grounds that "a knob that
cannot work is worse than no knob". A learned λ also cannot be reported as a property of the
method. §3.6 gives a counting argument instead.

### 2.5 Reporting whatever stays free

Dodge et al., *Show Your Work* (EMNLP 2019) and Bouthillier et al. (2021) are the citations for
reporting the tuning budget rather than the tuned value; Li et al. (Hyperband, JMLR 2018) and ASHA
are the tools if a search is actually run. This matters for the framing of the whole exercise: the
point of the reduction is that the T4 residue is small enough for an honest budget report to be
short.

## 3. Proposals

Ordered by (evidence strength × knobs removed) ÷ cost.

### P0. Housekeeping — removes 7 knobs, needs no argument

> **Status 2026-08-20 — P0 complete.** Removed from `HeatGeoConfig`:
> `cache_teacher`, `lambda_heatgeo`, `lambda_cosine`, `lambda_infonce`, `lambda_simcse`,
> `simcse_temp`, `simcse_start_epoch`, `lambda_sim`; removed from `BaseConfig`:
> `evaluate_test_each_epoch`. Every consumer of those reads through `getattr` with the same
> default the deleted line carried (the one direct access, `cfg.lambda_heatgeo` in the multi-layer
> branch, was changed to `getattr(cfg, "lambda_heatgeo", 1.0)`), so behaviour is unchanged; tests
> pass and all six method configs still build. `w_task`, `alpha_dtw`, `w_cls` and `temperature`
> stay in `BaseConfig` — they are dead for HeatGeo but live for cdm/dskd/stella/emo/talas, and
> `talas_config` does not declare `alpha_dtw`/`w_cls`, so moving them per-method needs its own
> verification pass.
>
> The 31 `getattr(cfg, "...", fallback)` reads on the HeatGeo path are now direct attribute
> access, so the config is the single source of truth and a config that cannot express the method
> raises instead of silently running a different one — `num_walks` in particular can no longer
> fall back to 0 and turn the walk term off without a word. Two sites were handled differently:
> the walk curriculum runs for every method, so it is now guarded by
> `cfg.distill_method == "heatgeo"` rather than by a fallback; and `resample_candidates_per_epoch`
> in `train_epoch` deliberately keeps its fallback, because that code is gated on the dataset
> having `set_epoch`, not on the method. `eps_norm`, `heatgeo_anchor_column` and
> `heatgeo_source_column` are now declared, and `HeatGeoConfig.__init__` raises on an unknown
> option instead of discarding it.

Delete the dead knobs, reconcile the eight fallbacks with `HeatGeoConfig` (or drop the fallbacks and
let a missing attribute raise — preferable: a config that cannot express the method should fail
loudly), declare `eps_norm` / `heatgeo_anchor_column` / `heatgeo_source_column`, and make
`HeatGeoConfig.__init__` raise on an unknown kwarg instead of discarding it.

Tier: n/a. 24 → 17 stated knobs. Do this first — every later measurement is worthless if the config
that produced it disagrees with itself.

### P1. `graph_temp` → per-row entropic affinity at a fixed perplexity — **T1 + T2**

Replace the single global temperature with a per-node τ_i chosen so that each transition row has a
prescribed entropy:

> H(P(·|i)) = log K,  K = target perplexity (effective number of neighbours)

**Well-posedness (T1).** With p_j ∝ exp(β s_j), β = 1/τ, we have H(β) = log Z(β) − β E_p[s] and

> dH/dβ = −β Var_p(s) < 0 for β > 0 and non-constant s,

so H is strictly monotone in τ, ranging over (0, log d_i). For any target in that interval there is
a **unique** τ_i, and bisection converges — no tuning, no failure mode. (This is exactly the
property Vladymyrov and Carreira-Perpiñán prove and exploit.)

**Why this is the knob that matters (T1).** Under an affine rescaling of the teacher's cosine scale,
s → a·s + b with a > 0 — which is precisely what changes between Qwen and a differently-calibrated
teacher — the entropic-affinity solution moves to τ_i' = a·τ_i and the resulting row P is
**exactly invariant**:

> exp((a s_j + b)/(aτ)) ∝ exp(s_j/τ).

A fixed `graph_temp` has no such invariance, which is the mechanism behind having to retune it per
teacher-student pair. The perplexity K is scale-free; the temperature is not.

**Compatibility with the existing proofs.** The tie generalizes verbatim per row: row i's target is
a softmax at τ_i, the student's row-i softmax uses the same τ_i, and the attainable set is still the
shift family cos_S(i,·) = cos_T(i,·) + a_i — which does not depend on τ_i. So `prop:temptie` holds
row-wise, and the gauge argument (`prop:walkgauge`), which only uses cosine symmetry across a
traversed edge, is untouched. **Verify this in the proof text before shipping**, since the
proposition is currently stated for a scalar τ.

Cost: build-time change (cache rebuild); the criterion must gather τ per row for the r=1 and walk
softmaxes. K keeps a literature default (t-SNE's 30) but stays T4 — see §4.

### P2. `diffusion_scales` / r_max → entropy knee — **T2 + T1**

Adopt the criterion from the paper this method is named after: compute t ↦ H(H_t) and take the
knee to get t\*, then set the ladder to the dyadic scales covering [1, t\*] instead of hard-coding
(1, 2, 4).

**Proof complement (T1).** Scales past mixing carry no anchor-specific information. For the lazy
walk (I+P)/2 with second eigenvalue λ₂,

> ‖P^t(i,·) − π‖_TV ≤ ½ √((1−π_i)/π_i) · λ_*^t,   λ_* = (1+λ₂)/2

(Levin and Peres, *Markov Chains and Mixing Times*, Ch. 12). Once 2^r exceeds the mixing time, every
anchor's target is the same distribution π, its KL term stops depending on the anchor, and the
scale supervises nothing. λ₂ is measurable by Lanczos on the sparse operator, per connected
component. Two independent criteria — the literature's knee and the spectral bound — should agree
on the same ceiling; if they disagree, that disagreement is itself worth a paragraph in the paper.

Removes `diffusion_scales`. Note this is *not* the α/γ direction: nothing is fitted, the ladder is
read off the operator's spectrum.

### P3. Capacity knobs → one tolerance — **T1**

`pool_size`, `walk_keep_topk`, `hard_neg_pool`, `walk_topk` are all the same operation: keep a
subset S of a probability row and renormalize. If the discarded mass is δ = 1 − p(S), then the
renormalized row p̃ satisfies, exactly,

> TV(p, p̃) = δ,  KL(p̃ ‖ p) = −log(1 − δ) ≤ δ/(1−δ).

The perturbation to the target is bounded **in nats — the units of the loss**. With δ = 1%, the
targets are perturbed by ≤ 0.01 nats against a total loss around 0.84. So: pick one tolerance δ,
set every capacity to the smallest value whose *measured* residual mass is under it, and report δ.
Four knobs collapse into one reported tolerance, and the graph builder already computes and warns
on the truncation mass, so most of the machinery exists.

### P4. `graph_k` → connectivity criterion — **T2 + T3**

Set k to the smallest value with (a) zero fallback rows (the builder already logs `fallback_count`)
and (b) a giant component covering ≥ 1 − ε of nodes. Literature says k ∼ log n ≈ 9.5 suffices for
connectivity; we run 200.

Two measured facts already point the same way: the walk probe shows `hops_reached` ≈ 1.0 at k = 200
— walks essentially never leave the anchor's one-hop neighbourhood, because that neighbourhood is
~170 nodes wide — and non-backtracking buys +1% there versus +4.5% at k = 20. A smaller k makes the
graph a graph rather than a near-clique, and makes the walk term's premise true.

Practical note: the builder floors its cosine top-k at `max(graph_k, hard_neg_pool + pool_size)` =
456, so lowering `graph_k` alone does not reduce build cost — `hard_neg_pool` and `pool_size` must
come down with it (P3 gives the criterion for those).

### P5. `num_walks`, `walk_length` → tolerance on already-logged ratios — **T3**

The usable-supervision budget is bounded by the candidate slots that can hold walk nodes. Choose the
smallest M·L with measured `walk_node_hit_ratio` ≥ 1 − ε and `walk_valid_ratio` ≥ 1 − ε; both
metrics already exist in `_compute_walk_loss`. Same tolerance as P3.

### P6. `walk_weight` → row-count tie — **T2 (definitional), weakest of the set**

Treat the objective as one estimator: expected KL per supervised conditional, uniform over
supervised conditionals. With B anchors and R_w walk rows per batch,

> L = (B·L_diff + R_w·L_walk)/(B + R_w) ∝ L_diff + (R_w/B)·L_walk,  so λ = E[R_w]/B.

λ becomes measurable from `walk_rows`, which is already logged, and adapts across teacher pairs
instead of being retuned.

**Two honest caveats.** First, "uniform over rows" is a choice, not a theorem — it is defensible and
*stated*, which 0.5 is not, but it is not T1. Second, the tie almost certainly predicts a much
larger λ than 0.5 (with B = 32 and up to 16 visits per anchor, R_w/B can be an order of magnitude
above the current value), so this is a real change to the objective's balance, not a relabelling.
**Measure λ_implied from existing logs before adopting it.** If the measured value is far from 0.5
and performance drops, the honest outcome is to keep λ free (T4), not to bend the counting.

### P7. `scale_weights`, `broad_scale_temps` → measured sensitivity, not fiat — **T3 or T4**

These came back when the exponent laws were deleted, and no formula will be proposed for them. The
non-fiat route is to measure: sweep each over a range and report the sensitivity curve. If the
objective is flat, fix at the flat region's centre and publish the curve as the evidence (T3); if it
is not flat, it is a genuine design choice and belongs in the ablation table (T4). A published
sensitivity curve is stronger with reviewers than either a tuned value or an unfalsifiable formula.

## 4. Ledger

| knob | now | after | tier |
|---|---|---|---|
| 7 dead knobs | free | gone | — |
| `graph_temp` | free | perplexity K | T1+T2 |
| `diffusion_scales` | free | entropy knee / spectral gap | T2+T1 |
| `pool_size`, `walk_keep_topk`, `hard_neg_pool`, `walk_topk` | 4 free | one tolerance δ | T1 |
| `graph_k` | free | connectivity + measured | T2+T3 |
| `num_walks`, `walk_length` | 2 free | same tolerance δ | T3 |
| `walk_weight` | free | E[R_w]/B, or stays free | T2 or T4 |
| `scale_weights`, `broad_scale_temps`, `student_temp` | 3 free | sensitivity curve | T3 or T4 |
| `candidate_size`, `batch_size`, `lr`, `epochs` | free | compute budget, stated as such | T4 |
| `walk_start_epoch`, `direct_temp`, `direct_weight`, hard/random split | 4 free | ablated | T4 |

Roughly: 24 stated knobs → 2 reported constants (perplexity K, tolerance δ) + a T4 residue small
enough to ablate honestly in one table.

## 5. Order of work

1. **P0 housekeeping** — no evidence needed, and it makes every later measurement trustworthy.
2. **Measurement pass, no code changes to the method**: from one existing artifact, compute the
   entropy curve H(H_t) and its knee, λ₂ per component, residual truncation mass at current
   capacities, the connectivity/fallback curve vs k, and λ_implied = E[R_w]/B from training logs.
   This is one offline script and it decides P2, P3, P4 and P6 before anything is retrained.
3. **P1 (entropic affinity)** — biggest single win (kills per-teacher retuning) but touches the
   graph build and the proof text; do it once the measurement pass has confirmed the rest.
4. **P7 sensitivity curves** — expensive (training runs), so last, and only for what remains.

## 6. What not to do

- **Do not re-introduce a fitted exponent law** over temperatures or scale weights. Closed.
- **Do not run black-box HPO over the whole space.** Most of these knobs are build-time: each trial
  is a graph rebuild, and a per-teacher-pair tuned configuration is exactly the thing that does not
  transfer and that reviewers discount. HPO is for the T4 residue only, with the budget reported.
- **Do not make λ a learned parameter** (uncertainty weighting / GradNorm). The criterion is
  deliberately parameter-free, and a learned weight is not a reportable property of the method.
- **Do not scale any knob by an entropy "signal"** without a derivation — that is the retrofitted-
  formula pattern already rejected once.

## References

Brito, Chávez, Quiroz, Yukich (1997), *Connectivity of the mutual k-nearest-neighbor graph in
clustering and outlier detection*, Statistics & Probability Letters 35(1):33–42 ·
Bouthillier et al. (2021), *Accounting for variance in machine learning benchmarks* ·
Chen et al. (2018), *GradNorm*, ICML ·
Dodge et al. (2019), *Show Your Work*, EMNLP ·
Hinton, Roweis (2002), *Stochastic Neighbor Embedding*, NIPS ·
Hinton, Vinyals, Dean (2015), *Distilling the Knowledge in a Neural Network* ·
Huguet, Tong, De Brouwer, Zhang, Wolf, Adelstein, Krishnaswamy (2023), *A Heat Diffusion Perspective
on Geodesic Preserving Dimensionality Reduction*, NeurIPS, arXiv:2305.19043 ·
Kendall, Gal, Cipolla (2018), *Multi-Task Learning Using Uncertainty to Weigh Losses*, CVPR ·
Levin, Peres (2017), *Markov Chains and Mixing Times*, 2nd ed., Ch. 12 ·
Li et al. (2018), *Hyperband*, JMLR ·
Moon et al. (2019), *Visualizing structure and transitions in high-dimensional biological data*
(PHATE), Nature Biotechnology ·
Sun, Ren et al. (2024), *Logit Standardization in Knowledge Distillation*, CVPR Highlight,
arXiv:2403.01427 ·
van der Maaten, Hinton (2008), *Visualizing Data using t-SNE*, JMLR ·
Vladymyrov, Carreira-Perpiñán (2013), *Entropic Affinities: Properties and Efficient Numerical
Computation*, ICML ·
von Luxburg (2007), *A Tutorial on Spectral Clustering* ·
Zelnik-Manor, Perona (2004), *Self-Tuning Spectral Clustering*, NIPS.
