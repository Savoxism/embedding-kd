# Paper Flow: Corpus-Defined Relational Distillation

## 1. Central thesis

The paper should be built around one principle:

> **A mini-batch should determine which relational constraints are evaluated now, not what relational geometry exists.**

Short version:

> **Define relations at corpus scope; compute them through mini-batches.**

The problem is therefore not relation selection. It is the **entanglement between the scope of the relational objective and the unit of computation**.

GGPKD should be presented as a corpus-defined relational objective that is executed efficiently through mini-batches, not as a teacher-guided sampling, mining, or batch-construction method.

## 2. Positioning guardrails

### What the paper claims

- Text embedding behavior is relational: similarity, ranking, and neighborhood structure matter directly to downstream use.
- Conventional relational KD commonly constructs relational targets from the current mini-batch.
- This makes the learned geometry depend on an optimization artifact: batch composition.
- The relational target should instead be defined independently over the training corpus.
- A sparse, persistent corpus graph makes this objective tractable without a dense all-pairs matrix.
- Mini-batches then sample row centers and organize the computation of their predefined relational constraints.
- Candidate pooling and auxiliary-row reuse amortize student encoding cost.

### What the paper does not claim

- The teacher “selects the best relations.”
- GGPKD is a teacher-guided batching or hard-example-mining method.
- Relations should be prioritized according to teacher relevance or teacher probability mass.
- Every mini-batch pairwise estimator is mathematically biased.
- A sparse one-hop graph exactly preserves every pairwise relation in the corpus.
- Better geometry automatically guarantees improvement on every downstream task.
- The current method performs multi-scale diffusion, unless multi-hop scales are restored and evaluated.

### Preferred vocabulary

| Prefer | Avoid |
|---|---|
| corpus-defined relational target | teacher-selected relations |
| persistent relational scaffold | informative relation selection |
| batch-independent graph row | teacher-relevant support |
| scope--computation decoupling | support-selection problem |
| graph-defined candidates | mass-aware candidates |
| numerical tail truncation | teacher-mass coverage |
| mini-batch execution / row sampling | teacher-guided batching |
| global in scope, local in realization | exhaustive global preservation |

## 3. Narrative in one pass

1. Large text embedding models are effective but expensive to deploy.
2. Distillation can produce compact students, but pointwise alignment alone does not directly preserve the similarities and rankings used by embedding applications.
3. Relational KD addresses this by matching relations among examples, yet these relations are usually instantiated only inside the current mini-batch.
4. This creates a conceptual mismatch: the target is a common embedding space over the corpus, while the training signal is a sequence of transient batch-local geometries.
5. For row-normalized relational objectives, a change in batch composition changes the target distribution itself, not merely the subset evaluated in that iteration.
6. A dense corpus-level relational objective would remove this dependence but costs quadratic pairwise computation.
7. GGPKD resolves the tension by defining a fixed sparse relational scaffold over the corpus and using mini-batches only to execute rows of that scaffold.
8. Shared candidate encoding makes the fixed-row objective practical; auxiliary rows reuse already encoded texts to instantiate additional graph constraints.
9. Experiments should show not only downstream gains, but also reduced dependence on batch size/partition and improved preservation of fixed corpus geometry at comparable compute.

## 4. Title and framing options

Recommended title:

> **Beyond Batch-Local Relations: Corpus-Level Geometry Distillation for Text Embeddings**

More method-centric alternative:

> **GGPKD: Global Geometry through Batch-Independent Local Alignment**

More technical alternative:

> **Decoupling Relational Objectives from Mini-Batch Computation in Text Embedding Distillation**

If the current title is retained, define “global” explicitly:

> Global refers to the scope at which the relational scaffold is defined—the entire training corpus—not to a dense all-pairs loss.

## 5. Abstract flow

The abstract should contain six moves.

### Move 1: Application and problem

Text embedding models serve similarity-based applications, but high-capacity models are costly to deploy.

### Move 2: Why relational distillation

Compact students should preserve the relations induced by the teacher, rather than only match individual representations.

### Move 3: Identify the batch-local mismatch

Conventional relational KD defines its targets among examples in the current mini-batch. As a result, batching controls both computation and the relational geometry being optimized.

### Move 4: Principle

Corpus geometry should be defined independently of batch composition; mini-batches should only provide a tractable execution unit.

### Move 5: Method

GGPKD builds a fixed sparse graph from cached corpus embeddings. During training, sampled anchors retrieve their predefined graph rows, whose candidates are pooled and deduplicated. The same encoded pool is reused for auxiliary graph-row supervision.

### Move 6: Evidence

State the strongest supported result across the three teacher--student pairs and nine tasks. Emphasize STS if that remains the most consistent result. Only claim reduced batch dependence or better corpus-geometry preservation after the corresponding experiments have been added.

### Draft abstract skeleton

> Text embedding models support retrieval, clustering, and semantic similarity through the geometry they induce among texts, yet high-capacity embedding models can be costly to deploy. Relational knowledge distillation seeks to transfer this geometry to a compact student, but existing objectives are commonly instantiated only among examples in the current mini-batch. This makes mini-batch composition determine not only which constraints are computed but also the relational target itself. We argue that relational supervision should be defined at corpus scope, while mini-batches should serve only as units of computation. Based on this principle, we introduce Global Geometry Preservation Knowledge Distillation (GGPKD), which constructs a fixed sparse relational scaffold from cached corpus embeddings. At each update, GGPKD evaluates predefined graph rows for the sampled anchors, deduplicates their candidates into a shared encoding pool, and reuses the pool to supervise additional graph rows without extra encoder passes. [Insert verified empirical results.] GGPKD requires neither task labels nor online teacher inference, and deployment uses only the student encoder.

Do not mention alignment--coverage decomposition, teacher mass, mass-aware sampling, or multi-scale diffusion in the abstract under the current method.

## 6. Introduction flow

### Paragraph 1: Why compact text embeddings

**Purpose:** Establish deployment need without repeating it.

Content:

- Embedding models support semantic search, retrieval, clustering, STS, and classification.
- Larger encoders often provide stronger representations but incur memory, latency, and serving cost.
- This motivates distilling their behavior into compact encoders.

Exit sentence:

> The central question is therefore not only how to compress an embedding model, but which aspects of its embedding behavior a compact student must retain.

### Paragraph 2: Embedding knowledge is relational

**Purpose:** Move from generic KD to relational KD.

Content:

- Pointwise matching transfers individual embeddings or hidden representations.
- Embedding applications consume similarities, rankings, and neighborhoods.
- Two models may align individual features imperfectly yet preserve useful relations, or match pointwise targets without adequately controlling corpus geometry.
- Relational KD directly constrains pairwise similarities, distances, angles, or conditional affinities.

Exit sentence:

> For text embeddings, the natural object of distillation is therefore a geometry shared by many examples, rather than a collection of isolated vectors.

Candidate citations: `tung2019similarity`, `park2019rkd`, `passalis2020pkt`, `kim2023embeddistill`, `kim2024topkd`.

### Paragraph 3: The scope--computation entanglement

**Purpose:** State the actual research problem.

Content:

- Relational KD is normally implemented with a mini-batch relation matrix because dense corpus computation is expensive.
- This implementation silently gives mini-batches two roles:
  1. they provide the examples that fit in memory;
  2. they define the relational universe against which every anchor is compared.
- The first role is necessary for stochastic optimization; the second is not intrinsic to the distillation objective.
- Across updates, an anchor is therefore trained against changing, batch-contingent local coordinate systems rather than a persistent corpus-level reference.

Core sentence:

> This conflates the scope of relational supervision with the unit of computation.

Exit sentence:

> A mini-batch should determine when a relational constraint is evaluated, not what that constraint means.

Candidate citations: `tung2019similarity`, `park2019rkd`, `passalis2020pkt`, `qian2022fullkernel`, `ckabert`.

### Paragraph 4: Why this is more than incomplete pair coverage

**Purpose:** Give a precise technical argument without teacher relevance or mass.

For a batch-defined normalized teacher row,

\[
q^T_{i,B}(j)
=
\frac{\exp(c^T_{ij}/\tau)}
{\sum_{u\in B\setminus\{i\}}\exp(c^T_{iu}/\tau)}.
\]

Changing $B$ changes both the available columns and the normalization. Consequently, the same pair can receive a different target value under a different batch. In general,

\[
\mathbb E_B\!\left[q^T_{i,B}(j)\right]
\neq
q^T_i(j),
\]

where $q^T_i$ denotes a target defined relative to a fixed corpus-level comparison set.

Important qualification:

> This criticism is specific to objectives whose relational target is batch-conditioned. An additive loss over uniformly sampled independent pairs can still be an unbiased estimator of a full pairwise objective, although it may have sparse exposure or high variance.

This qualification prevents the paper from making an overly broad claim about all mini-batch relational losses.

### Paragraph 5: Existing routes beyond the batch and the remaining design need

**Purpose:** Establish novelty without claiming that batch locality is newly discovered.

Content:

- Full Kernel Matrix Transfer explicitly targets dataset-level pairwise structure and uses Nyström approximation.
- Memory-based methods expand the comparison set beyond the current batch.
- Topological methods summarize more global structure.
- These works establish the value of going beyond batch-local relations.
- GGPKD occupies a different design point: a fixed sparse corpus scaffold, fresh student encodings for the constraints evaluated in the current update, and no online teacher.

Required design properties:

1. corpus-level in definition;
2. sparse in representation;
3. compatible with mini-batch optimization;
4. economical in student encoder calls;
5. absent from deployment.

Candidate citations: `qian2022fullkernel`, `ckabert`, `kim2024topkd`. Cross-Batch Memory and MoCo may be cited as adjacent representation-learning precedents after adding them to the bibliography.

### Paragraph 6: Principle and method overview

**Purpose:** Introduce GGPKD as the resolution of the scope--computation problem.

Content:

- Encode the training corpus once with the frozen teacher.
- Construct a mutual kNN graph that defines a persistent relational row for every corpus item.
- Treat the graph as a sparse representation of corpus geometry, not as a batching policy.
- Sample ordinary mini-batch anchors.
- Fetch the predefined row associated with every sampled anchor.
- Union and deduplicate the required candidate texts into a shared student-encoding pool.
- Match fixed teacher graph rows with student rows.
- Reuse non-anchor candidates as auxiliary row centers where their graph neighbors are already present.

Key wording:

> GGPKD does not form batches by grouping teacher-similar examples. It samples anchors normally and materializes the predefined relational constraints needed to train those anchors.

Exit sentence:

> In this way, the corpus defines the geometry, while the mini-batch only schedules and amortizes its computation.

### Paragraph 7: Contributions

Recommended contribution list:

1. We identify a scope--computation entanglement in relational distillation: mini-batches are commonly used both to define relational targets and to compute them.
2. We formulate corpus-level relational distillation as stochastic optimization over persistent relational rows, separating the target geometry from mini-batch composition.
3. We introduce GGPKD, which realizes this formulation through a cached sparse corpus graph, batch-independent anchor rows, shared candidate encoding, and auxiliary-row reuse.
4. We evaluate GGPKD across three teacher--student settings and nine downstream datasets, and [insert only evidence supported by the final experiments].

Do not use “first” unless a broader and systematic literature search supports it.

## 7. Formal problem setup and theoretical story

### 7.1 Dense ideal

Let $\mathcal D=\{x_i\}_{i=1}^N$ be the training corpus and $c^T_{ij}$, $c^S_{ij}$ the teacher and student cosine similarities. Begin with an ideal corpus-level relational objective:

\[
\mathcal L_{\mathrm{dense}}
=
\frac{1}{N}
\sum_{i=1}^{N}
D\!\left(Q_i^T, Q_i^S\right),
\]

where each row is defined against a corpus-level reference set. State that direct evaluation is infeasible because it requires dense pairwise similarities and large student encoding/comparison cost.

This dense objective is motivation only; do not claim GGPKD exactly optimizes it unless a bound is provided.

### 7.2 Failure mode of a batch-defined target

Define $Q_{i,B}^T$ over $B\setminus\{i\}$. Explain that $B$ affects the row domain and its normalization. Therefore repeated batch losses are not automatically stochastic evaluations of one fixed normalized corpus row.

A small three- or four-point counterexample can make this concrete: keep anchor $i$ fixed, place it once with a close and a distant item, then with two close items, and show that the target probability for the same neighbor changes solely because the other batch member changed.

### 7.3 Sparse corpus objective

Let $G_T$ be a fixed graph constructed once over the full corpus and let $\Omega_i$ be the fixed columns associated with node $i$. Define

\[
\mathcal L_G
=
\frac{1}{N}
\sum_{i=1}^{N}
D_{\mathrm{KL}}\!\left(
q_i^T
\,\middle\|\,
p_i^S
\right),
\]

where both rows are evaluated on the same fixed $\Omega_i$.

The graph is global in scope because every row is defined with access to the corpus, while each constraint is local and sparse.

### 7.4 Mini-batch as row sampling

For uniformly sampled anchors $B$, define

\[
\widehat{\mathcal L}_{G,B}
=
\frac{1}{|B|}
\sum_{i\in B}
D_{\mathrm{KL}}\!\left(
q_i^T
\,\middle\|\,
p_i^S
\right).
\]

Because the row target and its columns do not depend on which other anchors occur in $B$,

\[
\mathbb E_B
\left[\widehat{\mathcal L}_{G,B}\right]
=
\mathcal L_G.
\]

This is the clean theoretical statement of scope--computation decoupling. It requires only uniform row sampling and fixed anchor rows; it does not require teacher-mass or coverage arguments.

### 7.5 Limit the scope of the theorem

The unbiased row-sampling statement applies directly to the fixed graph-row loss $\mathcal L_{r=1}$. It does not automatically apply to:

- the current ambient loss, whose columns are $\mathcal P_B$ and therefore depend on the batch;
- auxiliary rows restricted to whichever neighbors happen to be present in $\mathcal P_B$.

Present these as computationally motivated auxiliary regularizers, not as part of the theorem. Alternatively, redefine their candidate domains offline if the whole objective must satisfy strict batch independence.

## 8. Method section flow

Recommended section title:

> **GGPKD: Corpus-Level Geometry through Mini-Batch Computation**

### 8.1 Overview

Start by separating two stages:

1. **Corpus stage:** build and cache a fixed relational scaffold.
2. **Training stage:** sample graph rows and execute them through deduplicated student encodings.

State explicitly that batches contain ordinary anchors; the method does not create batches of nearest neighbors.

### 8.2 Corpus relational scaffold

- Encode every deduplicated training text once with the frozen teacher.
- Normalize embeddings and construct mutual top-$k$ neighborhoods.
- Use a fallback raw top-$k$ row for isolated nodes.
- Derive the row bandwidth from the raw retrieval range.
- Convert each neighborhood to a fixed transition row.
- Treat the $0.99$ prefix as numerical tail truncation only; do not elevate it into a conceptual notion of teacher-mass coverage.

Explain why mutual kNN is used in structural terms: it yields a sparse and more stable local scaffold by retaining reciprocal neighborhoods. Avoid saying that it selects the teacher's most important relations.

### 8.3 Mini-batch row execution

- Sample anchor indices using the normal data loader.
- Retrieve each anchor's fixed graph row.
- Form the union of texts required by those rows.
- Deduplicate the union so every candidate is encoded once.
- Make clear that $\mathcal P_B$ is a computational pool, not the definition of the anchor relation.

### 8.4 Fixed anchor-row objective

Present $\mathcal L_{r=1}$ as the core loss and direct realization of the paper thesis. Its target row, candidate columns, and temperature are fixed before mini-batch training.

### 8.5 Pool-level calibration

If $\mathcal L_{r=0}$ is retained, describe it as an auxiliary pool-level calibration term. Acknowledge that it is conditioned on the current shared pool and therefore is not the component that establishes batch-independent relational supervision.

Do not call $r=0$ a global or ambient corpus objective. The current pool contains only the union needed by the sampled graph rows.

### 8.6 Auxiliary-row reuse

- Candidate texts are already represented by fresh student embeddings.
- Some candidate-to-candidate graph constraints can therefore be evaluated without additional encoder calls.
- The row loss opportunistically reuses these embeddings.
- Its purpose is amortization: execute more predefined graph constraints per forward pass.
- Do not claim that auxiliary rows improve teacher-mass coverage.

### 8.7 Cost

Separate costs clearly:

- one-time teacher encoding;
- graph construction, exact or approximate;
- per-update number of unique student texts encoded;
- similarity/KL computation;
- zero graph/teacher overhead at deployment.

Report cost against batch-local baselines using both wall-clock time and unique student encodings, not only the nominal number of anchors.

## 9. Related Work flow

### 9.1 Text embedding distillation

Cover pointwise representation alignment, token/layer alignment, contrastive distillation, and embedding-specific methods. End by explaining why downstream embedding behavior motivates relation-level transfer.

The current Sparse Logit Sampling paragraph should not lead the section because truncation bias and probability mass no longer motivate the method. It can be removed or moved to a brief paragraph on efficient distribution transfer if needed.

### 9.2 Relational knowledge distillation

Organize by the relational object transferred:

- pairwise similarities;
- distances and angles;
- conditional probability rows;
- instance graphs;
- topology.

Then distinguish whether the object is defined within a batch or beyond the batch.

### 9.3 Beyond batch-local relational learning

Discuss:

- Full Kernel Matrix Transfer: dense dataset geometry approximated using landmarks;
- memory-augmented global feature structures;
- Cross-Batch Memory and MoCo as adjacent examples of decoupling comparison scope from batch size;
- topology/graph approaches as persistent structural representations.

Close with the exact paper position:

> GGPKD represents corpus geometry as fixed sparse local rows and evaluates those rows with fresh, pooled student embeddings. Its graph determines the relational objective, whereas mini-batches determine only which rows are executed together.

Do not describe GGPKD as BINGO-style teacher grouping. Also replace the stale method name “RIPPLE” with GGPKD.

## 10. Experiment flow

The experiments must test the central claim, not only downstream accuracy.

### RQ1: Does corpus-defined relational training improve downstream embeddings?

- Three teacher--student pairs.
- Nine downstream datasets.
- Separate classification, pair classification, and STS.
- Report in-domain, out-of-domain, and overall averages.
- Do not claim uniform superiority if a setting or task family is mixed.

### RQ2: Is the gain caused by decoupling relations from batch composition?

Required controlled comparison:

- **Batch-local row alignment:** construct and normalize each relational row using current batch examples.
- **Fixed corpus-row alignment:** use persistent graph rows.
- Match anchor count, relation count, training steps, and as closely as possible the number of unique student encodings.

This is the most important missing experiment for the proposed motivation.

### RQ3: How dependent is each method on batch size?

Evaluate batch sizes such as 16, 32, 64, and 128. Report:

- downstream average;
- fixed-pair similarity correlation;
- kNN overlap;
- wall-clock and peak memory.

Expected evidence: the batch-local baseline changes more strongly with batch size, while fixed graph-row training remains comparatively stable.

### RQ4: How dependent is each method on batch partition?

Keep initialization and optimization settings controlled while changing data shuffles or fixed partitions. Report mean and standard deviation over partitions.

This directly tests whether relational supervision is an accidental property of which examples co-occur.

### RQ5: Does GGPKD preserve a fixed corpus geometry?

Evaluate on a held-out set of corpus items or relations using metrics such as:

- Spearman correlation between teacher and student cosine similarities;
- teacher/student kNN overlap or recall;
- neighborhood rank correlation;
- graph-edge similarity error;
- trustworthiness/continuity if implementation is reliable.

Do not use teacher probability mass as the headline metric.

### RQ6: What does each method component contribute?

Ablations:

1. fixed graph-row loss only;
2. plus pool-level calibration;
3. plus auxiliary-row reuse;
4. raw top-$k$ versus mutual kNN;
5. fixed graph neighborhoods versus fixed random corpus neighborhoods;
6. different graph widths;
7. different auxiliary-row weights.

The key result should establish that the fixed corpus-row objective provides the main gain. Pool calibration and row reuse should be secondary improvements.

### RQ7: What is the computational trade-off?

Report:

- unique student texts encoded per update;
- relation constraints evaluated per update;
- constraints per student encoding;
- training time;
- peak GPU memory;
- offline graph construction time and storage;
- inference cost.

Candidate pooling should be justified by constraints-per-encoding rather than only by an informal “no extra encoding” statement.

## 11. Claim--evidence matrix

| Claim | Evidence required | Current status |
|---|---|---|
| Batch-defined normalized relations change with batch composition | Formal derivation plus small counterexample | Missing |
| GGPKD defines fixed anchor-row targets at corpus scope | Method definition of $G_T$, $\Omega_i$, and $\mathcal L_{r=1}$ | Present |
| Mini-batch row sampling estimates a fixed graph objective | Short expectation argument | Missing |
| Fixed corpus rows beat batch-local rows | Compute-matched ablation | Missing |
| GGPKD is less batch-size dependent | Batch-size sweep against batch-local baseline | Missing |
| GGPKD better preserves corpus geometry | Held-out geometry metrics | Missing |
| Auxiliary rows add constraints without extra encoder calls | Encoding-count analysis and ablation | Partially present |
| GGPKD improves compact students | Main downstream table | Present |
| Gains are strongest on STS | Per-task results across settings | Present, wording should remain calibrated |
| No online teacher or inference overhead | Cached teacher pipeline and student-only evaluation | Present |

## 12. Discussion and limitations flow

### Discussion

Return to the central principle:

- relational semantics should not be an accidental outcome of batch formation;
- sparse local rows are one practical representation of a corpus-level relational objective;
- candidate pooling separates semantic scope from memory constraints.

Clarify that the method is not opposed to mini-batch training. It relies on mini-batches, but restricts their role to stochastic execution.

### Limitations

State explicitly:

- exact graph construction has quadratic similarity cost, though approximate nearest-neighbor construction can replace it;
- the corpus graph is fixed and may become stale if the corpus changes;
- local graph alignment does not guarantee preservation of every long-range relation;
- graph width controls both neighborhood scale and the current row-bandwidth construction;
- shared candidate pools can substantially increase student encoding cost relative to a plain anchor batch;
- the ambient and partial auxiliary-row losses retain some dependence on the current pool;
- downstream gains are not uniform across all pair-classification tasks;
- results currently cover a roughly 15K-sentence training corpus, so larger-corpus scaling should be evaluated.

## 13. Conclusion flow

The conclusion should contain three sentences/moves:

1. Relational KD should learn a stable geometry whose definition is independent of mini-batch composition.
2. GGPKD realizes this using persistent sparse corpus rows and pooled mini-batch computation.
3. Empirical results show [verified downstream and geometry findings], while only the student is retained for deployment.

Suggested closing sentence:

> More broadly, our results suggest that mini-batching should be treated as a computational interface to relational learning, rather than as the boundary of the relational knowledge being learned.

## 14. Immediate LaTeX rewrite map

### `00_abstract.tex`

- Replace alignment/coverage and teacher-relevance motivation.
- Introduce batch-conditioned relational targets and scope--computation decoupling.
- Remove diffusion/multi-scale claims under the current method.
- Standardize the name to GGPKD.

### `01_introduction.tex`

- Remove the duplicated deployment-cost sentence.
- Replace the support-selection paragraphs with the seven-paragraph flow above.
- Add the normalized-row argument or defer its complete derivation to a problem-setup subsection.
- Replace the unfinished contribution list.

### `02_related_work.tex`

- Remove Sparse Logit Sampling as a central motivation.
- Organize work around batch-local versus corpus-level relation definitions.
- Add Cross-Batch Memory and possibly MoCo as adjacent precedents.
- Replace “RIPPLE” with “GGPKD.”

### `03_method.tex`

- Open with the two-stage corpus-definition/training-execution distinction.
- Make $\mathcal L_{r=1}$ the core objective.
- Describe $\mathcal L_{r=0}$ as a pool-conditioned calibration regularizer.
- Describe $\mathcal L_{\mathrm{row}}$ as computational reuse.
- Keep the 0.99 truncation as a numerical implementation detail.

### `04_experiments.tex`

- Add research questions before the main table.
- Add compute-matched batch-local versus fixed-row experiments.
- Add batch-size and batch-partition robustness.
- Add direct corpus-geometry metrics.
- Retain sensitivity analysis after mechanism and ablation results.

### `06_limitations_conclusion.tex`

- Add explicit limitations.
- Close on separation of objective scope and computation unit.

## 15. Final consistency checklist

Before submission, verify that:

- [ ] The paper never describes GGPKD as teacher-selected batching.
- [ ] “Teacher mass,” “mass-aware,” and alignment--coverage theory are absent from the main narrative.
- [ ] “Global” is defined as corpus scope, not dense all-pairs supervision.
- [ ] No multi-scale or diffusion claim remains unless implemented and evaluated.
- [ ] The mathematical claim is limited to batch-conditioned normalized objectives.
- [ ] $\mathcal L_{r=1}$ is clearly the core fixed-row objective.
- [ ] The batch dependence of $\mathcal L_{r=0}$ and partial auxiliary rows is acknowledged.
- [ ] The main experiment directly compares batch-local and corpus-defined rows at comparable compute.
- [ ] Claims about in-domain, out-of-domain, and task-family superiority match the table exactly.
- [ ] Method naming is consistently GGPKD throughout the paper.
- [ ] The contribution list contains no placeholder.
