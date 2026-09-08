# GGPKD — Method Specification

Relational knowledge distillation from an embedding teacher to a small student,
supervised by the teacher's own kNN transition rows.

The method has **two hyperparameters**: `graph_k` and `row_weight`. Every other
quantity — bandwidths, temperatures, term weights, the candidate set, the set of
supervised rows — is derived from those two and the corpus.

The default radius is one hop, so the objective matches the teacher's transition
rows directly; multi-hop diffusion is a baseline arm, not part of the method.

---

## 1. Offline: the teacher graph

Built once per (teacher, corpus) and cached.

**1.1 Encode.** The teacher embeds the deduplicated corpus; vectors are
L2-normalized, so all similarities below are cosines.

**1.2 Retrieve.** For each node `i`, the `k = graph_k` highest-cosine other nodes.
Self is excluded during retrieval.

**1.3 Bandwidth.** Each row reads its temperature off the retrieval width:

$$\tau_i = \frac{s_i^{(1)} - s_i^{(k)}}{\log k}$$

where $s_i^{(j)}$ is the $j$-th largest cosine from `i`. Reading it back: the
$k$-th retrieved neighbour sits $\log k$ nats below the nearest, so it is exactly
$k$ times less likely — for every node, with no constant left to choose.

*Affine invariance.* Under $s \mapsto as + b$ the bandwidth scales as
$\tau \mapsto a\tau$, so the logits become $s_j/\tau_i + b/(a\tau_i)$. The second
term does not depend on $j$ and a softmax is shift-invariant, so **the row is
unchanged**. This is the property that rules out a single global temperature,
which would otherwise have to be retuned for every teacher whose cosines are
spread differently.

*Fixed sample size.* The scores come from the raw top-$k$, read **before** the
mutual filter, so all $k$ values exist for every node whenever $k < n$. There is
no degree to fall short of and no target entropy to miss.

**1.4 Mutual filter.** Keep the edge `i→j` only if `j ∈ topk(i)` **and**
`i ∈ topk(j)`. A hub that everything retrieves but that retrieves nothing back
loses those edges. Nodes left isolated fall back to their raw top-$k$.

**1.5 Transition row.** Over the surviving neighbours,

$$P(j \mid i) = \operatorname{softmax}_j\!\left(s_{ij} / \tau_i\right)$$

**1.6 Truncate.** Each row keeps enough columns that the discarded tail holds at
most 1% of its mass. This is a numerical-fidelity constant, not a method choice:
any value small enough that the discarded tail cannot change a ranking gives the
same objective, and the build reports the residual mass when it binds.

Row lengths are therefore **ragged** — on the production corpus (13,553 texts,
`graph_k = 200`): mean 66.8 columns, min 1, max 181, average degree 80.9, 0.08%
fallback nodes.

The artifact stores the truncated rows and their bandwidths.

---

## 2. Training: the candidate set

For each anchor, the candidate set is **its whole truncated transition row** —
every column the teacher put mass on, and nothing else. No negatives are drawn.

There is no budget, no selection and no RNG. The set is a deterministic function
of the graph and is **identical in every epoch**. For collation, rows are padded
to a common width with the anchor's own index, which the self-mask removes from
every softmax, so a short row is simply supervised on a narrower support.

Each step encodes the deduplicated union of the batch's candidate sets — the
**shared pool**, ~1,400 texts for a batch of 64. This is essentially the whole
step cost.

---

## 3. Objective

$$\mathcal{L} = \underbrace{\mathcal{L}_{r=0} + \mathcal{L}_{r=1}}_{\mathcal{L}_{\text{rel}}} + \lambda_{\text{row}}\,\mathcal{L}_{\text{row}}$$

### 3.1 Graph scale, `r=1`

For each anchor, KL between the teacher's transition row and the student's
softmax, over **the anchor's own candidate columns**, at $\tau_i$ on both sides.
Because the `r=1` target *is* the transition row, the student reuses the
bandwidth stored with that row — there is no temperature to choose.

### 3.2 Ambient scale, `r=0`

For each anchor, KL over the **whole shared pool** at a single temperature,
applied to teacher and student alike (same-temperature distillation, Hinton et
al. 2015). That temperature is derived as the median bandwidth of the graph.

The two groups deliberately use **different column sets**: `r=1` ranks *within*
the neighbourhood, `r=0` calibrates similarity levels *across* the batch. Neither
holds an opinion the other contradicts. The ambient group carries the same total
weight as the graph group.

### 3.3 Row supervision

Non-anchor nodes `j` already in the shared pool are promoted to auxiliary rows:

$$\mathcal{L}_{\text{row}} = \sum_j \nu_B(j)\, \mathrm{KL}\!\left(P^T_j|_{\Omega_j} \,\|\, p^S_j|_{\Omega_j}\right), \qquad \Omega_j = \{\text{teacher neighbours of } j\} \cap \text{pool}$$

with the dense transition row as target — row-kernel matching, not a trajectory
likelihood — each row at its own stored bandwidth $\tau_j$, weighted uniformly.
Batch anchors are excluded: $\mathcal{L}_{\text{rel}}$ already matches their row
at `r=1`. Rows with fewer than two available columns are dropped.

The row set is a deterministic function of the pool, so this term carries **no
selection hyperparameter** and reuses computation $\mathcal{L}_{\text{rel}}$ has
already paid for.

### 3.4 Temperature ties

None of these is a free parameter; the criterion rejects them by name.

| | tied to |
|:---|:---|
| $\tau_1(i)$ | $\tau_i$ — the `r=1` target *is* the transition row |
| $\tau_{\text{row}}(j)$ | $\tau_j$ — row targets are transition rows |
| ambient | one temperature on both sides, the median $\tau_i$ |
| ambient weight | = total graph-group weight |

---

## 4. Hyperparameters

| | default | role |
|:---|---:|:---|
| `graph_k` | 200 | retrieval width **and**, through $\tau_i$, row sharpness |
| `row_weight` | 1.0 | $\lambda_{\text{row}}$; 1.0 reads as an unweighted sum of two KLs in nats |

Plus standard training settings (batch size, epochs, learning rate, seed).

> `graph_k` does two jobs. A sweep over it cannot separate neighbourhood width
> from row sharpness — that is the price of not carrying a second constant, and it
> must be stated rather than hidden. Too large a `k` measures the distance out of
> the anchor's neighbourhood rather than the local decay, and the rows go uniform;
> `scripts/ggpkd/pick_graph_k.py` reports the induced sharpness per `k` from one
> teacher-embedding pass.