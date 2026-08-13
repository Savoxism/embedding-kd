# Random-Hard Direct Distillation for BGE-M3 to MiniLM-H768

Date: 2026-08-13
Status: Implemented and deployed

## Goal

Add an ablation that replaces HeatGeo diffusion/random-walk candidates with fresh
uniform samples while retaining teacher-nearest hard negatives. Fine-tune
`BAAI/bge-m3` into
`nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large` using only the existing
direct teacher-distribution loss and Similarity Gauge Calibration (SGC).

The new run must not build a mutual-kNN transition matrix or execute a diffusion
step. The existing diffusion-based HeatGeo path remains available and unchanged
for comparison.

## Decisions

### Candidate composition

For every anchor and epoch, draw 64 distinct candidates:

- 32 uniform random candidates from the deduplicated training corpus;
- 24 candidates sampled without replacement from the anchor's precomputed
  hard-negative pool;
- 8 additional uniform random negatives.

The anchor itself and all duplicate indices are excluded. The 32 random-candidate
slots and 8 random-negative slots are statistically identical under the selected
direct-only objective, but remain distinct configuration quotas and diagnostics so
the ablation matches the requested `32 + 24 + 8` composition.

Internally, select the 24 hard rows first so uniform draws cannot consume their
quota. Return candidates in the documented `[32 random, 24 hard, 8 random-negative]`
layout; this makes quota validation and run diagnostics unambiguous even though the
loss is invariant to column order.

The sampler is seeded from `(global_seed, epoch, anchor_index)`. Repeating a sample
within the same epoch is deterministic, while changing the epoch redraws the set.

### Hard-negative definition

Hard-negative mining is separated from the diffusion graph builder. For each
anchor, rank corpus rows by BGE-M3 cosine similarity, excluding the anchor itself.
Fill the configured pool with nearest rows from the same source dataset first, then
use nearest cross-source rows if the same-source pool is exhausted.

The new builder only computes teacher top-k cosine neighbours. It does not call the
mutual-kNN transition, sparse transition matrix, lazy-walk, or diffusion-pool code.

### Objective

For a batch of size \(B\), let the scored column set be the deduplicated union of
all candidates drawn by anchors in that rank:

\[
\mathcal U = \bigcup_{i=1}^{B} C_i.
\]

The dense teacher and student distributions are

\[
q_{ij} =
\operatorname{softmax}_{j\in\mathcal U,\,j\ne i}
\left(\frac{\cos(t_i,t_j)}{\tau_T}\right),
\qquad
p_{ij} =
\operatorname{softmax}_{j\in\mathcal U,\,j\ne i}
\left(\frac{\cos(s_i,s_j)}{\tau_S}\right).
\]

The new mode optimizes

\[
\mathcal L =
\frac{1}{B}\sum_i D_{\mathrm{KL}}(q_i\Vert p_i)
+ \lambda_{\mathrm{SGC}}\frac{1}{B}\sum_i
\operatorname{Huber}\left(
\sum_j q_{ij}\cos(s_i,s_j)
- \sum_j q_{ij}\cos(t_i,t_j)
\right).
\]

Use \(\tau_T=\tau_S=0.10\), \(\lambda_{\mathrm{SGC}}=0.05\), and the existing
SGC Huber delta of 0.10. There is no local diffusion KL, no diffusion-scale
temperature, and no scale weight in this mode. The implementation must not emulate
an absent diffusion target with an all-zero `teacher_probs` tensor.

Candidate sharing remains rank-local under DDP, matching the existing HeatGeo
behavior. The single-GPU launch uses one DDP process, while the code path remains
valid for multiple processes.

## Configuration and compatibility

Add `candidate_sampling_mode` with these values:

- `diffusion`: current HeatGeo behavior and the default for backward compatibility;
- `random_hard_direct`: the new candidate and objective path.

Add a CLI override:

```text
--heatgeo_sampling_mode random_hard_direct
```

Add a dedicated `random_candidate_k = 32` configuration field. Do not repurpose
`diffusion_quota`; it continues to describe only the diffusion mode. The existing
`hard_neg_k = 24`, `random_neg_k = 8`, and `candidate_size = 64` values are shared.
Validate that the selected mode's quotas do not exceed `candidate_size`.

The diffusion builder, sampler, loss path, cache schema, and launcher task remain
usable without migration. Mode selection must be explicit in the new BGE run.

## Components

### Hard-negative artifact builder

Create a lightweight hard-negative artifact containing:

- `hard_neg_indices: LongTensor[N, hard_neg_pool]`;
- `source_ids: LongTensor[N]`;
- metadata with artifact version, mode, item count, hard-pool size, teacher
  fingerprint, and source fingerprint;
- summary diagnostics such as average/minimum pool fill and same-source fill.

Cache loading must validate all metadata. A teacher, corpus, source assignment, pool
size, or schema change forces a rebuild. The file contains no `pool_indices`,
`pool_probs`, diffusion scale, transition matrix, or walk statistic.

Only rank zero builds or writes the artifact. Other ranks wait at the existing
distributed barrier and then load the validated cache.

### Candidate sampler

Add a sampler dedicated to `random_hard_direct`. It consumes only the hard-negative
artifact and corpus size, returns candidate indices, and exposes `set_epoch`.
Sampling order is not semantically meaningful to the direct loss, but quota labels
must be retained in configuration and startup diagnostics.

If the corpus has enough eligible rows, return exactly 64 unique candidates. If it
does not, raise a descriptive error rather than silently duplicating candidates or
returning a short batch.

### Dataset and collate

In direct-only mode, the dataset item and collate output omit `teacher_probs`.
Candidate tokenization, deduplication, length sorting, chunking, and inverse mapping
remain the same as the current HeatGeo data path.

The training step passes `teacher_probs=None` to the criterion in direct-only mode.
The diffusion mode continues to require and stack the existing tensor.

### Criterion

Extend `HeatGeoDistillation` with an explicit `use_diffusion_loss` switch. In
direct-only mode it:

1. builds/deduplicates the shared candidate embedding pool without needing a
   diffusion target;
2. masks each anchor from its own teacher and student distributions;
3. computes direct KL and SGC only;
4. reports direct and SGC diagnostics without fabricated diffusion metrics.

`teacher_embeddings`, `candidate_idx`, and `anchor_idx` are mandatory for the new
mode. Missing inputs produce an early, descriptive `ValueError`. The diffusion
branch and its numerical behavior stay intact.

## Logging and output isolation

Add a launcher task named:

```text
bge_m3_to_minilmv2_h768_random_hard_direct
```

The launch log must include mode, quota composition, objective state
(`diffusion_loss=disabled`), physical GPU, free memory before launch, command, PID,
and timestamp. Use these server paths:

```text
/home/tensara/projects/ICLR-HeatGeo/models/bge_m3_to_minilmv2_h768_random_hard_direct/<timestamp>/
/home/tensara/projects/ICLR-HeatGeo/artifacts/bge_m3_to_minilmv2_h768_random_hard_direct/
/home/tensara/projects/ICLR-HeatGeo/logs/bge_m3_to_minilmv2_h768_random_hard_direct_<timestamp>.log
```

The artifact directory contains the BGE teacher embedding cache and
`hard_negative_pool.pt`. Existing BGE diffusion models, artifacts, and logs are not
overwritten.

At deployment time, inspect `nvidia-smi` and select a GPU with sufficient free
memory. Launch through the server project's `.venv/bin/torchrun` with
`--nproc_per_node=1`. Do not install packages globally.

## Tests

Automated coverage must verify:

1. the sampler returns the exact `32 + 24 + 8 = 64` quota when enough rows exist;
2. the anchor and duplicate candidates are absent;
3. sampling is repeatable within an epoch and changes across epochs;
4. same-source hard neighbours are preferred and cross-source fallback fills a
   short pool;
5. the hard-negative artifact has the expected minimal schema and cache
   invalidation behavior;
6. building the new artifact never invokes transition/random-walk functions;
7. direct-only KL and SGC are finite and backpropagate into student embeddings;
8. missing direct-only inputs fail with clear errors;
9. existing diffusion and SGC tests continue to pass;
10. the existing distributed smoke test continues to pass.

Run tests with the project virtual environment when present. No global `pip`
installation is permitted.

## Acceptance criteria

Implementation is complete when:

- all local targeted and regression tests pass;
- the new mode startup output confirms that no diffusion artifact is built and no
  diffusion loss is active;
- code is migrated to `H200_Tensara` without removing existing remote artifacts;
- a free GPU is selected and the timestamped BGE-M3 to MiniLM-H768 job starts from
  the project virtual environment;
- the PID remains alive through startup, the log advances into training with finite
  loss, and model/artifact/log paths match the isolated layout above;
- the five-epoch run completes successfully and leaves final student weights plus
  evaluation metrics in its timestamped model directory and log.

## Deployment result

Implemented in commit `b6f6f8a` and deployed to `H200_Tensara` on GPU 4 as run
`20260813-045123`. All 13 local and remote tests passed. The run completed five
epochs without non-finite loss or gradient skips, produced a hard-only artifact
with a full `200/200` hard pool for every anchor, and did not create a diffusion
graph.

Final epoch-5 validation averages were IOD 70.58, OOD 83.28, and overall 79.05.
Final published-test averages were IOD 69.46, OOD 79.93, and overall 76.44. Durable
outputs are stored under the timestamped `models` directory, with the teacher cache
and hard-negative artifact under the task's `artifacts` directory and the complete
timestamped run log under `logs`.
