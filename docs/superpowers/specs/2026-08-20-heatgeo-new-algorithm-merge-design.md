# HeatGeo New-Algorithm Merge Design

## Objective

Resolve the incomplete merge at `95cecfc` by making the new HeatGeo algorithm the
only executable HeatGeo path. The result must not accept or silently emulate the
removed fixed-capacity and fixed-bandwidth APIs.

## Selected approach

Use the algorithm-complete implementation from `65aeae2` as the contract for
configuration, CLI, distiller, and criterion, while retaining the later
non-backtracking walk implementation and graph diagnostics already present at HEAD.
This preserves newer compatible work without keeping a hybrid of old and new
semantics.

Rejected alternatives:

- Reset the whole repository to `65aeae2`: this would discard later documentation,
  diagnostics, tests, and the non-backtracking walk probe.
- Keep compatibility aliases for old knobs: this would allow two objective
  definitions and make experiment provenance ambiguous.

## Algorithm contract

### Per-row entropic affinity

For each graph row `i`, solve a positive temperature `tau_i` such that

```text
H(P_i) = log(perplexity),
P_ij proportional to exp(sim_ij / tau_i).
```

The default perplexity is 30. A non-positive CLI perplexity selects the explicit
fixed-bandwidth baseline, where every row uses `graph_temp`. The graph artifact must
store `row_temps`, and cache metadata must distinguish the two modes.

### Temperature use in the criterion

The sharpest diffusion row and every walk-supervised row use the corresponding
artifact `row_temps`. Broader scales keep the explicit student temperature ladder
defined by `broad_scale_temps`; the ambient/direct target and student distribution
share `direct_temp`.

### Mass-bounded truncation

Replace `pool_size`, `walk_keep_topk`, and `walk_topk` with one build-time
`truncation_tolerance`, default 0.01. Every truncated probability row keeps the
smallest prefix carrying at least `1 - tolerance` mass. Allocation ceilings remain
implementation guards and must be reported when they bind.

### Non-backtracking walks

Enable `walk_non_backtracking` by default. A step may not immediately return to the
previous node unless the current node has no other positive-probability outgoing
edge. This changes the row-sampling measure only; targets remain complete artifact
transition rows.

### Objective and scale weights

Use the explicit scale weights and temperature settings from the new algorithm
contract in `65aeae2`. Remove `temp_exponent` and `weight_exponent` from the HeatGeo
configuration and CLI so the exponent-law branch cannot be mixed into this method.
The objective remains

```text
L = L_diff + walk_weight * L_walk.
```

## Data flow

1. Load or compute teacher embeddings.
2. Build graph transition rows with per-row entropic affinities.
3. Build diffusion targets using tolerance-bounded mass prefixes.
4. Pass transition rows and `row_temps` to the criterion.
5. Sample candidates and non-backtracking walk trajectories from the same artifact.
6. Match anchor diffusion rows and visited transition rows with their correct
   temperatures.

## Failure behavior

- Unknown `HeatGeoConfig` overrides raise immediately.
- Old knobs are absent from CLI and configuration.
- Artifacts without `row_temps`, or with stale metadata/version, rebuild rather than
  being reused as the new algorithm.
- Invalid perplexity, tolerance, row temperatures, or transition shapes raise with
  actionable errors.

## Verification

- Existing mathematical tests for entropic affinity, affine invariance, monotonic
  entropy, clamping, and mass-prefix bounds must pass.
- CLI tests must cover `perplexity`, `truncation_tolerance`, and
  `walk_non_backtracking`.
- Criterion tests must show that different row temperatures produce different
  student entropies and are gathered by anchor/walk row index.
- Non-backtracking tests must cover ordinary cycles, degree-one fallback, and seed
  reproducibility.
- A smoke test must instantiate the full graph-builder -> sampler -> criterion path
  without legacy keyword errors.

## Out of scope

- Re-running GPU experiments.
- Changing the three teacher-student model pairs.
- Tuning perplexity, tolerance, walk weight, or training hyperparameters.
- Migrating the repaired code to the server before local verification succeeds.
