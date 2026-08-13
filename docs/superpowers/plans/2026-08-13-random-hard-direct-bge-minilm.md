# Random-Hard Direct BGE-M3 → MiniLM-H768 Implementation Plan

## Scope and assumptions

- Preserve `candidate_sampling_mode="diffusion"` as the default and keep its
  outputs numerically unchanged.
- Add only the approved `random_hard_direct` mode.
- Use the checked-in project `.venv`; install nothing globally.
- Reuse the current DDP coordination pattern: rank zero builds, barrier, other
  ranks load.
- The server run is complete only after five epochs and final evaluation finish.

## Task 1: Lock the new behavior with tests

Files:

- Add `tests/test_random_hard_direct.py`.
- Extend `tests/test_heatgeo_sgc.py` only for direct-only criterion behavior.

Steps:

1. Add sampler tests for a deterministic 32/24/8 layout, uniqueness, anchor
   exclusion, stable same-epoch sampling, and changed cross-epoch sampling.
2. Add hard-pool builder tests for same-source priority and cross-source fallback.
3. Add an artifact-schema test that rejects stale metadata and proves the new
   builder does not call diffusion helpers.
4. Add direct-only loss tests for finite forward/backward and required inputs.
5. Run the new tests first and confirm they fail for missing functionality.

Verification:

```bash
.venv/bin/python -m pytest -q tests/test_random_hard_direct.py tests/test_heatgeo_sgc.py
```

## Task 2: Build the lightweight hard-negative artifact

Files:

- Add `src/heatgeo/hard_negative_builder.py`.
- Update `src/heatgeo/__init__.py`.

Steps:

1. Implement chunked teacher-cosine ranking that selects same-source nearest
   neighbours before cross-source fallback.
2. Add fingerprinted metadata and exact cache validation.
3. Persist only hard indices, source ids, summary stats, and metadata.
4. Print a concise pool-fill summary; do not create transition or walk data.

Verification:

- Builder unit tests pass.
- Saved artifact has no `pool_indices`, `pool_probs`, or diffusion metadata.

## Task 3: Add the random-hard sampler and optional teacher targets

Files:

- Update `src/heatgeo/candidate_sampler.py`.
- Update `src/data_utils/dataset_cache.py`.

Steps:

1. Add `RandomHardDirectCandidateSampler`.
2. Internally select 24 hard candidates first, then draw 32 and 8 uniform rows
   without replacement; return the documented 32/24/8 layout.
3. Raise if the corpus or hard pool cannot fill the requested unique quota.
4. Let the HeatGeo dataset/collate omit `teacher_probs` when the sampler returns no
   diffusion target, while leaving the current tensor path unchanged.

Verification:

- Sampler/data-path tests pass.
- Existing diffusion sampler tests/behavior remain intact.

## Task 4: Add the direct-only criterion branch

Files:

- Update `src/criterions/heatgeo_distillation.py`.
- Update `distiller.py`.

Steps:

1. Add `use_diffusion_loss=True` to preserve the current default.
2. In direct-only mode, build the shared embedding/index pool without a target
   tensor, then compute direct KL and SGC.
3. Require teacher bank, candidate indices, and anchor indices with descriptive
   errors.
4. Keep diffusion forward logic unchanged behind its branch.
5. Make training transfer/pass `teacher_probs` conditionally.
6. Emit mode-specific diagnostics without fake diffusion scales.

Verification:

- Direct-only loss is finite and differentiable.
- Existing `tests/test_heatgeo_sgc.py` expectations still pass.

## Task 5: Wire configuration, CLI, and launcher

Files:

- Update `config/heatgeo_config.py`.
- Update `main.py`.
- Update `scripts/launch_h200_jobs.sh`.

Steps:

1. Add default `candidate_sampling_mode="diffusion"` and
   `random_candidate_k=32`.
2. Add `--heatgeo_sampling_mode` CLI choices.
3. In `distiller.py`, choose the diffusion artifact/sampler or hard-only
   artifact/sampler from the mode.
4. Add isolated launcher task
   `bge_m3_to_minilmv2_h768_random_hard_direct` with `hard_negative_pool.pt` and
   timestamped model/log directories.
5. Allow the new task GPU to be supplied at launch after checking live free memory.

Verification:

```bash
bash -n scripts/launch_h200_jobs.sh
.venv/bin/python main.py --help | rg heatgeo_sampling_mode
```

## Task 6: Regression and smoke verification

Steps:

1. Run targeted tests.
2. Run the full test suite.
3. Run syntax compilation for changed Python files.
4. Run the distributed smoke script through project `torchrun` where supported.
5. Review `git diff --check` and the final diff for unrelated changes.

Verification:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src config main.py distiller.py
.venv/bin/torchrun --standalone --nproc_per_node=2 scripts/distributed_smoke.py
git diff --check
```

## Task 7: Migrate and finish the H200 run

Steps:

1. Confirm SSH alias `H200_Tensara`, remote project path, remote `.venv`, current
   code revision, and live GPU memory/processes.
2. Sync only the approved code/tests/config/scripts/docs changes, preserving remote
   models, artifacts, logs, datasets, and `.venv`.
3. Run remote targeted tests with the remote project `.venv`.
4. Select a free GPU meeting the launcher's memory threshold.
5. Launch only the new BGE task and record manifest, PID, GPU, model, artifact, and
   log paths.
6. Monitor artifact creation, finite loss, epoch checkpoints, evaluation, and
   process health until all five epochs complete.
7. Report final IOD/OOD/average metrics and durable output paths.

Verification:

- Remote tests pass.
- Log states `sampling_mode=random_hard_direct` and
  `diffusion_loss=disabled`.
- `hard_negative_pool.pt` exists and no new HeatGeo diffusion graph is created for
  the task.
- Final weights and evaluation are present under the timestamped model directory.
