# Two Remaining Random-Hard Training Runs

Date: 2026-08-13
Status: Approved design; implementation pending

## Goal

Run the approved `random_hard_direct` ablation for the two remaining teacher and
student pairs on two idle H200 GPUs:

- `Qwen/Qwen3-Embedding-4B` to `google-bert/bert-base-uncased`;
- `Qwen/Qwen3-Embedding-0.6B` to
  `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large`.

Each run uses five epochs and saves periodic state only after epoch 3, followed by
the mandatory final checkpoint and student weights after epoch 5.

## Training behavior

Both runs use the already implemented random-hard direct mode:

- 32 uniform random candidates;
- 24 teacher-nearest hard candidates;
- 8 additional uniform random negatives;
- direct teacher-distribution KL plus SGC;
- no mutual-kNN transition, diffusion pool, random walk, or diffusion loss.

All other training parameters match the completed BGE random-hard run: batch size
32, learning rate `3e-5`, maximum length 256, SGC weight 0.05, and one `torchrun`
process per GPU. The Qwen teachers retain their existing `last_token` pooling.

## Save policy

Expose the existing `save_every` configuration through a CLI argument and launch
both tasks with:

```text
--epochs 5 --save_every 3
```

The current train lifecycle already performs an unconditional final save after the
epoch loop. Therefore the expected durable files are exactly:

```text
checkpoint_epoch_3.pt
checkpoint_epoch_5.pt
best_model.pt
weights/student_epoch_3.pt
weights/student_epoch_5.pt
```

No epoch 1, 2, or 4 checkpoint/weight file is expected. The final epoch-5 save is
mandatory even though 5 is not divisible by 3.

Reject non-positive `--save_every` values before training begins.

## Launcher and GPU allocation

Add two isolated launcher task names:

```text
qwen3_4b_to_bert_base_random_hard_direct
qwen3_0_6b_to_minilmv2_h384_random_hard_direct
```

The launcher passes `random_hard_direct`, `hard_negative_pool.pt`, and
`--save_every 3` to these tasks only. Existing task names and default save behavior
remain unchanged.

Immediately before launch, re-query all GPUs. A selected GPU must have at least
100 GB free and no active compute workload. Assign two distinct eligible GPUs in a
single snapshot so both jobs launch together or neither starts. At design time,
GPU 4 and GPU 6 satisfy these requirements; their eligibility must be revalidated
at launch rather than assumed.

## Cache and output isolation

Reuse the completed diffusion runs' teacher embedding caches because the teacher,
pooling, deduplicated corpus, and row order are unchanged. Copy each cache into its
new task artifact directory and verify source/destination checksums before launch.
Each new task then builds its own fingerprinted `hard_negative_pool.pt`.

Use these server roots:

```text
/home/tensara/projects/ICLR-HeatGeo/models/qwen3_4b_to_bert_base_random_hard_direct/<timestamp>/
/home/tensara/projects/ICLR-HeatGeo/artifacts/qwen3_4b_to_bert_base_random_hard_direct/
/home/tensara/projects/ICLR-HeatGeo/logs/qwen3_4b_to_bert_base_random_hard_direct_<timestamp>.log

/home/tensara/projects/ICLR-HeatGeo/models/qwen3_0_6b_to_minilmv2_h384_random_hard_direct/<timestamp>/
/home/tensara/projects/ICLR-HeatGeo/artifacts/qwen3_0_6b_to_minilmv2_h384_random_hard_direct/
/home/tensara/projects/ICLR-HeatGeo/logs/qwen3_0_6b_to_minilmv2_h384_random_hard_direct_<timestamp>.log
```

Existing diffusion and BGE random-hard outputs must not be overwritten.

## Verification and monitoring

Add tests for CLI override and the save schedule predicate. Run the full local and
remote test suites, Python compilation, shell syntax validation, and the existing
DDP smoke test before launch.

For each remote run, verify:

- startup log states `sampling_mode=random_hard_direct`,
  `diffusion_loss=disabled`, `save_every=3`, and the physical GPU;
- hard-negative artifact fills its configured pool and no diffusion graph appears;
- loss and gradients stay finite;
- only epoch-3 and epoch-5 checkpoints and student weights are saved;
- validation runs after every epoch and the final published-test evaluation
  completes;
- worker processes exit and their GPUs return to the idle state.

Report epoch-5 validation, final-test IOD/OOD/overall averages, output paths, and
checksums for the final student weights and hard-negative artifacts.
