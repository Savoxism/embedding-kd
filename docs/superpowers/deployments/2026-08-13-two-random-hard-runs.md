# Two Remaining Random-Hard Runs Deployment

Date: 2026-08-13
Run timestamp: `20260813-054627`
Server: `H200_Tensara`
Project: `/home/tensara/projects/ICLR-HeatGeo`

## Result

Both approved `random_hard_direct` runs completed five epochs and final-test
evaluation. Each task used one `torchrun` process on a distinct physical H200:

| Pair | GPU | Epoch-5 validation IOD / OOD / ALL | Final test IOD / OOD / ALL |
|---|---:|---:|---:|
| Qwen3-Embedding-4B to BERT-base | 4 | 73.38 / 83.76 / 80.30 | 71.42 / 80.18 / 77.26 |
| Qwen3-Embedding-0.6B to MiniLMv2-H384 | 2 | 69.02 / 80.56 / 76.71 | 66.25 / 77.13 / 73.51 |

Compared with the original diffusion HeatGeo final tests, the random-hard run
changed overall score by -1.04 points for Qwen3-4B to BERT-base and -2.13 points
for Qwen3-0.6B to MiniLMv2-H384.

## Save-policy verification

Each run contains exactly the requested periodic/final durable files:

```text
best_model.pt
checkpoint_epoch_3.pt
checkpoint_epoch_5.pt
weights/student_epoch_3.pt
weights/student_epoch_5.pt
```

There are no epoch 1, 2, or 4 checkpoint/weight files.

## Outputs

Qwen3-4B to BERT-base:

```text
models/qwen3_4b_to_bert_base_random_hard_direct/20260813-054627/
artifacts/qwen3_4b_to_bert_base_random_hard_direct/
logs/qwen3_4b_to_bert_base_random_hard_direct_20260813-054627.log
```

Qwen3-0.6B to MiniLMv2-H384:

```text
models/qwen3_0_6b_to_minilmv2_h384_random_hard_direct/20260813-054627/
artifacts/qwen3_0_6b_to_minilmv2_h384_random_hard_direct/
logs/qwen3_0_6b_to_minilmv2_h384_random_hard_direct_20260813-054627.log
```

SHA-256:

| File | SHA-256 |
|---|---|
| 4B final student weight | `35298b62fd13019c95bc16d5f2e0a005fbb718f1f343b2bc15176530e6a22840` |
| 4B hard-negative pool | `26a425e596a204d1738e68a495cb057cc64ee3b15bbe808605ae076b9fc9d46c` |
| 0.6B final student weight | `d072260b84565f52f78981a9c32ce58defb13450ccf734035d41017914e669e0` |
| 0.6B hard-negative pool | `196bb47731be84f925a54e7f59adb945697af71609f0361a06dd2a4681f8aa7a` |

## Verification

- Local and remote test suites: 16 passed.
- Python compilation and launcher shell syntax: passed.
- Two-process DDP smoke test: passed on CPU locally and CUDA remotely.
- Both logs state `random_hard_direct`, `save_every=3`, and diffusion loss disabled.
- Both metrics files contain five validation records plus one final-test record.
- Error/non-finite scans are clean.
- Physical GPUs 2 and 4 returned to idle state after their workers exited.
