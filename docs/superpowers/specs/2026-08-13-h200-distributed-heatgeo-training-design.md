# H200 Distributed HeatGeo Training Design

## Goal

Migrate the current `main` tree to the SSH host `H200_Tensara`, make the
training path compatible with `torchrun`/DistributedDataParallel (DDP), and
launch three independent HeatGeo fine-tuning jobs. Each job currently owns one
physical H200, while the code must also support assigning multiple GPUs to one
model pair later.

## Server Layout

The deployment root is:

```text
/home/tensara/projects/ICLR-HeatGeo/
├── .venv/
├── models/
│   ├── qwen3_4b_to_bert_base/<timestamp>/
│   ├── bge_m3_to_minilmv2_h768/<timestamp>/
│   └── qwen3_0_6b_to_minilmv2_h384/<timestamp>/
├── artifacts/
│   ├── qwen3_4b_to_bert_base/
│   ├── bge_m3_to_minilmv2_h768/
│   └── qwen3_0_6b_to_minilmv2_h384/
└── logs/
```

Each artifact directory contains the teacher-embedding cache, HeatGeo graph
artifact, and graph diagnostics for that model pair. `models/` contains local
checkpoints, per-epoch student weights, and JSONL metrics. No output is written
outside the deployment root except the existing Hugging Face download cache.

Each log is named:

```text
<task>_<YYYYMMDD-HHMMSS>.log
```

## Jobs

| Task | Physical GPU | Teacher | Student | Teacher pooling |
|---|---:|---|---|---|
| `qwen3_4b_to_bert_base` | 1 | `Qwen/Qwen3-Embedding-4B` | `google-bert/bert-base-uncased` | last token |
| `bge_m3_to_minilmv2_h768` | 2 | `BAAI/bge-m3` | `nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large` | CLS |
| `qwen3_0_6b_to_minilmv2_h384` | 6 | `Qwen/Qwen3-Embedding-0.6B` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large` | last token |

GPU 6 replaces the originally requested GPU 3 because GPU 3 is occupied by a
VLLM process using approximately 132 GiB. The launcher rechecks available VRAM
immediately before starting every job.

All jobs use the current HeatGeo/SGC objective with:

- 5 epochs;
- per-process batch size 32;
- learning rate `3e-5`;
- maximum sequence length 256;
- W&B disabled so server files are the authoritative logs.

## Distributed Architecture

`torchrun` owns rank assignment. `KnowledgeDistiller` initializes a process
group only when the `RANK`/`WORLD_SIZE` environment contract is present, binds
the process to `cuda:LOCAL_RANK`, and destroys the group on exit.

For `WORLD_SIZE > 1`:

1. the student is wrapped in DDP;
2. a `DistributedSampler` partitions the training corpus and receives
   `set_epoch(epoch)` before each epoch;
3. rank 0 builds the teacher and graph caches, then all other ranks load them
   after a barrier;
4. rank 0 alone evaluates, writes JSONL, and saves checkpoints/weights;
5. barriers keep ranks from entering the next training epoch while rank 0 is
   evaluating or saving.

The current launch uses `--nproc_per_node=1` per task. This exercises the
`torchrun` device/rank path while avoiding cross-model coupling. A future
multi-GPU task can expose several devices and raise `--nproc_per_node` without
changing the Python command.

Checkpoint code always unwraps DDP before reading the state dictionary, so
saved weights remain compatible with the existing single-process loader and do
not contain a `module.` prefix.

## Migration and Runtime

Code and data are copied with `rsync`. Git internals, local virtual environments,
model outputs, artifacts, caches, logs, and bytecode are excluded. The server
uses a project-local `.venv`; no package is installed into global Python.

One launcher script defines all model-pair-specific values, creates the output
directories, checks GPU free memory, and starts each task with `nohup` plus
`torchrun`. A manifest records task name, physical GPU, PID, log path, model
path, and artifact path.

## Failure Handling

- The launcher refuses a GPU with less than 100 GiB free, rather than silently
  oversubscribing another user's process.
- Existing timestamped logs are never overwritten.
- Model and artifact directories are task-specific, preventing cache reuse
  across teachers, students, or pooling methods.
- Rank-zero failures propagate through `torchrun`; each task is an independent
  OS process, so one failed pair does not terminate the other pairs.
- Existing non-empty model/artifact directories are retained. A new training
  launch appends metrics but writes epoch checkpoints using the repository's
  existing names; therefore the launcher creates a timestamped run directory
  beneath each model-pair directory for every launch.

## Verification

Before launch:

1. local unit tests and Python compilation pass;
2. a CPU/single-process distributed test verifies environment initialization,
   rank-zero gating, and DDP state-dict unwrapping;
3. the migrated tree matches the local source under the rsync exclusions;
4. the project virtual environment imports PyTorch/Transformers and sees CUDA;
5. a one-process `torchrun` smoke check succeeds;
6. GPUs 1, 2, and 6 each have at least 100 GiB free.

After launch, each job must have a live PID, a growing timestamped log, the
expected `CUDA_VISIBLE_DEVICES` assignment, and its own model/artifact paths.

## Out of Scope

- Changing the HeatGeo objective or benchmark protocol.
- Selecting a different epoch count, batch size, or model pair.
- Killing or moving processes already occupying GPU 3.
- Pushing local commits to GitHub.
