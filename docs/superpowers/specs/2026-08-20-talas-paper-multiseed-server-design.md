# TALAS Paper-Pair Multi-Seed Server Run Design

## Objective

Make the TALAS implementation reproducible for the three teacher-student pairs
reported in `docs/TALAS.pdf`, run every pair with seeds 42, 43, and 44 on the
`H200_Tensara` server, and report final epoch-5 test metrics as mean plus or
minus sample standard deviation.

The canonical training corpus remains
`data/train_set/merged_3_data_5k_each.csv`.

## Canonical Experiment Matrix

The implementation and launch scripts shall expose these exact presets:

| Pair key | Teacher | Student |
|---|---|---|
| `qwen3_0_6b_to_minilmv2_h384` | `Qwen/Qwen3-Embedding-0.6B` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large` |
| `bge_m3_to_minilmv2_h768` | `BAAI/bge-m3` | `nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large` |
| `qwen3_4b_to_bert_base` | `Qwen/Qwen3-Embedding-4B` | `google-bert/bert-base-uncased` |

Every pair runs with seeds `42`, `43`, and `44`, giving nine training runs.

All runs use the paper schedule:

- five epochs;
- batch size 32;
- learning rate `2e-5`;
- cosine learning-rate schedule;
- maximum sequence length 256;
- final epoch-5 checkpoint for test evaluation.

## TALAS Optimizer Alignment

The student encoder and every teacher-anchored linear projection shall use the
same learning rate:

$$
\eta_{\mathrm{student}}
=
\eta_{\mathrm{projection}}
=
2\times 10^{-5}.
$$

Projection parameters remain part of the same adaptive sharpness-aware
optimization problem as the student parameters. The current five-times
projection learning-rate multiplier shall be removed. No new projection-only
hyperparameter is introduced.

## Single-Run Launchers

### POSIX launcher

`scripts/train_talas.sh` shall:

1. derive the repository root from the script location rather than the current
   working directory;
2. require and use `<repo>/.venv/bin/python` by default;
3. accept a canonical pair key and seed through arguments or documented
   environment variables;
4. map the pair key to the exact teacher and student IDs above;
5. use the canonical training corpus;
6. use pair-isolated teacher caches and pair-plus-seed-isolated output paths;
7. pass through optional extra CLI arguments without changing paper defaults;
8. fail before model loading when the pair key, virtual environment, training
   data, or required output settings are invalid.

The launcher must work when invoked from the repository root, from `scripts/`,
or through an absolute path from another directory.

### PowerShell launcher

`scripts/train_talas.ps1` shall implement the same pair mapping, root
resolution, defaults, validation, and output naming for Windows PowerShell.
It shall use `<repo>/.venv/Scripts/python.exe` by default and allow an explicit
Python override for environments whose virtual environment lives elsewhere.

## Cache Preparation

Teacher embeddings are deterministic across training seeds, so each pair uses
one shared, pair-specific cache. The nine training jobs must not race to create
the same cache.

The orchestration workflow therefore has two phases:

1. prepare and validate the three caches, one per pair;
2. launch all nine seed runs read-only against those completed caches.

The cache preparation entry point shall terminate after data/model setup and
cache validation without entering a training epoch. A training run with an
existing cache shall not perform teacher inference. Cache file names shall
include the pair key so different teachers cannot collide by path.

At minimum, cache validation requires a readable tensor with the same row count
as the canonical training frame and a finite floating-point embedding for every
row. A failed or partial cache preparation stops the controller before any of
the nine training jobs begin.

## Server Orchestration

The target is SSH host `H200_Tensara`, with the deployed project at
`/home/tensara/projects/ICLR-HeatGeo` and its existing virtual environment at
`/home/tensara/projects/ICLR-HeatGeo/.venv`.

A dedicated TALAS paper-run controller shall:

1. create a timestamped run ID;
2. record a manifest containing pair, seed, GPU, model IDs, PID, and state;
3. prepare the three shared caches on separate GPUs;
4. schedule the nine runs over GPUs 0 through 7 with at most one training
   process per GPU;
5. queue the ninth run until a GPU becomes free;
6. run with Hugging Face offline mode after confirming all model snapshots are
   present, avoiding indefinite network waits;
7. write one log and one exit-code file per pair and seed;
8. run inside a named `tmux` session so SSH disconnects do not stop training;
9. return a non-zero controller status if cache preparation, any training job,
   final-test evaluation, output validation, or aggregation fails.

Each run saves only the final student weight payload, raw epoch metrics, final
test metrics, and logs. Full per-epoch optimizer checkpoints are excluded to
limit disk use on the server.

## Migration

Migration uses `rsync` from the local workspace to the existing server project.
It shall copy source, configs, launchers, tests, and the canonical data while
preserving server-owned runtime and experiment state.

The migration excludes:

- `.git/`;
- `.venv/`;
- Python and pytest caches;
- macOS `._*` and `.DS_Store` metadata;
- Hugging Face caches;
- existing `models/`, `results/`, `logs/`, `artifacts/`, and teacher caches.

Local verification must pass before migration. The same focused and full tests
must pass again with the server virtual environment before cache preparation or
training begins.

## Result Aggregation

For each seed, the aggregator reads the final test record produced after epoch
5. It computes benchmark metrics and the derived aggregates:

- `Avg In`: Emotion, WiC, and STS-B;
- `Avg Out`: Banking77, Tweet, MRPC, SciTail, SICK, and STS12;
- `Avg All`: all nine benchmarks.

For each pair and metric, with three seed values $x_1,x_2,x_3$, report

$$
\bar{x}=\frac{1}{3}\sum_{i=1}^{3}x_i,
\qquad
s=\sqrt{\frac{1}{2}\sum_{i=1}^{3}(x_i-\bar{x})^2}.
$$

The displayed form is `mean ± std`, in percentage points. The standard
deviation is the sample standard deviation (`ddof=1`).

The aggregator shall emit:

- a terminal table;
- a machine-readable TSV containing raw seed values, mean, and standard
  deviation;
- a Markdown summary containing the configuration, per-pair tables, run ID,
  server, model IDs, seeds, and output locations.

Aggregation fails if a pair is missing any requested seed, a run has a non-zero
exit code, a final test record is absent or duplicated, a benchmark is missing,
or any metric is non-finite.

## Testing and Verification

Local tests shall cover:

1. exact pair-key to model-ID mappings;
2. the TALAS default pair being a paper pair;
3. equal student and projection learning rates;
4. shell launcher root resolution and validation without loading models;
5. PowerShell mapping and path logic through a static or parameter-level test
   available on the development platform;
6. cache validation for valid, truncated, wrong-row-count, and non-finite
   tensors;
7. aggregation of three synthetic seed records, including the `ddof=1`
   calculation;
8. aggregation failure on missing, duplicated, failed, or non-finite runs;
9. a CPU TALAS smoke step showing a finite loss and a student update;
10. the complete existing test suite and Python compilation.

Server preflight shall verify:

- eight visible NVIDIA H200 GPUs;
- the project virtual environment and required imports;
- all six canonical model snapshots are available locally;
- sufficient free disk space for nine final student weight files, metrics, and
  logs;
- the migrated test suite passes;
- no existing result directory is overwritten.

Final completion requires all nine runs to exit successfully, exactly one final
student weight file per run, exactly one final test result per run, and a valid
three-seed mean-plus-or-minus-standard-deviation summary for all three pairs.

## Non-Goals

- Changing the canonical training corpus.
- Tuning TALAS loss weights, ASAM radius, temperature, epochs, batch size, or
  learning rate.
- Selecting checkpoints using test results or choosing the best seed.
- Modifying HeatGeo or other distillation methods.
- Replacing the existing server virtual environment or downloading packages
  into a global Python environment.
