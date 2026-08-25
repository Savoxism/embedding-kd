# ICLR Embedding KD

This repository distills compact text embedding students from large embedding
teachers. The reproduction target follows the Qwen3 to BERT-base pair in
`docs/TALAS.pdf`:

```text
Qwen3-Embedding-4B -> BERT-base 109M
```

The training corpus follows the TALAS paper setup: about 15K unlabeled sentences
sampled from the three in-domain datasets EMOTION, WiC, and STS-B. In this repo,
that corpus is:

```text
data/train_set/merged_3_data_5k_each.csv
```

Benchmark CSV files are separated under `data/train_set/`, `data/val_set/`,
and `data/test_set/`. Classification probe train and validation files are
checked for normalized-text leakage before evaluation.

## Environment

Do not install packages into the global Python environment. Create and use a
project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Weights & Biases logging:

```bash
wandb login
```

If you want a local/offline W&B run:

```bash
export WANDB_MODE=offline
```

## Methods

`--method` selects the distillation objective. Each one has its own config class
under `config/`, which supplies the defaults that the CLI flags then override:

| `--method` | Config | Objective |
| --- | --- | --- |
| `talas` | `config/talas_config.py` | Teacher-anchor KD on cached teacher embeddings, with a structural term and SAM |
| `cdm` | `config/cdm_config.py` | Contextual dynamic mapping across the two tokenizers |
| `dskd` | `config/dskd_config.py` | Dual-space KD with a learned projection |
| `emo` | `config/emo_config.py` | Optimal-transport embedding distillation |
| `stella` | `config/stella_config.py` | Stella multi-dimension student heads |

`talas` caches the teacher's sentence embeddings once (`cache_path`) and frees
the teacher model afterwards; the other methods run the teacher alongside the
student every step.

Training is single-process. Two visible CUDA devices place the student on
`cuda:0` and the teacher on `cuda:1`; one device puts both on `cuda:0`.

## Run

From the repo root:

```bash
source .venv/bin/activate
bash scripts/train_talas.sh
```

One script per method lives in `scripts/` (`train_talas.sh`, `train_cdm.sh`,
`train_dskd.sh`, `train_emo.sh`, `train_stella.sh`, plus `.ps1` equivalents).

Or run the Python entry point directly:

```bash
python3 main.py \
  --method talas \
  --train_data data/train_set/merged_3_data_5k_each.csv \
  --student_model google-bert/bert-base-uncased \
  --teacher_model Qwen/Qwen3-Embedding-4B \
  --batch_size 32 \
  --epochs 5 \
  --lr 2e-5 \
  --max_length 256 \
  --save_dir models/talas/qwen3_4b_to_bert_base
```

To disable W&B:

```bash
python3 main.py --method talas --no_wandb
```

To persist student weights after every epoch, provide a durable directory:

```bash
python3 main.py --method talas \
  --weights_dir "/content/drive/MyDrive/[ICLR] Embedding KD/weights/qwen3_4b_to_bert_base"
```

`test_mdd.ipynb` is the Colab runner: it trains several teacher/student pairs
sequentially and writes per-pair logs, tables and figures to Google Drive.

## Outputs

Model checkpoints and weights are saved under `--save_dir`, e.g.:

```text
models/talas/qwen3_4b_to_bert_base/
```

Training and benchmark metrics are written to:

```text
models/talas/qwen3_4b_to_bert_base/metrics.jsonl     # one record per epoch
models/talas/qwen3_4b_to_bert_base/step_metrics.jsonl # one record per optimizer step
```

Validation is evaluated and printed after every epoch. Test is evaluated and
printed once after training. The Colab notebook exports the two splits
separately:

```text
models/talas/qwen3_4b_to_bert_base/validation_by_epoch.csv
models/talas/qwen3_4b_to_bert_base/final_test_results.csv
```

The teacher embedding cache is written to `--cache_path`, e.g.:

```text
cache/talas/qwen3_4b_bert_base_teacher_train.pt
```

## Rebuilding Caches

The cache is keyed only by its path, so a change of training corpus, teacher
model or pooling needs the old file removed first:

```bash
rm cache/talas/qwen3_4b_bert_base_teacher_train.pt
```

Then rerun training. A cache whose row count does not match the corpus is
rejected at startup rather than silently misaligned.

## Benchmarks

The training loop evaluates these benchmark groups:

Classification:

```text
Banking77, Emotion, Tweet
```

Pair classification:

```text
MRPC, SciTail, WiC
```

Semantic textual similarity:

```text
SICK, STS12, STS-B
```

Validation runs after each epoch. Final test evaluation runs after training,
reusing pair-classification thresholds selected on validation.

Only the three classification benchmarks read a train split: their score comes
from a logistic-regression probe fitted on train embeddings. The pair and STS
benchmarks are scored without fitting anything, so their train splits are on
disk for completeness only. To (re)fetch them:

```bash
python3 scripts/download_eval_train_splits.py            # skips what exists
python3 scripts/download_eval_train_splits.py --force    # refetch everything
```

The script rebuilds each benchmark's existing validation and test files from the
upstream source first and refuses to write the train split unless they match, so
a source that has drifted fails loudly instead of landing a mismatched split.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```
