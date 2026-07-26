# HeatGeo Reproduction Guide

This repository trains compact text embedding models with HeatGeo: a heat-diffusion manifold distillation method. The reproduction target here follows the first teacher-student pair in `docs/TALAS.pdf`:

```text
Qwen3-Embedding-0.6B -> MiniLMv2 H384 22M
```

The training corpus follows the TALAS paper setup: about 15K unlabeled sentences sampled from the three in-domain datasets EMOTION, WiC, and STS-B. In this repo, that corpus is:

```text
data/merged_3_data_5k_each.csv
```

The benchmark CSV files under `data/multi-data/` are kept for evaluation.

## Environment

Do not install packages into the global Python environment. Create and use a project virtual environment:

```bash
cd /Users/savoxism/Documents/GitHub/ICLR-MDD
python3 -m venv venv
source venv/bin/activate
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

## Method

HeatGeo first caches teacher embeddings:

$$
t_i = T(x_i)
$$

Then it builds a teacher-induced kNN graph:

$$
\mathcal N_k(i)=\operatorname{TopK}_{j\neq i}\cos(t_i,t_j)
$$

The graph uses mutual kNN edges:

$$
j\in\mathcal N_k(i)
\quad\text{and}\quad
i\in\mathcal N_k(j)
$$

If a node has no mutual neighbor, HeatGeo falls back to its ordinary top-k neighbors and logs the fallback.

Each graph row becomes a transition distribution:

$$
P_{ij}
=
\frac{\exp(\cos(t_i,t_j)/\tau_g)}
{\sum_{u\in\mathcal N(i)}\exp(\cos(t_i,t_u)/\tau_g)}
$$

Multi-scale diffusion targets are:

$$
q_{i,r}=e_i^\top P^r,
\qquad
r\in\{1,2,4\}
$$

During training, the student predicts a distribution over candidate neighbors:

$$
p_i^S(j)
=
\frac{\exp(\cos(s_i,s_j)/\tau_s)}
{\sum_{u\in C_i}\exp(\cos(s_i,s_u)/\tau_s)}
$$

The main HeatGeo objective is:

$$
\mathcal L
=
w_{\mathrm{task}}\mathcal L_{\mathrm{InfoNCE}}
+
\lambda_{\mathrm{diff}}\mathcal L_{\mathrm{diff}}
+
\lambda_{\mathrm{spec}}\mathcal L_{\mathrm{spec}}
+
\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}
$$

The default config is in `config/heatgeo_config.py`.

## Run HeatGeo

From the repo root:

```bash
source venv/bin/activate
bash scripts/train_heatgeo.sh
```

The script is Mac Apple Silicon friendly by default:

```text
PYTORCH_ENABLE_MPS_FALLBACK=1
BATCH_SIZE=4
```

You can override settings without editing the file:

```bash
BATCH_SIZE=2 EPOCHS=5 WANDB_MODE=online bash scripts/train_heatgeo.sh
```

To disable W&B:

```bash
bash scripts/train_heatgeo.sh --no_wandb
```

Or run the Python entry point directly:

```bash
python3 main.py \
  --method heatgeo \
  --train_data data/merged_3_data_5k_each.csv \
  --student_model jim12345/MiniLMv2-L6-H384-distilled-from-BERT-Base \
  --teacher_model Qwen/Qwen3-Embedding-0.6B \
  --batch_size 4 \
  --epochs 5 \
  --lr 2e-5 \
  --max_length 256 \
  --save_dir models/heatgeo/qwen3_0_6b_to_minilmv2_h384
```

## Outputs

Model checkpoints and weights are saved under:

```text
models/heatgeo/qwen3_0_6b_to_minilmv2_h384/
```

Training and benchmark metrics are written to:

```text
models/heatgeo/qwen3_0_6b_to_minilmv2_h384/metrics.jsonl
```

Teacher and graph caches are written to:

```text
cache/heatgeo/qwen3_0_6b_minilmv2_h384_teacher_train.pt
cache/heatgeo/qwen3_0_6b_minilmv2_h384_graph.pt
```

kNN graph logs are written to:

```text
logs/heatgeo/knn_graph_neighbors.jsonl
```

Each graph-log node row contains:

```json
{
  "idx": 0,
  "fallback_used": false,
  "neighbors": [12, 91, 7],
  "transition_probs": [0.41, 0.33, 0.26],
  "cosine_scores": [0.82, 0.80, 0.78]
}
```

The first row is a summary with fallback counts, fallback rate, and degree statistics.

## Rebuilding Caches

If you change the training corpus, teacher model, or HeatGeo graph parameters, remove the old caches first:

```bash
rm cache/heatgeo/qwen3_0_6b_minilmv2_h384_teacher_train.pt
rm cache/heatgeo/qwen3_0_6b_minilmv2_h384_graph.pt
```

Then rerun:

```bash
bash scripts/train_heatgeo.sh
```

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

Validation runs after each epoch. Final test evaluation runs after training, reusing pair-classification thresholds selected on validation.
