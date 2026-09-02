# GGPKD: Heat-Diffusion Manifold Distillation for Text Embeddings

This repository implements GGPKD, a knowledge distillation method that transfers geometric structure from a large teacher embedding model to a compact student model using heat diffusion on a teacher-induced kNN graph and auxiliary supervision of the
teacher-selected candidate rows.

## Supported Distillation Pairs

The framework supports arbitrary teacher-student pairs. Tested configurations include:

| Teacher | Student | Notes |
|:---|:---|:---|
| `Qwen/Qwen3-Embedding-4B` | `google-bert/bert-base-uncased` | Default config |
| `Qwen/Qwen3-Embedding-0.6B` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base` | Lightweight |
| `BAAI/bge-m3` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base` | Alternative teacher |

The training corpus follows the TALAS paper setup: ~15K unlabeled sentences sampled from three in-domain datasets. The default corpus is `data/train_set/merged_3_data_5k_each.csv`.

## Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Weights & Biases logging:

```bash
wandb login
# Or for offline mode:
export WANDB_MODE=offline
```

## Method

### 1. Teacher Embedding Cache

Teacher embeddings are pre-computed and cached to avoid repeated forward passes:

$$t_i = T(x_i)$$

### 2. kNN Graph Construction

A mutual kNN graph is built from teacher cosine similarities:

$$\mathcal{N}_k(i) = \operatorname{TopK}_{j \neq i} \cos(t_i, t_j)$$

Edges are kept only when both endpoints agree (mutual kNN). Isolated nodes fall back to ordinary top-k neighbors.

### 3. Transition Distribution

Each graph row becomes a transition distribution with a row-specific bandwidth $\tau_i$:

$$P_{ij} = \frac{\exp(\left(\cos(t_i, t_j) / \tau_i\right)}{\sum_{u \in \mathcal{N}(i)} \exp(\left(\cos(t_i, t_u) / \tau_i\right)}$$

The bandwidth is not tuned globally. For every non-degenerate row, the graph builder solves $H(P_i)=\log\rho$ with target perplexity $\rho=30$. Rows whose degree is at most $\rho$ are clipped near their maximum attainable entropy. The solved bandwidths are stored in the graph artifact and reused by the student readouts.

### 4. Multi-Scale Diffusion Targets

Multi-scale targets capture structure at different resolutions:

$$q_{i,r} = e_i^\top P^r, \quad r \in \{1, 2, 4\}$$

### 5. Student Distribution

The student predicts a distribution over candidate neighbors:

$$p_i^S(j) = \frac{\exp(\cos(s_i, s_j) / \tau_r)}{\sum_{u \in C_i} \exp(\cos(s_i, s_u) / \tau_r)}$$

Each diffusion scale uses its own temperature $\tau_r$ to avoid the single-temperature collapse (see docstring in `src/criterions/ggpkd_distillation.py`).

### 6. Loss Function

All relational scales form one objective, augmented by transition-row supervision
and an optional unbiased geometry estimator:

$$\mathcal{L} = \mathcal{L}_{\text{rel}} + \lambda_{\text{row}} \mathcal{L}_{\text{row}} + \lambda_{\text{geom}} \widehat{\mathcal E}$$

where:

- $\mathcal{L}_{\text{rel}} = \sum_{r\in\{0,1,2,4\}}\omega_r\,\mathrm{KL}(q_{i,r}\|p^S_{i,r})$, with the single fixed rule $\omega_r\propto1/\max(1,r)$, normalized over the scales. Here $r=0$ is ambient, $r=1$ is direct-neighbor matching, and $r>1$ is multi-hop diffusion; these are diagnostic names rather than separately weighted auxiliary losses.
- $\mathcal{L}_{\text{row}}$ promotes every pool column the teacher selected (the diffusion support, excluding hard and uniform negatives) to an auxiliary row and matches its available teacher transition row with a dense KL, weighted uniformly. Batch anchors are excluded, since $\mathcal{L}_{\text{rel}}$ already matches their transition row at $r=1$. The row set is a deterministic function of the candidate pool, so this term carries no selection hyperparameter.
- $\widehat{\mathcal E}$ evaluates the sampled head exactly and weights one teacher-proportional tail draw by its remaining mass. It is unbiased for teacher-weighted cosine distortion on each cached diffusion target. `unbiased_geometry_weight=0` keeps the established KL objective unchanged.

### 7. Per-Epoch Candidate Sampling

Each epoch, every anchor draws a fresh candidate set composed of:

- **Diffusion neighbors** from the graph's diffusion pools
- **Hard negatives** (high teacher similarity but outside the mutual kNN graph)
- **Random negatives** from the remaining complement

In-batch sharing deduplicates candidates and exposes each anchor to the full union of candidates in the batch.

## Configuration

Only genuine experiment controls live in `config/ggpkd_config.py`. Derived
weights, capacities, and correctness policies are resolved internally:

| Group | Parameters | Description |
|:---|:---|:---|
| Teacher Graph | `graph_k`, `perplexity`, `diffusion_scales`, `truncation_tolerance` | kNN construction, adaptive row bandwidths, and diffusion |
| Candidate Sampling | `diffusion_quota`, `hard_neg_k`, `random_neg_k` | Per-anchor composition; total size is their sum |
| Row Supervision | `row_weight` | Weight of the auxiliary transition-row KL (`row_start_epoch` defaults to 1, i.e. always on) |
| Geometry estimator | `unbiased_geometry_weight` | Weight of the optional unbiased head--tail cosine-distortion term; default `0` |
| Training | `batch_size`, `epochs`, `learning_rate`, `min_lr` | Standard training setup |
| Ambient profile | `direct_temp` | Shared teacher/student temperature for scale 0 |

GGPKD always uses in-batch sharing, per-epoch stochastic resampling, and corpus
deduplication. Scale weights are `1/r`, the ambient
weight equals the `r=1` weight, hard-negative storage equals `graph_k`, and the
fixed-bandwidth baseline uses temperature `0.05`; none is a tunable method knob.

## Training

### GGPKD on Colab

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb),
choose one of the three canonical teacher--student pairs, set `ROW_WEIGHT`, and
run all cells. The notebook clones `nqd_mass_geom_loss` on its first run and
fetches/resets/pulls the latest remote commit on every later run.

### RKD baseline

RKD uses the paper-default RKD-DA objective (distance weight 1, angle weight 2,
no task loss) and the original metric-learning optimizer schedule (Adam,
batch 128, 80 epochs, learning rate `1e-4`, decays at epochs 40 and 60). Teacher
embeddings are cached before student training.

```bash
bash scripts/train_rkd.sh
```

Select another supported teacher-student pair or prepare only its cache:

```bash
bash scripts/train_rkd.sh bge_m3_to_minilmv2_h768
bash scripts/train_rkd.sh qwen3_4b_to_bert_base --prepare-cache
```

For a quick smoke run, override the expensive paper defaults through environment
variables:

```bash
BATCH_SIZE=16 EPOCHS=1 bash scripts/train_rkd.sh
```

### Using the shell script

```bash
source venv/bin/activate
bash scripts/train_ggpkd.sh
```

Override settings via environment variables:

```bash
STUDENT_MODEL="nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base" \
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B" \
BATCH_SIZE=32 \
EPOCHS=5 \
bash scripts/train_ggpkd.sh --no_wandb
```

To persist student weights to a durable directory:

```bash
WEIGHTS_DIR="/path/to/weights" bash scripts/train_ggpkd.sh --no_wandb
```

### Using Python directly

```bash
python3 main.py \
  --method ggpkd \
  --train_data data/train_set/merged_3_data_5k_each.csv \
  --student_model nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base \
  --teacher_model Qwen/Qwen3-Embedding-0.6B \
  --batch_size 32 \
  --epochs 5 \
  --lr 2e-5 \
  --unbiased_geometry_weight 0.1 \
  --save_dir models/ggpkd/qwen3_0_6b_to_minilmv2
```

## Outputs

Model checkpoints are saved under the configured `save_dir`. Training metrics are written to `metrics.jsonl` in the same directory.

Teacher embedding and graph caches are written to `cache/ggpkd/`. If you change the training corpus, teacher model, or graph parameters, delete the old caches before retraining:

```bash
rm -f cache/ggpkd/*.pt
```

kNN graph diagnostics are logged to `logs/ggpkd/knn_graph_neighbors.jsonl`.

## Benchmarks

The training loop evaluates on 9 benchmarks after each epoch:

| Family | Benchmarks |
|:---|:---|
| Classification | Banking77, Emotion, Tweet |
| Pair Classification | MRPC, SciTail, WiC |
| Semantic Textual Similarity | SICK, STS12, STS-B |

Validation runs after each epoch. Test evaluation runs once after training.

## Other Distillation Methods

This repository also includes implementations of other distillation baselines for comparison:

- **TALAS** (`config/talas_config.py`, `scripts/train_talas.sh`)
- **RKD** (`config/rkd_config.py`, `scripts/train_rkd.sh`)
- **CDM** (`config/cdm_config.py`, `scripts/train_cdm.sh`)
- **DSKD** (`config/dskd_config.py`, `scripts/train_dskd.sh`)
- **EMO** (`config/emo_config.py`, `scripts/train_emo.sh`)
- **Stella** (`config/stella_config.py`, `scripts/train_stella.sh`)

## Project Structure

```
.
├── main.py                          # Entry point
├── distiller.py                     # Training loop and evaluation
├── config/
│   ├── base_config.py               # Shared defaults
│   ├── ggpkd_config.py            # GGPKD hyperparameters
│   └── ...                          # Other method configs
├── src/
│   ├── criterions/
│   │   ├── ggpkd_distillation.py  # Relational, row, and geometry objectives
│   │   └── ...                      # Other method losses
│   ├── ggpkd/
│   │   ├── graph_builder.py         # kNN graph and diffusion pool construction
│   │   └── candidate_sampler.py     # Per-epoch candidate sampling
│   ├── data_utils/                  # Dataset and collation
│   ├── evaluation/                  # Benchmark evaluation
│   ├── cache_teacher.py             # Teacher embedding caching
│   ├── pooling.py                   # Pooling strategies
│   └── loss.py                      # Shared loss utilities
├── scripts/                         # Training shell scripts
├── data/                            # Train/val/test CSV datasets
├── docs/                            # Reference papers
```
