# HeatGeo: Heat-Diffusion Manifold Distillation for Text Embeddings

This repository implements HeatGeo, a knowledge distillation method that transfers geometric structure from a large teacher embedding model to a compact student model using heat-diffusion on a teacher-induced kNN graph, augmented with random walk trajectory distillation.

## Supported Distillation Pairs

The framework supports arbitrary teacher-student pairs. Tested configurations include:

| Teacher | Student | Notes |
|:---|:---|:---|
| `Qwen/Qwen3-Embedding-4B` | `google-bert/bert-base-uncased` | Default config |
| `Qwen/Qwen3-Embedding-0.6B` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large` | Lightweight |
| `BAAI/bge-m3` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large` | Alternative teacher |

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

Each diffusion scale uses its own temperature $\tau_r$ to avoid the single-temperature collapse (see docstring in `src/criterions/heatgeo_distillation.py`).

### 6. Loss Function

The total loss combines three components:

$$\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda_{\text{direct}} \cdot \mathcal{L}_{\text{direct}} + \lambda_{\text{walk}} \cdot \mathcal{L}_{\text{walk}}$$

where:

- $\mathcal{L}_{\text{diff}} = \sum_r \omega_r \, \text{KL}(q_{i,r} \| p_{i,r}^S)$ -- multi-scale diffusion matching over the anchor's candidate set
- $\mathcal{L}_{\text{direct}}$ -- direct teacher similarity matching over the full in-batch shared pool (provides absolute calibration)
- $\mathcal{L}_{\text{walk}}$ -- random walk trajectory NLL that scores how well the student follows step-by-step paths sampled from the teacher's transition matrix (see `WALK_DISTILLATION_GUIDE.md`)

### 7. Per-Epoch Candidate Sampling

Each epoch, every anchor draws a fresh candidate set composed of:

- **Diffusion neighbors** from the graph's diffusion pools
- **Hard negatives** (high teacher similarity but outside the mutual kNN graph)
- **Random negatives** (replaced by walk nodes when walk distillation is active)

In-batch sharing deduplicates candidates and exposes each anchor to the full union of candidates in the batch.

## Configuration

All hyperparameters are in `config/heatgeo_config.py`. Key groups:

| Group | Parameters | Description |
|:---|:---|:---|
| Teacher Graph | `graph_k`, `perplexity`, `diffusion_scales`, `truncation_tolerance` | kNN construction, adaptive row bandwidths, and diffusion |
| Candidate Sampling | `candidate_size`, `diffusion_quota`, `hard_neg_k`, `random_neg_k` | Per-anchor candidate composition |
| Walk Distillation | `num_walks`, `walk_length`, `walk_weight`, `walk_start_epoch` | Walk-selected row matching; each row reuses its stored bandwidth |
| Training | `batch_size`, `epochs`, `learning_rate`, `min_lr` | Standard training setup |
| Objectives | `temp_exponent`, `walk_weight`, `lambda_simcse`, `lambda_sim` | Temperature-law exponent and loss weights |

## Training

### Using the shell script

```bash
source venv/bin/activate
bash scripts/train_heatgeo.sh
```

Override settings via environment variables:

```bash
STUDENT_MODEL="nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large" \
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B" \
BATCH_SIZE=32 \
EPOCHS=5 \
bash scripts/train_heatgeo.sh --no_wandb
```

To persist student weights to a durable directory:

```bash
WEIGHTS_DIR="/path/to/weights" bash scripts/train_heatgeo.sh --no_wandb
```

### Using Python directly

```bash
python3 main.py \
  --method heatgeo \
  --train_data data/train_set/merged_3_data_5k_each.csv \
  --student_model nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large \
  --teacher_model Qwen/Qwen3-Embedding-0.6B \
  --batch_size 32 \
  --epochs 5 \
  --lr 2e-5 \
  --save_dir models/heatgeo/qwen3_0_6b_to_minilmv2
```

## Outputs

Model checkpoints are saved under the configured `save_dir`. Training metrics are written to `metrics.jsonl` in the same directory.

Teacher embedding and graph caches are written to `cache/heatgeo/`. If you change the training corpus, teacher model, or graph parameters, delete the old caches before retraining:

```bash
rm -f cache/heatgeo/*.pt
```

kNN graph diagnostics are logged to `logs/heatgeo/knn_graph_neighbors.jsonl`.

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
│   ├── heatgeo_config.py            # HeatGeo hyperparameters
│   └── ...                          # Other method configs
├── src/
│   ├── criterions/
│   │   ├── heatgeo_distillation.py  # HeatGeo loss (diffusion + direct + walk)
│   │   └── ...                      # Other method losses
│   ├── heatgeo/
│   │   ├── graph_builder.py         # kNN graph and diffusion pool construction
│   │   └── candidate_sampler.py     # Per-epoch candidate and walk sampling
│   ├── data_utils/                  # Dataset and collation
│   ├── evaluation/                  # Benchmark evaluation
│   ├── cache_teacher.py             # Teacher embedding caching
│   ├── pooling.py                   # Pooling strategies
│   └── loss.py                      # Shared loss utilities
├── scripts/                         # Training shell scripts
├── data/                            # Train/val/test CSV datasets
├── docs/                            # Reference papers
└── WALK_DISTILLATION_GUIDE.md       # Walk distillation documentation
```
