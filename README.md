# HeatGeo: Heat-Diffusion Manifold Distillation for Text Embeddings

This repository implements HeatGeo, a knowledge distillation method that transfers geometric structure from a large teacher embedding model to a compact student model using heat diffusion on a teacher-induced kNN graph and explicit support-mass calibration.

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

The ambient and diffusion scales form one derived-weight relational objective:

$$\mathcal{L} = \mathcal{L}_{\text{rel}} + \lambda_{\text{mass}} \cdot \mathcal{L}_{\text{mass}}$$

where:

- $\mathcal{L}_{\text{rel}} = \sum_{r\in\{0,1,2,4\}} \omega_r \, \text{KL}(q_{i,r} \| p_{i,r}^S)$, with $\omega_r=1/r$ for diffusion and $\omega_0=\omega_1$; scale 0 is the ambient teacher-similarity profile.
- $\mathcal{L}_{\text{mass}}$ matches the teacher and student probability assigned to the selected support with a Bernoulli KL. Hard and uniform ambient samples estimate the student's complement partition using inverse inclusion probabilities.

### 7. Per-Epoch Candidate Sampling

Each epoch, every anchor draws a fresh candidate set composed of:

- **Diffusion neighbors** from the graph's diffusion pools
- **Hard negatives** (high teacher similarity but outside the mutual kNN graph)
- **Random negatives** from the remaining complement

Hard and random negatives are sampled as disjoint strata. In-batch sharing deduplicates candidates and exposes each anchor to the full union of candidates in the batch.

## Configuration

Only genuine experiment controls live in `config/heatgeo_config.py`. Derived
weights, capacities, and correctness policies are resolved internally:

| Group | Parameters | Description |
|:---|:---|:---|
| Teacher Graph | `graph_k`, `perplexity`, `diffusion_scales`, `truncation_tolerance` | kNN construction, adaptive row bandwidths, and diffusion |
| Candidate Sampling | `diffusion_quota`, `hard_neg_k`, `random_neg_k` | Per-anchor composition; total size is their sum |
| Mass Calibration | `mass_weight` | Weight of selected-vs-complement Bernoulli KL |
| Training | `batch_size`, `epochs`, `learning_rate`, `min_lr` | Standard training setup |
| Ambient profile | `direct_temp` | Shared teacher/student temperature for scale 0 |

RIPPLE always uses in-batch sharing, per-epoch stochastic resampling, and corpus
deduplication. Scale weights are `1/r`, the ambient
weight equals the `r=1` weight, hard-negative storage equals `graph_k`, and the
fixed-bandwidth baseline uses temperature `0.05`; none is a tunable method knob.

## Training

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
│   ├── heatgeo_config.py            # HeatGeo hyperparameters
│   └── ...                          # Other method configs
├── src/
│   ├── criterions/
│   │   ├── heatgeo_distillation.py  # HeatGeo loss (relational + support mass)
│   │   └── ...                      # Other method losses
│   ├── heatgeo/
│   │   ├── graph_builder.py         # kNN graph and diffusion pool construction
│   │   └── candidate_sampler.py     # Support and ambient-stratum sampling
│   ├── data_utils/                  # Dataset and collation
│   ├── evaluation/                  # Benchmark evaluation
│   ├── cache_teacher.py             # Teacher embedding caching
│   ├── pooling.py                   # Pooling strategies
│   └── loss.py                      # Shared loss utilities
├── scripts/                         # Training shell scripts
├── data/                            # Train/val/test CSV datasets
├── docs/                            # Reference papers
```
