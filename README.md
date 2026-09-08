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

The bandwidth is not tuned globally, and it is not solved either. Each row reads it straight off the retrieval width:

$$\tau_i = \frac{s_i^{(1)} - s_i^{(k)}}{\log k}$$

where $s_i^{(j)}$ is the $j$-th largest cosine from $i$. The $k$-th retrieved neighbour therefore sits $\log k$ nats below the nearest and is exactly $k$ times less likely, for every node, with no constant left to choose. The row is exactly invariant to $s \mapsto as+b$ — the bandwidth scales with the similarities and a softmax is shift-invariant — which is the property that rules out a single fixed temperature.

This replaced a target-perplexity solve. That solve ran on the mutual-filtered neighbour list, where degree varies: on the production corpus 18.9% of nodes had degree at or below $\rho=30$, never reached the target entropy, and were clipped at their own ceiling, yielding near-uniform targets at bandwidths up to 145x the median. Reading the bandwidth off the raw top-$k$ gives every node the same sample size, so there is no target to miss.

The cost is that `graph_k` now sets sharpness as well as width, and the two cannot be varied independently. It is no longer free headroom: too large a $k$ measures the distance out of the anchor's neighbourhood rather than the local decay, and the rows go uniform. `scripts/ggpkd/pick_graph_k.py` reports the induced sharpness per $k$ from one teacher pass; the build warns when `target_kl_uniform_r1` falls under 0.05.

### 4. Multi-Scale Diffusion Targets

Multi-scale targets capture structure at different resolutions:

$$q_{i,r} = e_i^\top P^r, \quad r \in \{1, 2, 4\}$$

### 5. Student Distribution

The student predicts a distribution over candidate neighbors:

$$p_i^S(j) = \frac{\exp(\cos(s_i, s_j) / \tau_r)}{\sum_{u \in C_i} \exp(\cos(s_i, s_u) / \tau_r)}$$

Each diffusion scale uses its own temperature $\tau_r$ to avoid the single-temperature collapse (see docstring in `src/criterions/ggpkd_distillation.py`).

### 6. Loss Function

All relational scales form one objective, augmented by transition-row supervision:

$$\mathcal{L} = \mathcal{L}_{\text{rel}} + \lambda_{\text{row}} \mathcal{L}_{\text{row}}$$

where:

- $\mathcal{L}_{\text{rel}} = \sum_{r\in\{0,1,2,4\}}\omega_r\,\mathrm{KL}(q_{i,r}\|p^S_{i,r})$, with the single fixed rule $\omega_r\propto1/\max(1,r)$, normalized over the scales. Here $r=0$ is ambient, $r=1$ is direct-neighbor matching, and $r>1$ is multi-hop diffusion; these are diagnostic names rather than separately weighted auxiliary losses.
- $\mathcal{L}_{\text{row}}$ promotes every pool column the teacher selected (the diffusion support, which is now the whole draw) to an auxiliary row and matches its available teacher transition row with a dense KL, weighted uniformly. Batch anchors are excluded, since $\mathcal{L}_{\text{rel}}$ already matches their transition row at $r=1$. The row set is a deterministic function of the candidate pool, so this term carries no selection hyperparameter.

### 7. Per-Epoch Candidate Sampling

Every anchor's candidate set is its **whole truncated transition row** — every
column the teacher put diffusion mass on, and nothing else. There is no budget,
no draw and no RNG: the set is a deterministic function of the graph and is
identical in every epoch. Row width for collation is the pool width; anchors with
shorter rows are padded with their own index, which `self_mask` removes from every
softmax.

This removed the last tuned quantity in the candidate path. It also removed
`support_policy` from the method: at full width, `topk`, `proportional` and
`uniform` return the same set, so that flag now only means something for an
ablation arm given a budget smaller than the row.

The method draws **no negatives**. It previously added 40 hard negatives (high
teacher similarity, outside the mutual kNN graph) and 26 random negatives per
anchor -- two tuned constants in a method whose other quantities are derived, and
three quarters of its encoder cost. Removing them has two consequences worth
stating together:

- The diffusion softmax now scores no zero-target column at all, so the
  false-zero gradient that motivated the ambient scale is gone by construction.
  The `amb_mass_on_zero_diff` diagnostic reads ~0.
- The shared pool falls from ~4,445 to ~1,400 texts per step, so the ambient
  scale calibrates over columns that are all someone's teacher-selected
  neighbour. Its comparison is local where it used to reach across the corpus,
  and STS Spearman plus the pair-classification thresholds are where that would
  show up first.

The negative machinery remains reachable through `hard_neg_k` / `random_neg_k`,
because the `no_graph_support` baseline in Tables 2 and 3 spends its entire
budget on uniform corpus draws.

## Configuration

Only genuine experiment controls live in `config/ggpkd_config.py`. Derived
weights, capacities, and correctness policies are resolved internally:

| Group | Parameters | Description |
|:---|:---|:---|
| Teacher Graph | `graph_k`, `diffusion_scales` | `graph_k` sets both the kNN width and the per-row bandwidth; `truncation_tolerance` is a numerical-fidelity constant in `policy.py` |
| Candidate Sampling | `diffusion_quota`, `hard_neg_k`, `random_neg_k` | All three are `None`/0 in the method: the candidate set is the whole transition row. The ablation baselines set them explicitly |
| Row Supervision | `row_weight` | Weight of the auxiliary transition-row KL (`row_start_epoch` defaults to 1, i.e. always on) |
| Training | `batch_size`, `epochs`, `learning_rate`, `min_lr` | Standard training setup |
| Ambient profile | `direct_temp` | Shared teacher/student temperature for scale 0 |

GGPKD always uses Top-k support, in-batch sharing and corpus deduplication. Scale weights are `1/r`, the ambient
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
bash scripts/rkd/train.sh
```

Select another supported teacher-student pair or prepare only its cache:

```bash
bash scripts/rkd/train.sh bge_m3_to_minilmv2_h768
bash scripts/rkd/train.sh qwen3_4b_to_bert_base --prepare-cache
```

For a quick smoke run, override the expensive paper defaults through environment
variables:

```bash
BATCH_SIZE=16 EPOCHS=1 bash scripts/rkd/train.sh
```

### Using the shell script

```bash
source venv/bin/activate
bash scripts/ggpkd/train.sh
```

Override settings via environment variables:

```bash
STUDENT_MODEL="nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base" \
TEACHER_MODEL="Qwen/Qwen3-Embedding-0.6B" \
BATCH_SIZE=32 \
EPOCHS=5 \
bash scripts/ggpkd/train.sh --no_wandb
```

To persist student weights to a durable directory:

```bash
WEIGHTS_DIR="/path/to/weights" bash scripts/ggpkd/train.sh --no_wandb
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

- **TALAS** (`config/talas_config.py`, `scripts/talas/train.sh`)
- **RKD** (`config/rkd_config.py`, `scripts/rkd/train.sh`)
- **CDM** (`config/cdm_config.py`, `scripts/cdm/train.sh`)
- **DSKD** (`config/dskd_config.py`, `scripts/dskd/train.sh`)
- **EMO** (`config/emo_config.py`, `scripts/emo/train.sh`)
- **Stella** (`config/stella_config.py`, `scripts/stella/train.sh`)

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
├── scripts/                         # Launchers, one folder per method
│   ├── <method>/train.sh            # Bash launcher (train.ps1 = PowerShell)
│   ├── ggpkd/floor.py               # L_rel floor diagnostic
│   └── talas/                       # + run_paper.sh, summarize.py
├── data/                            # Train/val/test CSV datasets
├── docs/                            # Reference papers
```
