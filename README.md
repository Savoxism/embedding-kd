# GGPKD: Heat-Diffusion Manifold Distillation for Text Embeddings

This repository implements GGPKD, a knowledge distillation method that transfers geometric structure from a large teacher embedding model to a compact student model by matching the teacher's one-hop kNN transition rows and auxiliary supervision of the
teacher-selected candidate rows.

## Supported Distillation Pairs

The framework supports arbitrary teacher-student pairs. Tested configurations include:

| Pair key | Teacher | Student |
|:---|:---|:---|
| `qwen3_0_6b_to_minilmv2_h384` | `Qwen/Qwen3-Embedding-0.6B` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base` |
| `bge_m3_to_minilmv2_h768` | `BAAI/bge-m3` | `nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base` |
| `qwen3_4b_to_bert_base` | `Qwen/Qwen3-Embedding-4B` | `google-bert/bert-base-uncased` |

`qwen3_0_6b_to_minilmv2_h384` is the default in `config/ggpkd_config.py`. The
launchers take the pair key as their first argument and derive the model names,
pooling, cache and log paths from it, so two pairs never share a cache file.

The training corpus follows the TALAS paper setup: ~15K unlabeled sentences sampled from three in-domain datasets. The default corpus is `data/train_set/merged_3_data_5k_each.csv`.

## Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Method

### 1. Teacher Embedding Cache

Teacher embeddings are pre-computed and cached to avoid repeated forward passes:

$$t_i = T(x_i)$$

### 2. kNN Graph Construction

A kNN graph is built from teacher cosine similarities:

$$\mathcal{N}(i) = \operatorname{TopK}_{j \neq i} \cos(t_i, t_j)$$

No edge is filtered: the neighbour set *is* the retrieved list, every node has degree exactly `graph_k`, and no node can be isolated. A mutual filter (keep $i\to j$ only when the two retrieve each other) suppresses hubs at the cost of deleting teacher-selected relations, and is now the `--knn_mode mutual` arm; E2 measured it 0.36 points below the unfiltered graph.

### 3. Transition Distribution

Each graph row becomes a transition distribution with a row-specific bandwidth $\tau_i$:

$$P_{ij} = \frac{\exp(\left(\cos(t_i, t_j) / \tau_i\right)}{\sum_{u \in \mathcal{N}(i)} \exp(\left(\cos(t_i, t_u) / \tau_i\right)}$$

The bandwidth is not tuned globally, and it is not solved either. Each row reads it straight off the retrieval width:

$$\tau_i = \frac{s_i^{(1)} - s_i^{(k)}}{\log k}$$

where $s_i^{(j)}$ is the $j$-th largest cosine from $i$. The $k$-th retrieved neighbour therefore sits $\log k$ nats below the nearest and is exactly $k$ times less likely, for every node, with no constant left to choose. The row is exactly invariant to $s \mapsto as+b$ — the bandwidth scales with the similarities and a softmax is shift-invariant — which is the property that rules out a single fixed temperature.

This replaced a target-perplexity solve. That solve ran on a filtered neighbour list, where degree varies: on the production corpus 18.9% of nodes had degree at or below $\rho=30$, never reached the target entropy, and were clipped at their own ceiling, yielding near-uniform targets at bandwidths up to 145x the median. Reading the bandwidth off the raw top-$k$ gives every node the same sample size, so there is no target to miss.

The cost is that `graph_k` now sets sharpness as well as width, and the two cannot be varied independently. It is no longer free headroom: too large a $k$ measures the distance out of the anchor's neighbourhood rather than the local decay, and the rows go uniform. `scripts/ggpkd/pick_graph_k.py` reports the induced sharpness per $k$ from one teacher pass; the build warns when `target_kl_uniform_r1` falls under 0.05.

### 4. Transition Targets

The target of anchor $i$ is its transition row $P_{i\cdot}$ itself — one hop, no
composition. Multi-hop targets $q_{i,r} = e_i^\top P^r$ were removed with the
machinery that produced them: the method supervises $r = 1$, so a pool *is* a
transition row, and the sparse matrix powers only ever read back the rows they
were built from.

### 5. Student Distribution

The student predicts a distribution over candidate neighbors:

$$p_i^S(j) = \frac{\exp(\cos(s_i, s_j) / \tau_r)}{\sum_{u \in C_i} \exp(\cos(s_i, s_u) / \tau_r)}$$

The graph scale is matched at the bandwidth its own row was built at, $\tau_i$; the ambient scale uses one temperature on both sides, derived as the median $\tau_i$. Neither is configurable (see the docstring in `src/criterions/ggpkd_distillation.py`).

### 6. Loss Function

All relational scales form one objective, augmented by transition-row supervision:

$$\mathcal{L} = \mathcal{L}_{\text{rel}} + \lambda_{\text{row}} \mathcal{L}_{\text{row}}$$

where:

- $\mathcal{L}_{\text{rel}} = \tfrac{1}{2}\mathrm{KL}(q_{i,0}\|p^S_{i,0}) + \tfrac{1}{2}\mathrm{KL}(q_{i,1}\|p^S_{i,1})$: the ambient scale $r=0$ over the shared pool and the graph scale $r=1$ over the anchor's own row. The 50/50 split is fixed, not tuned.
- $\mathcal{L}_{\text{row}}$ promotes every pool column the teacher selected (the whole draw) to an auxiliary row and matches its available teacher transition row with a dense KL, weighted uniformly. Batch anchors are excluded, since $\mathcal{L}_{\text{rel}}$ already matches their transition row at $r=1$. The row set is a deterministic function of the candidate pool, so this term carries no selection hyperparameter.

### 7. Per-Epoch Candidate Sampling

Every anchor's candidate set is its **whole transition row** — every column the
teacher retrieved, and nothing else. There is no truncation, no budget, no draw
and no RNG: the set is a deterministic function of the graph, identical in every
epoch, and the same width `graph_k` for every anchor. (The padding path — short
rows padded with the anchor's own index, removed from every softmax by
`self_mask` — stays for the arms that produce ragged rows, and is inert here.)

This removed the last tuned quantity in the candidate path. It also removed
`support_policy` from the method: at full width, `topk`, `proportional` and
`uniform` return the same set, so that flag now only means something for an
ablation arm given a budget smaller than the row.

The method draws **no negatives**. It previously added 40 hard negatives (high
teacher similarity, outside the graph) and 26 random negatives per
anchor -- two tuned constants in a method whose other quantities are derived, and
three quarters of its encoder cost. Removing them has two consequences worth
stating together:

- The graph softmax now scores no zero-target column at all, so the
  false-zero gradient that motivated the ambient scale is gone by construction.
  The `amb_mass_on_zero_diff` diagnostic reads ~0.
- Every column the shared pool spans is now someone's teacher-selected
  neighbour, so the ambient scale's comparison is local where it used to reach
  across the corpus; STS Spearman plus the pair-classification thresholds are
  where that would show up first. (The ~4,445 → ~1,400 texts per step recorded
  here was measured with negatives on, then off, both on the mutual graph with
  rows truncated at 99% of their mass. Rows are now kept whole on an unfiltered
  graph, so the pool is wider; re-measure before quoting a number.)

The negative machinery remains reachable through `hard_neg_k` / `random_neg_k`,
because the `no_graph_support` baseline in Tables 2 and 3 spends its entire
budget on uniform corpus draws.

## Configuration

Only genuine experiment controls live in `config/ggpkd_config.py`. Derived
weights, capacities, and correctness policies are resolved internally:

| Group | Parameters | Description |
|:---|:---|:---|
| Teacher Graph | `graph_k` | `graph_k` sets both the kNN width and the per-row bandwidth. `knn_mode` defaults to `directed` (no filter) and `truncation_tolerance` to `0.0` (whole row); both are ablation arms |
| Candidate Sampling | `diffusion_quota`, `hard_neg_k`, `random_neg_k` | All three are `None`/0 in the method: the candidate set is the whole transition row. The ablation baselines set them explicitly |
| Row Supervision | `row_weight` | Weight of the auxiliary transition-row KL; on for every epoch |
| Training | `batch_size`, `epochs`, `learning_rate`, `min_lr` | Standard training setup |
| Ambient profile | `calibration_mode` | `pool` is the method. The shared teacher/student temperature for scale 0 is derived as the median row bandwidth and is not configurable |

GGPKD always uses Top-k support, in-batch sharing and corpus deduplication. Scale weights are `1/r`, the ambient
weight equals the `r=1` weight, and the fixed-bandwidth baseline uses temperature
`0.05`; none is a tunable method knob. Hard-negative storage is sized from
`graph_k` only when an arm actually draws negatives, so the method's build
retrieves top-`graph_k` rather than top-`2*graph_k`.

## Training

### GGPKD on Colab

Open [`notebooks/train_colab_topk_no_neg.ipynb`](notebooks/train_colab_topk_no_neg.ipynb),
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
bash scripts/ggpkd/train.sh
```

To persist student weights to a durable directory:

```bash
WEIGHTS_DIR="/path/to/weights" bash scripts/ggpkd/train.sh
```

### Run cost: time and memory

Every launcher wraps its `main.py` invocation in `scripts/common/run_stats.sh`, so
each run writes a `run_stats.json` next to its own outputs (`SAVE_DIR` /
`RUN_DIR`) and echoes a one-line summary into its log:

```
[stats] wall 1:47:12 (6432.4s) | peak host RSS 11.20 GiB | peak GPU 18.60 GiB | exit 0 -> .../run_stats.json
```

The file records `wall_seconds`, `peak_host_rss_mib`, `peak_gpu_mib`,
`exit_code`, the start/finish timestamps, the pinned device and the exact
command. Both figures are measured from outside the process over the whole
tree -- dataloader workers included -- so a run that dies mid-epoch still leaves
its numbers behind. Host memory comes from the kernel's own high-water mark
(`VmHWM`), GPU memory from per-process `nvidia-smi` accounting sampled every
`RUN_STATS_POLL_SECONDS` (default 2), which is what bounds the GPU spike a run
can hide; `peak_gpu_mib` is `null` where `nvidia-smi` is unavailable. Set
`RUN_STATS_DISABLE=1` to run the command bare, or `STATS_FILE` to redirect the
file.

The multi-run suites additionally collect one row per run into
`<run_root>/stats.tsv` as each finishes -- cache and graph builds included --
and print it before aggregating, so the table survives a sweep that loses a run.

### Multi-seed suites

`run_paper.sh` trains the three paper pairs at three seeds (9 runs), one run per
GPU, and reports mean ± std per pair. `sensitivity.sh` sweeps the two declared
hyperparameters -- `graph_k` and `row_weight` -- on one pair at three seeds
(8 arms, 24 runs) and reports how far the benchmark average moves along each
axis. Both build every teacher cache and graph artifact up front, so no two
concurrent runs write the same cache, and both refuse to aggregate if any run
failed.

```bash
bash scripts/ggpkd/run_paper.sh                    # 3 pairs x 3 seeds
GPUS=0,1,2 bash scripts/ggpkd/sensitivity.sh       # graph_k and row_weight
DRY_RUN=1 GPUS=0 bash scripts/ggpkd/sensitivity.sh # print the arm matrix only
```

Overrides: `PAIRS` / `PAIR`, `SEEDS`, `GPUS`, `RUN_ID`, `RESULT_BASE`,
`CACHE_ROOT`, `PYTHON_BIN`. A sweep over another axis needs no edit to the
script -- `ARMS` (or `ARMS_FILE`) and `GRAPH_SPEC` replace the arm matrix; give
any axis that changes the graph its own `GRAPH_SPEC` key so each artifact is
built once instead of being rebuilt per arm. Both summaries can be re-run
standalone over a finished tree:

```bash
python scripts/ggpkd/summarize.py results/ggpkd/<run_id>
python scripts/ggpkd/summarize_sensitivity.py results/ggpkd_sensitivity/<run_id>
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

Teacher embedding and graph caches are written to `cache/ggpkd/<pair_key>/`. If you
change the training corpus, teacher model, or graph parameters, delete that pair's
caches before retraining:

```bash
rm -f cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/*.pt
```

kNN graph diagnostics are logged to
`logs/ggpkd/<pair_key>/knn_graph_neighbors.jsonl`.

## Benchmarks

The training loop evaluates on 9 benchmarks after each epoch:

| Family | Benchmarks |
|:---|:---|
| Classification | Banking77, Emotion, Tweet |
| Pair Classification | MRPC, SciTail, WiC |
| Semantic Textual Similarity | SICK, STS12, STS-B |

Validation runs after each epoch. Test evaluation runs once after training.

## Adding a Method

`distiller.py` never compares `config.distill_method` against a method name. It
reads a `MethodSpec` from `src/methods/`, whose flags answer the questions the
shared pipeline asks and whose hooks replace the parts that differ:

```python
SPEC = MethodSpec(
    name="ggpkd",
    config_cls=GGPKDConfig,
    step=step,                  # src/distill/steps/ggpkd.py
    uses_teacher_cache=True,    # precomputed teacher embeddings
    batch_relational=True,      # -> DataLoader(drop_last=True)
    prepare_frame=prepare_frame,
    build_data=build_data,
    build_criterion=build_criterion,
    on_epoch_start=on_epoch_start,
)
```

Every hook defaults to `None`, meaning "use the shared path", so a spec carries
only what actually differs. A new method is one module under `src/methods/` plus
one line in its `REGISTRY`; `--method` choices and the config class follow from
there. `src/methods/spec.py` documents each field.

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
├── distiller.py                     # Method-agnostic training loop and evaluation
├── config/
│   ├── base_config.py               # Shared defaults, override checking, validate()
│   ├── ggpkd_config.py              # GGPKD hyperparameters
│   └── ...                          # Other method configs
├── src/
│   ├── methods/                     # One module per method: what it *is*
│   │   ├── spec.py                  # MethodSpec: the flags and hooks
│   │   ├── __init__.py              # REGISTRY -- the only list of methods
│   │   └── ggpkd.py, talas.py, ...  # Per-method data, criterion, optimizer, KD term
│   ├── criterions/
│   │   ├── ggpkd_distillation.py    # Relational and row objectives
│   │   └── ...                      # Other method losses
│   ├── ggpkd/
│   │   ├── graph_builder.py         # kNN graph and candidate pool construction
│   │   ├── candidate_sampler.py     # Candidate-pool construction
│   │   └── policy.py                # Derived capacities and tolerances
│   ├── distill/
│   │   ├── steps/                   # One training step per shape of step
│   │   └── ...                      # Checkpointing, telemetry, geometry, benchmarks
│   ├── data_utils/                  # Datasets and collates
│   ├── models/                      # Students that are not a plain AutoModel
│   ├── evaluation/                  # Benchmark evaluation
│   ├── cache_teacher.py             # Teacher embedding caching
│   ├── pooling.py                   # Pooling strategies
│   └── loss.py                      # Shared loss utilities
├── scripts/                         # Launchers, one folder per method
│   ├── <method>/train.sh            # Bash launcher
│   ├── common/run_stats.sh          # Wall clock + peak host/GPU memory
│   ├── ggpkd/pick_graph_k.py        # graph_k sharpness report
│   ├── ggpkd/bench_encode.py        # Candidate-encoder throughput bench
│   ├── ggpkd/run_paper.sh           # 3 pairs x 3 seeds -> summarize.py
│   ├── ggpkd/sensitivity.sh         # knob sweep -> summarize_sensitivity.py
│   ├── ggpkd/run_metrics.py         # Shared run-tree reader for both summaries
│   └── talas/                       # + run_paper.sh, summarize.py
├── data/                            # Train/val/test CSV datasets
├── notebooks/                       # Colab training notebook
├── tests/                           # pytest suite
├── docs/                            # Reference papers and experiment notes
```
