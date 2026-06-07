## Project Setup

### Files NOT Included in Repository (gitignored)

The following directories and files are excluded from version control and must be set up manually:

**Data directories:**
- `data/` - Training and evaluation datasets
- `model_hub/` - Local model files
- `*.ipynb` - Jupyter notebook files

**Generated files:**
- `checkpoints/` - Model checkpoints during/after training
- `cache/` - Cached teacher embeddings
- `*.pt`, `*.pth`, `*.ckpt` - PyTorch model weights
- `logs/`, `*.log` - Training logs

**Python artifacts:**
- `__pycache__/`, `*.pyc`, `*.pyo` - Python bytecode
- `*.egg-info/`, `dist/`, `build/` - Package build files

**Environment:**
- `venv/`, `env/` - Virtual environment directories
- `.env`, `.env.local` - Environment variables

**IDE and temp files:**
- `.vscode/`, `.idea/` - IDE configuration
- `*.swp`, `*.tmp`, `*.bak` - Temporary files

### Required Directory Setup

Before running training, create these directories and add your data:

```bash
# Create required directories
mkdir -p data/multi-data
mkdir -p model_hub
mkdir -p scripts/checkpoints
mkdir -p scripts/cache
```

**1. Training Data Setup:**

Get model_hub from `https://github.com/microsoft/unilm/tree/master/minilm`

Place your training CSV file in `data/` directory:
- File format: CSV with columns `text` or `premise`, `hypothesis`
- Example files: `merged_3_data_5k_each.csv`

Update the `--train_data` parameter in training scripts:
- **PowerShell**: Edit `scripts/train_*.ps1`
- **Bash**: Edit `scripts/train_*.sh`

```powershell
# Example in train_talas.ps1
$TRAIN_DATA = "data/your_training_file.csv"
```

**2. Evaluation Data Setup:**

Download or prepare evaluation datasets and place in `data/multi-data/`:

Required files:
- `banking_train.csv`, `banking77_test.csv`, `banking77_validation.csv`
- `emotion_train.csv`, `emotion_test.csv`, `emotion_validation.csv`
- `tweet_train.csv`, `tweet_test.csv`, `tweet_validation.csv`
- `sick_test.csv`, `sick_validation.csv`
- `sts12_test.csv`, `sts12_validation.csv`
- `stsb_test.csv`, `stsb_validation.csv`
- `mrpc_test.csv`, `mrpc_validation.csv`
- `scitail_test.csv`, `scitail_validation.csv`
- `wic_test.csv`, `wic_validation.csv`
- `qnli_test.csv`, `qnli_validation.csv`
- `rte_test.csv`, `rte_validaion.csv`

**3. Model Hub Setup (Optional):**

If using local models instead of HuggingFace Hub, place model files in `model_hub/`:

```
model_hub/
├── MiniLMv2-L6-H384-distilled-from-BERT-Base/
│   └── MiniLM-L6-H384-distilled-from-BERT-Base/
│       └── config.json
├── MiniLMv2-L6-H768-distilled-from-BERT-Base/
└── MiniLMv2-L6-H768-distilled-from-BERT-Large/
```

Update model paths in training scripts:
```powershell
$STUDENT_MODEL = "model_hub/MiniLM-L6-H384-distilled-from-BERT-Base"
$TEACHER_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # Or local path
```

## File Structure

```
TALAS/
├── main.py                      # Main entry point for training
├── distiller.py                 # Unified training engine for all KD methods
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
├── .gitignore                   # Git ignore rules
│
├── config/                      # Configuration files for each method
│   ├── __init__.py
│   ├── base_config.py           # Base configuration class
│   ├── talas_config.py          # TALAS method configuration
│   ├── dskd_config.py           # DSKD method configuration
│   ├── cdm_config.py            # CDM method configuration
│   ├── emo_config.py            # EMO method configuration
│   └── stella_config.py         # Stella method configuration
│
├── src/                         # Core source code modules
│   ├── __init__.py
│   ├── loss.py                  # Loss functions (InfoNCE, cosine, triplet)
│   ├── pooling.py               # Pooling utilities (mean, CLS pooling)
│   ├── cache_teacher.py         # Teacher embedding caching utilities
│   │
│   ├── data_utils/              # Dataset and data loading
│   │   ├── __init__.py
│   │   ├── dataset.py           # TextPairRaw dataset
│   │   └── dataset_cache.py     # TextPairWithTeacher (for TALAS)
│   │
│   ├── criterions/              # Knowledge Distillation implementations
│   │   ├── __init__.py
│   │   ├── teacher_anchor_kd.py           # TALAS implementation
│   │   ├── dual_space_kd.py               # DSKD implementation
│   │   ├── contextual_dynamic_mapping.py  # CDM implementation
│   │   ├── emo_embedding_distillation.py  # EMO implementation
│   │   └── stella_distillation.py         # Stella implementation
│   │
│   └── evaluation/              # Evaluation metrics and tasks
│       ├── __init__.py
│       ├── evaluation_automodel.py        # AutoModel-based evaluation
│       └── evaluation_model_define.py     # Custom model evaluation
│
├── scripts/                     # Training shell scripts
│   ├── train_talas.ps1          # TALAS training (PowerShell)
│   ├── train_talas.sh           # TALAS training (Bash)
│   ├── train_dskd.ps1           # DSKD training (PowerShell)
│   ├── train_dskd.sh            # DSKD training (Bash)
│   ├── train_cdm.ps1            # CDM training (PowerShell)
│   ├── train_cdm.sh             # CDM training (Bash)
│   ├── train_emo.ps1            # EMO training (PowerShell)
│   ├── train_emo.sh             # EMO training (Bash)
│   ├── train_stella.ps1         # Stella training (PowerShell)
│   ├── train_stella.sh          # Stella training (Bash)
│   ├── checkpoints/             # Saved model checkpoints (gitignored)
│   └── cache/                   # Cached embeddings (gitignored)
│
├── data/                        # Training and evaluation data (gitignored)
│   ├── merged_3_data_5k_each.csv
│   ├── test_debug.csv
│   └── multi-data/              # Evaluation datasets
│       ├── banking77_*.csv
│       ├── emotion_*.csv
│       ├── tweet_*.csv
│       ├── sick_*.csv
│       ├── sts12_*.csv
│       ├── stsb_*.csv
│       ├── mrpc_*.csv
│       ├── scitail_*.csv
│       └── wic_*.csv
│
└── model_hub/                   # Local model storage (gitignored)
    ├── MiniLMv2-L6-H384-distilled-from-BERT-Base/
    ├── MiniLMv2-L6-H768-distilled-from-BERT-Base/
    └── MiniLMv2-L6-H768-distilled-from-BERT-Large/
```

## Installation


### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dependencies

- **Deep Learning**: PyTorch >= 2.0.0
- **Transformers**: transformers, huggingface_hub, tokenizers, PEFT
- **Data Processing**: pandas, numpy
- **ML/Metrics**: scikit-learn, scipy
- **String Matching**: Levenshtein, editdistance, fastdtw
- **Utils**: tqdm, kagglehub, datasets

## Supported Tasks

### 1. Semantic Textual Similarity (STS)
- SICK, STS12, STSb datasets
- Metric: Spearman correlation

### 2. Text Classification
- Banking77, Emotion, Tweet datasets
- Metric: Accuracy, F1-score (macro)

### 3. Pair Classification
- MRPC, SciTail, WiC datasets
- Metric: Accuracy, F1, Precision, Recall, Average Precision

## Usage

### Quick Start

#### 1. Prepare Data

Place training CSV file in `data/` directory:
```csv
# Format: CSV with 'text' column or 'premise', 'hypothesis' columns
# Example: data/AllNLI.csv, data/merged_3_data_5k_each.csv
```

**Important:** Training scripts use `test_debug.csv` by default for quick testing. Change to your actual training data:

**Edit training scripts before running:**

For PowerShell (`.ps1` files):
```powershell
# In scripts/train_talas.ps1, change line:
$TRAIN_DATA = "data/test_debug.csv"
# To your actual data file:
$TRAIN_DATA = "data/merged_3_data_5k_each.csv"
```

For Bash (`.sh` files):
```bash
# In scripts/train_talas.sh, change line:
TRAIN_DATA="../data/test_debug.csv"
# To your actual data file:
TRAIN_DATA="../data/merged_3_data_5k_each.csv"
```


#### 2. Run Training

**Windows PowerShell:**
```powershell
# TALAS method (our method)
cd scripts
.\train_talas.ps1

# Other methods
.\train_dskd.ps1
.\train_cdm.ps1
.\train_emo.ps1
.\train_stella.ps1
```

**Linux/Mac:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# TALAS method
cd scripts
./train_talas.sh

# Other methods
./train_dskd.sh
./train_cdm.sh
./train_emo.sh
./train_stella.sh
```

### TALAS training

If you change data, delete cache first to re-calculate cache teacher.

### Baseline training

When using Qwen3 as the teacher model, apply last-token pooling to obtain the sentence representation. For BGE-M3 as the teacher, use the CLS token for pooling.

### Precision

The training precision can be switched between float16 and float32 as required.

### Hyperparameter Tuning

You can access the corresponding config file (e.g., `config/talas_config.py`, `config/stella_config.py`) to adjust hyperparameters such as learning rate, batch size, epochs, loss weights, and other training configurations.


## Evaluation

```python
# Evaluate STS tasks
eval_sts_task(model, test_sts_tasks)

# Evaluate classification tasks
eval_classification_task(model, test_cls_tasks)

# Evaluate pair classification tasks
eval_pair_task(model, test_pair_tasks)
```

