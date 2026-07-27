# Data Splits, Evaluation Tables, and Google Drive Weights Design

## Goal

Reorganize the repository datasets into explicit train, validation, and test
directories; eliminate classification train-validation leakage; update all
training and evaluation paths; print validation and test results as separate
tables; and persist student weights after every epoch to Google Drive.

The reproduction target remains:

- teacher: `Qwen/Qwen3-Embedding-4B`;
- student: `google-bert/bert-base-uncased`;
- method: HeatGeo.

## Scope

This change covers:

- the physical CSV layout under `data/`;
- classification split repair for Banking77 and Tweet;
- evaluation task path definitions;
- validation and final-test table formatting and CSV exports;
- per-epoch student weight persistence;
- the HeatGeo shell script and Colab notebook configuration;
- documentation and focused verification.

It does not change the HeatGeo loss, optimizer, model architecture, benchmark
metrics, or teacher-student pair.

## Dataset Layout

The final layout is:

```text
data/
├── train_set/
│   ├── merged_3_data_5k_each.csv
│   ├── banking77_train.csv
│   ├── emotion_train.csv
│   └── tweet_train.csv
├── val_set/
│   ├── banking77_validation.csv
│   ├── emotion_validation.csv
│   ├── tweet_validation.csv
│   ├── mrpc_validation.csv
│   ├── qnli_validation.csv
│   ├── rte_validation.csv
│   ├── scitail_validation.csv
│   ├── sick_validation.csv
│   ├── sts12_validation.csv
│   ├── stsb_validation.csv
│   └── wic_validation.csv
└── test_set/
    ├── banking77_test.csv
    ├── emotion_test.csv
    ├── mrpc_test.csv
    ├── qnli_test.csv
    ├── rte_test.csv
    ├── scitail_test.csv
    ├── sick_test.csv
    ├── sts12_test.csv
    ├── stsb_test.csv
    ├── tweet_test.csv
    └── wic_test.csv
```

The old `data/multi-data/` directory is removed only after every source file
has a verified destination and the repaired split checks pass.

## Classification Split Repair

### Banking77

The current `banking77_validation.csv` contains 1,000 rows that are also
present in the 10,003-row `banking_train.csv`. The validation file is retained
as the authoritative validation split. Its rows are removed from the training
file using normalized text as the identity key.

Expected result:

- train: 9,003 rows;
- validation: 1,000 rows;
- train-validation text overlap: zero.

### Emotion

The current Emotion train and validation files have no normalized-text overlap.
Their contents are retained unchanged and only moved.

Expected result:

- train: 15,956 rows;
- validation: 1,988 rows;
- train-validation text overlap: zero.

### Tweet

The current 26,732-row `tweet_train.csv` contains every row of the current
23,732-row `tweet_validation.csv`. The current validation file is therefore
treated as the intended training subset. The new validation split starts with
the rows from the current full train file that are not present in that subset.
Six rows in the intended training subset normalize to the same text as three
new validation rows, so those six rows are also moved to validation. This keeps
every original ID while making normalized train and validation text disjoint.

Tweet identity is determined by `id` when it is present and non-null, with
normalized text as a defensive cross-check. Any duplicate identity with
conflicting labels is a hard error.

Expected result:

- train: 23,726 rows;
- validation: 3,006 rows;
- train-validation ID and normalized-text overlap: zero;
- the union of the repaired train and validation IDs equals the original full
  train IDs.

### Leakage Guard

Before classification evaluation, the evaluator validates that train and
validation examples are disjoint. For normalized text keys:

\[
\mathcal{K}_{\mathrm{train}}
\cap
\mathcal{K}_{\mathrm{validation}}
=
\varnothing
\]

A non-empty train-validation intersection raises a descriptive error naming
the benchmark and overlap count. Test data is also checked against the repaired
train and validation splits for exact identity overlap. Because the published
Banking77 split contains a small number of repeated utterances, test overlap is
reported as an explicit warning with counts rather than blocking evaluation.

## Evaluation Protocol

### During Training

After every epoch:

1. evaluate validation benchmarks only;
2. select pair-classification thresholds on validation;
3. print one table titled `VALIDATION - EPOCH N`;
4. append the structured result to `metrics.jsonl`;
5. save the local full checkpoint and the Google Drive student weights.

Test benchmarks are not evaluated per epoch and cannot be used to select an
epoch.

### After Training

After the final epoch:

1. reuse the pair thresholds selected by the final validation evaluation;
2. evaluate every test benchmark exactly once;
3. print one table titled `FINAL TEST`;
4. append the structured test result to `metrics.jsonl`.

The notebook exports:

- `validation_by_epoch.csv`: one row per validation benchmark and family mean
  for each epoch;
- `final_test_results.csv`: one row per final test benchmark and family mean.

### Metrics

The primary comparison metric for each family matches TALAS:

| Family | Primary metric |
|---|---|
| Classification | Macro-F1 |
| Pair classification | Average Precision |
| Semantic textual similarity | Spearman correlation |

Accuracy, precision, recall, selected pair threshold, and other already
available metrics remain in the tables when applicable for auditability.

### Console Table

The training process prints compact, dependency-free plain-text tables with:

- split and epoch/finality in the title;
- family;
- benchmark;
- primary metric name;
- primary metric value on a 0-100 scale;
- additional available metrics on the same 0-100 scale.

Raw metrics in `metrics.jsonl` and CSV exports remain on the original 0-1 scale
to avoid changing the existing data contract.

## Path Updates

All references to the old paths are updated:

- HeatGeo training corpus:
  `data/train_set/merged_3_data_5k_each.csv`;
- classification probe training files:
  `data/train_set/`;
- validation files:
  `data/val_set/`;
- test files:
  `data/test_set/`.

Both evaluation modules that currently define task lists are updated so that no
legacy `data/multi-data/` path remains. Configuration defaults, shell scripts,
the notebook, and README examples use the new training path.

## Google Drive Student Weights

The Colab notebook defines:

```text
/content/drive/MyDrive/[ICLR] Embedding KD/
└── weights/
    └── qwen3_4b_to_bert_base/
        ├── student_epoch_1.pt
        ├── student_epoch_2.pt
        ├── student_epoch_3.pt
        ├── student_epoch_4.pt
        └── student_epoch_5.pt
```

Each file contains:

```python
{
    "epoch": 1,
    "student_model_name": "google-bert/bert-base-uncased",
    "teacher_model_name": "Qwen/Qwen3-Embedding-4B",
    "model_state_dict": {...},
}
```

Epoch numbers stored in files and filenames are one-based.

## Weight-Saving Flow

A new optional `weights_dir` configuration value is exposed through:

1. `main.py` as `--weights_dir`;
2. `scripts/train_heatgeo.sh` as the `WEIGHTS_DIR` environment override;
3. the Colab notebook as
   `DRIVE_DIR / "weights" / "qwen3_4b_to_bert_base"`.

When `weights_dir` is provided, each epoch:

1. serialize the student weight payload to a local temporary file;
2. ensure the Drive destination directory exists;
3. copy the temporary file to a temporary destination name in Drive;
4. atomically replace the final epoch filename where supported;
5. verify that the final file exists and has non-zero size;
6. remove the local temporary file.

An incomplete copy never uses the final `student_epoch_N.pt` name. A copy or
verification failure stops training with a descriptive error because durable
per-epoch weights are an explicit requirement.

The existing full checkpoints, optimizer state, scheduler state, logs,
`metrics.jsonl`, and evaluation CSV files remain in the Colab workspace run
directory. The final epoch is not saved twice.

## Notebook Flow Changes

The notebook:

1. mounts Google Drive and validates `[ICLR] Embedding KD`;
2. defines and creates the dedicated Drive weights directory;
3. resolves the new training data path under `data/train_set/`;
4. validates that all required train, validation, and test files exist;
5. passes `WEIGHTS_DIR` to `scripts/train_heatgeo.sh`;
6. prints the local run directory and Drive weights directory before training;
7. verifies the expected five non-empty epoch weight files after training;
8. parses validation and final-test records separately;
9. displays separate validation and final-test tables;
10. exports `validation_by_epoch.csv` and `final_test_results.csv`.

The notebook does not delete the Drive weights directory automatically.
Existing epoch filenames are replaced only by a successful new copy for the
same reproduction target.

## Error Handling

The workflow fails early for:

- missing split directories or required CSV files;
- classification train-validation identity overlap;
- conflicting labels for the same identity;
- incomplete Tweet split reconstruction;
- missing Google Drive mount or weights directory creation failure;
- missing or empty epoch weight files;
- a missing validation record for an epoch;
- a missing final test record.

Validation failures do not silently fall back to test data.
Classification test identity overlap is printed as a warning with the affected
benchmark and count.

## Verification

Implementation verification will use the project `.venv` and will not install
packages globally.

Checks include:

1. verify the expected final data tree and file counts;
2. verify Banking77, Emotion, and Tweet train-validation disjointness;
3. verify repaired Tweet train-validation union against the original source
   identity set during migration;
4. verify no evaluator, config, script, notebook, or README reference uses
   `data/multi-data/` or the old root training CSV path;
5. test evaluation-table formatting using synthetic nested validation and test
   results;
6. verify validation executes per epoch and test executes only once after the
   final epoch;
7. verify `save_every = 1` produces one local checkpoint and one student weight
   payload per epoch without duplicating the final epoch;
8. verify a saved student weight payload contains the expected epoch, model
   identifiers, and model state dictionary;
9. parse `test_mdd.ipynb` as valid JSON and compile applicable Python cells;
10. run focused existing tests or syntax checks for all changed Python and shell
    files.

Full Qwen3-Embedding-4B to BERT-base training is not part of local verification
because it requires benchmark-scale compute and model downloads.
