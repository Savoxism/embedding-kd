# HeatGeo Reproduction Notebook Design

## Goal

Update `reproduce_talas.ipynb` so that it reproduces the benchmark workflow with
the current HeatGeo implementation, reports the actual PyTorch training device,
evaluates all validation benchmarks after every epoch, evaluates test benchmarks
only once after training, and produces a reusable evaluation summary table.

## Scope

Only `reproduce_talas.ipynb` will be changed during implementation. The notebook
will call the existing HeatGeo training entry point and consume the existing
`metrics.jsonl` output rather than duplicating the training loop from
`KnowledgeDistiller`.

The notebook will preserve the Colab and Google Drive workflow. It will use a
project-local `.venv` for dependency installation and training, and it will not
install packages into the global Python environment.

The source archive is fixed to:

`/content/drive/MyDrive/[ICLR] Embedding KD/ICLR_MDD_npa_test_1.zip`

Every notebook run will replace only the dedicated extraction workspace at
`/content/ICLR_MDD_npa_test_1_workspace`. This guarantees that the training run
cannot silently reuse an older extracted codebase.

The reproduction model pair is fixed in the notebook, independently of the
defaults stored in the uploaded ZIP:

- student: `google-bert/bert-base-uncased`;
- teacher: `Qwen/Qwen3-Embedding-4B`.

## Notebook Flow

1. Mount Google Drive and define `DRIVE_DIR`, `ARCHIVE_PATH`, and `EXTRACT_DIR`
   in one path-configuration cell.
2. Assert that `ICLR_MDD_npa_test_1.zip` exists, safely remove the dedicated
   extraction workspace, and extract the archive from scratch.
3. Discover the repository root inside the extraction workspace by requiring
   both `main.py` and `scripts/train_heatgeo.sh`. This supports archives with or
   without a single enclosing root directory.
4. Define the model IDs, dataset, virtual environment, HeatGeo cache, output,
   log, and metrics paths in one shared path cell and validate required inputs.
5. Remove any `.venv` accidentally included in the ZIP, create a clean
   project-local `.venv`, install `requirements.txt` through
   `.venv/bin/python`, and print the active PyTorch device:
   - CUDA when `torch.cuda.is_available()` is true;
   - Apple MPS when CUDA is unavailable and MPS is available;
   - CPU otherwise.
6. Remove only the HeatGeo cache and the selected run output directory so stale
   artifacts cannot contaminate a reproduction run.
7. Run `scripts/train_heatgeo.sh` with explicit model IDs and experiment paths.
   The shell process will activate `.venv` first. Environment overrides allow
   the existing uploaded ZIP to remain unchanged.
8. Rely on `KnowledgeDistiller.train()` with `eval_every = 1` to run validation
   after every epoch. Pair-classification thresholds are selected on validation.
9. Run test evaluation only once after the final epoch, reusing the final
   validation thresholds.
10. Read the run's `metrics.jsonl`, normalize every validation result into a
   single long-form table, append a `MEAN` row for each benchmark family and
   epoch, display the table, and save it as `evaluation_by_epoch.csv`.

## Evaluation Table Contract

Each row represents one benchmark result or a family mean. Columns are:

- `epoch`
- `family`: `classification`, `pair`, or `sts`
- `benchmark`: the CSV stem or `MEAN`
- `accuracy`
- `f1`
- `precision`
- `recall`
- `average_precision`
- `spearman`

Metrics that do not apply to a benchmark family remain empty. The table includes
validation records only because test data is reserved for the final evaluation.
The final test result remains available in `metrics.jsonl`.

## GPU Behavior

The notebook does not force a device. It reports and uses the same automatic
selection implemented by `KnowledgeDistiller`:

- two or more CUDA GPUs: student on `cuda:0`, teacher on `cuda:1`;
- one CUDA GPU: both initially on `cuda:0`;
- no CUDA and available Apple GPU: both on `mps`;
- otherwise: CPU.

For HeatGeo, the teacher is used to create or load cached embeddings and is then
released. Student training and the HeatGeo criterion remain on the selected
student device.

## Model-Specific Artifacts

The run directory is
`models/heatgeo/qwen3_4b_to_bert_base`. The notebook removes all extracted
`cache/heatgeo` contents before training. This prevents cache files whose names
reflect the codebase's older 0.6B/MiniLM defaults from being reused with the
Qwen3-4B/BERT-base pair. The uploaded ZIP and its source defaults are not patched
at runtime.

## Failure Handling

- Missing archive, repository, training dataset, HeatGeo script, virtual
  environment interpreter, training metrics, or epoch validation records will
  fail with a descriptive assertion.
- Archive members are checked to ensure they remain inside the dedicated
  extraction workspace before extraction.
- Cleanup targets are resolved and checked before deletion. No Google Drive
  directory or arbitrary `/content` directory is removed.
- A missing `nvidia-smi` command will be reported without stopping the notebook.
- The training shell uses `set -o pipefail`, so failures are not hidden by
  logging through `tee`.
- The summary parser ignores the final test-only JSONL record and validates that
  at least one per-epoch validation record exists.

## Verification

Implementation verification will:

1. parse the notebook as valid JSON;
2. compile every Python code cell that does not contain notebook shell or magic
   syntax;
3. inspect the notebook to confirm all TMKD-specific run paths and commands were
   replaced by HeatGeo equivalents;
4. inspect the notebook to confirm that the 0.6B/MiniLM model IDs are absent and
   the Qwen3-4B/BERT-base IDs are passed to the training script;
5. run the summary-table parser against representative synthetic
   `metrics.jsonl` records using the project `.venv`;
6. confirm the resulting CSV schema and family mean rows.

Full model training is not part of local verification because it requires model
downloads and benchmark-scale compute. The notebook remains configured for a
Colab GPU reproduction run.
