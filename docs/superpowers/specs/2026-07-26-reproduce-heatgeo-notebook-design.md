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

## Notebook Flow

1. Inspect NVIDIA GPU availability with `nvidia-smi` when present.
2. Mount Google Drive and locate the extracted repository by checking for
   `main.py` and `scripts/train_heatgeo.sh`.
3. Change into the detected project directory.
4. Create `.venv` only when it is absent, install `requirements.txt` through
   `.venv/bin/python`, and print the active PyTorch device:
   - CUDA when `torch.cuda.is_available()` is true;
   - Apple MPS when CUDA is unavailable and MPS is available;
   - CPU otherwise.
5. Remove only the HeatGeo cache and the selected run output directory so stale
   artifacts cannot contaminate a reproduction run.
6. Run `scripts/train_heatgeo.sh` with the current HeatGeo defaults and explicit
   experiment paths. The shell process will activate `.venv` first.
7. Rely on `KnowledgeDistiller.train()` with `eval_every = 1` to run validation
   after every epoch. Pair-classification thresholds are selected on validation.
8. Run test evaluation only once after the final epoch, reusing the final
   validation thresholds.
9. Read the run's `metrics.jsonl`, normalize every validation result into a
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

## Failure Handling

- Missing archive, repository, training dataset, HeatGeo script, virtual
  environment interpreter, training metrics, or epoch validation records will
  fail with a descriptive assertion.
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
4. run the summary-table parser against representative synthetic
   `metrics.jsonl` records using the project `.venv`;
5. confirm the resulting CSV schema and family mean rows.

Full model training is not part of local verification because it requires model
downloads and benchmark-scale compute. The notebook remains configured for a
Colab GPU reproduction run.
