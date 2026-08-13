# Two Remaining Random-Hard Runs Implementation Plan

Date: 2026-08-13
Design: `docs/superpowers/specs/2026-08-13-two-random-hard-runs-design.md`

## Goal and success criteria

Launch the two approved Qwen teacher/student random-hard direct runs on two
distinct idle H200 GPUs. Both runs must finish five epochs, save only epochs 3
and 5, finish final-test evaluation, and leave task-isolated models, artifacts,
and timestamped logs.

## Tasks

1. Add tests proving `--save_every 3` overrides `HeatGeoConfig.save_every` and
   non-positive values are rejected before training.
2. Expose `save_every` in `main.py` without changing the default for existing
   callers.
3. Extend the H200 launcher with the two new task names and pass the approved
   random-hard configuration plus `--save_every 3`.
4. Select the requested pair of GPUs from one `nvidia-smi` snapshot, requiring
   at least 100 GB free, low utilization, and distinct physical IDs.
5. Run the full local tests, Python compilation, shell syntax check, and DDP
   smoke test; commit only the scoped source, test, and documentation changes.
6. Sync the changed files to `/home/tensara/projects/ICLR-HeatGeo`, then repeat
   tests in the server project virtual environment.
7. Copy the two compatible teacher embedding caches into isolated artifact
   directories and verify source/destination SHA-256 checksums.
8. Recheck the live GPU snapshot and launch both tasks in one launcher call.
9. Monitor both processes through epoch 5 and final-test evaluation. Verify the
   exact checkpoint/weight schedule, finite training, output checksums, and idle
   GPUs after exit.

## Verification commands

Use the project `.venv` only:

```text
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q main.py config heatgeo distiller.py
bash -n scripts/launch_h200_jobs.sh
.venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29621 scripts/distributed_smoke.py
```

The remote equivalents run from
`/home/tensara/projects/ICLR-HeatGeo` using that directory's `.venv`.
