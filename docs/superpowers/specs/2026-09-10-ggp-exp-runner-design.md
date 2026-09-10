# GGP Experiment 1–4 Runner Design

## Goal

Restore the missing shared experiment harness and run the new GGP experiment
scripts on `H200_ANNP36` in `/home/annp36/work/heatgeo`, using only GPU IDs
0, 1, 2, and 3. Training sweeps must keep four independent runs in flight
whenever at least four runnable tasks remain.

## Existing problem

`exp1_batch_intervention.sh`, `exp2_support_intervention.sh`, and
`exp4_exposure_scaling.sh` source `scripts/exp/lib/run_arms.sh`, but that file is
absent from commit `53bb1a1`. The repository-wide Python `.gitignore` rule
`lib/` also ignores the intended directory, which explains how the scripts and
README were committed without their harness.

The public interface expected by the scripts is:

- `GRAPH_SPEC`: newline-delimited `graph_key|corpus|graph_flags` records.
- `ARMS_SPEC`: newline-delimited `label|graph_key|method|training_flags` records.
- `PAIR`, `SEEDS`, `GPUS`, `RUN_ID`, `RESULT_BASE`, `CACHE_ROOT`, `CSV_OUT`, and
  `DRY_RUN` environment settings.
- `run_arms <repo_root> <experiment>` as the single entry point.

## Chosen approach

Implement `run_arms.sh` as a generalization of the already-used scheduler in
`scripts/ggpkd/sensitivity.sh`. Run the four top-level scripts in dependency
order:

1. E1: 27 training runs.
2. E2: 15 training runs for the default model pair.
3. E3: post-hoc scoring of the E2 checkpoints; no training.
4. E4: 84 training runs.

Each training sweep uses a fixed four-worker GPU pool, one worker per GPU. The
experiments themselves are not run concurrently. This avoids oversubscribing a
GPU, eliminates cross-controller cache races, and preserves E3's dependency on
E2.

## Runner behavior

### Validation and parsing

Before creating outputs, the runner validates:

- the project virtual-environment Python exists and is executable;
- all requested GPU IDs are non-negative integers and unique;
- seeds are non-negative integers;
- graph keys and arm labels are unique;
- every arm references a declared graph key;
- methods are limited to the training script's supported values;
- an existing run root is never overwritten.

`DRY_RUN=1` parses and prints the complete matrix without building caches or
starting training.

### Output layout

For `<RESULT_BASE>/<RUN_ID>` the runner writes:

```text
run_config.tsv
manifest.tsv
stats.tsv
controller.exit
logs/
status/
graph_setup/
graph_logs/
runs/<arm>/seed_<seed>/
```

The default result base is `results/<experiment>`. The successful run root is
written atomically to `runs/<experiment>/last_run_root.txt`, which is the input
contract used by E3. `export_runs.py` produces the requested CSV after all
training tasks have exited successfully.

### Cache construction

Graphs are prepared serially on GPU 0 before training. The teacher embedding
cache is shared by graph variants for the same pair and corpus. Cache paths are
scoped by pair and corpus key:

```text
<CACHE_ROOT>/<pair>/<corpus_key>/teacher_train.pt
<CACHE_ROOT>/<pair>/<corpus_key>/graph_<graph_key>.pt
```

Serial preparation prevents multiple processes from writing the same teacher
cache. Training treats all completed cache artifacts as read-only.

### Training scheduling

The runner expands the Cartesian product of unique arms and seeds. It launches
at most one task per requested GPU. When a task exits, the next queued task is
immediately assigned to that same GPU. Thus four runs remain active until fewer
than four tasks remain.

Every task receives explicit `GPU`, `PYTHON_BIN`, cache paths, log paths, save
paths, and weights paths. It runs through `scripts/ggpkd/train.sh` with
`--final_weights_only --eval_every 0`, plus the graph and arm flags.

### Failures and resumability

Each task writes a numeric exit file and resource statistics. One failed task
does not terminate other active tasks, but the controller exits nonzero after
the current sweep and refuses aggregation. Logs and partial outputs remain for
diagnosis. The first implementation intentionally does not silently retry a
failed scientific run; the operator inspects and fixes the cause, then launches
a new run ID so provenance stays unambiguous.

Signals received by the controller are propagated to all active child process
groups before exit so a stopped sweep does not leave orphan GPU processes.

## Deployment and execution

The local branch is migrated to the existing remote checkout while preserving
the remote `.venv` and durable cache/output directories. Before launching:

1. verify local and remote commit hashes;
2. run `bash -n` on all experiment scripts and the new harness;
3. run focused experiment tests with the project `.venv`;
4. confirm `pip check` succeeds without installing global packages;
5. confirm GPUs 0–3 are free and do not touch processes on GPUs 4–7;
6. perform dry runs for E1, E2, and E4 and verify counts 27, 15, and 84.

A detached `tmux` controller runs E1–E4 in order and writes a top-level log.
Health checks inspect the controller, child PIDs, GPU utilization, task exit
files, and recent logs. ETA is calculated after completed training samples are
available, using observed wall time and the remaining queued GPU waves rather
than a speculative pre-launch estimate.

## Verification criteria

The implementation is accepted when:

- shell syntax checks pass;
- dry-run matrices report exactly 27, 15, and 84 training runs;
- a reduced smoke matrix schedules one task per GPU without path collisions;
- all run directories and exit files agree with the manifest;
- E2 writes `last_run_root.txt` and E3 resolves it;
- aggregation returns one row per arm and seed with no failed status;
- remote GPU 0–3 each have at most one experiment process and GPU 4–7 remain
  untouched.

