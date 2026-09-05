#!/usr/bin/env bash
# Shared, locked protocol for the paper ablations. Source this file; do not run it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Paper setting. Override from the environment only when intentionally starting
# a separate experiment root.
PAIR_KEY="${PAIR_KEY:-qwen3_0_6b_to_minilmv2_h384}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
STUDENT_MODEL="${STUDENT_MODEL:-nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base}"
POOLING_METHOD="${POOLING_METHOD:-last_token}"
TRAIN_DATA="${TRAIN_DATA:-data/train_set/merged_3_data_5k_each.csv}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEEDS="${SEEDS:-42 43 44}"

# Pin every method-defining value so all arms stay comparable if config defaults
# later move. The canonical support quota remains derived from the graph.
GRAPH_K="${GRAPH_K:-200}"
PERPLEXITY="${PERPLEXITY:-30}"
TRUNCATION_TOLERANCE="${TRUNCATION_TOLERANCE:-0.01}"
# The paper method is the one-hop graph transition objective. Multi-hop variants
# are isolated in radius.sh and never silently become the reference arm.
DIFFUSION_SCALES="${DIFFUSION_SCALES:-1}"
ROW_WEIGHT="${ROW_WEIGHT:-1.0}"
ROW_START_EPOCH="${ROW_START_EPOCH:-1}"
DIRECT_TEMP="${DIRECT_TEMP:-0}"
HARD_NEG_K="${HARD_NEG_K:-40}"
RANDOM_NEG_K="${RANDOM_NEG_K:-26}"

EXPERIMENT_KEY="${EXPERIMENT_KEY:-paper_r1_v2}"
RUNS_BASE="${RUNS_BASE:-runs/ablation}"
CACHE_BASE="${CACHE_BASE:-cache/ggpkd}"
LOG_BASE="${LOG_BASE:-logs/ggpkd}"
ABL_ROOT="${ABL_ROOT:-${RUNS_BASE}/${PAIR_KEY}/${EXPERIMENT_KEY}}"
CACHE_ROOT="${CACHE_ROOT:-${CACHE_BASE}/${PAIR_KEY}/${EXPERIMENT_KEY}}"
LOG_ROOT="${LOG_ROOT:-${LOG_BASE}/${PAIR_KEY}/${EXPERIMENT_KEY}}"
TEACHER_CACHE="${TEACHER_CACHE:-${CACHE_BASE}/${PAIR_KEY}/teacher_train.pt}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" && "${PYTHON_BIN}" == "python3" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

base_graph_path() {
    printf '%s\n' "${CACHE_ROOT}/graph_base.pt"
}

# run_arm <group> <arm> <seed> [main.py overrides ...]
#
# A graph-changing arm must set GRAPH_KEY immediately before this call. All
# other arms share graph_base.pt. A completed arm is represented by one compact
# result.csv; interrupted runs retain their temporary diagnostics and are rerun
# with a fresh run_id, which the repository telemetry keeps distinguishable.
run_arm() {
    local group="$1"
    local arm="$2"
    local seed="$3"
    shift 3

    local graph_key="${GRAPH_KEY:-base}"
    unset GRAPH_KEY || true

    local save_dir="${ABL_ROOT}/${group}/${arm}/seed${seed}"
    local weights_dir="${save_dir}/weights"
    local graph_cache="${CACHE_ROOT}/graph_${graph_key}.pt"
    local graph_log="${LOG_ROOT}/graph_${graph_key}"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry ] ${group}/${arm}/seed${seed} | GPU=${CUDA_VISIBLE_DEVICES} graph=${graph_key}"
        echo "       overrides: $*"
        return 0
    fi

    # Different group scripts can request the same shared arm concurrently.
    # flock is process-scoped and automatically releases on exit, including a
    # failed or interrupted training process, so it cannot leave a stale lock.
    local lock_dir="${ABL_ROOT}/.locks"
    local lock_path="${lock_dir}/${group}_${arm}_seed${seed}.lock"
    local lock_fd
    mkdir -p "${lock_dir}"
    exec {lock_fd}>"${lock_path}"
    flock "${lock_fd}"

    # A path is not an experiment identity. Materialize the complete requested
    # protocol (plus a fingerprint of executable training code) before deciding
    # whether an existing compact result is reusable.
    mkdir -p "${save_dir}"
    local pending_request="${save_dir}/.request.pending.${BASHPID}"
    "${PYTHON_BIN}" - "${pending_request}" "${REPO_ROOT}" \
        "${PAIR_KEY}" "${EXPERIMENT_KEY}" "${group}" "${arm}" "${seed}" \
        "${graph_key}" "${TRAIN_DATA}" "${TEACHER_MODEL}" "${STUDENT_MODEL}" \
        "${POOLING_METHOD}" "${BATCH_SIZE}" "${EPOCHS}" "${LR}" \
        "${MAX_LENGTH}" "${NUM_WORKERS}" "${GRAPH_K}" "${PERPLEXITY}" \
        "${TRUNCATION_TOLERANCE}" "${DIFFUSION_SCALES}" "${ROW_WEIGHT}" \
        "${ROW_START_EPOCH}" "${DIRECT_TEMP}" "${HARD_NEG_K}" \
        "${RANDOM_NEG_K}" "$@" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

(
    output, repo_root, pair, experiment, group, arm, seed, graph_key,
    train_data, teacher, student, pooling, batch_size, epochs, learning_rate,
    max_length, num_workers, graph_k, perplexity, truncation_tolerance,
    diffusion_scales, row_weight, row_start_epoch, direct_temp, hard_neg_k,
    random_neg_k, *extra_args,
) = sys.argv[1:]

root = Path(repo_root)
code_files = [root / "main.py", root / "distiller.py"]
for directory in (root / "config", root / "src"):
    code_files.extend(sorted(directory.rglob("*.py")))
code_files.extend(sorted((root / "scripts" / "ablation").glob("*.sh")))
digest = hashlib.sha256()
for path in sorted({path.resolve() for path in code_files if path.is_file()}):
    digest.update(str(path.relative_to(root.resolve())).encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")

payload = {
    "schema": 1,
    "pair": pair,
    "experiment": experiment,
    "group": group,
    "arm": arm,
    "seed": int(seed),
    "graph_key": graph_key,
    "code_sha256": digest.hexdigest(),
    "protocol": {
        "train_data": train_data,
        "teacher_model": teacher,
        "student_model": student,
        "pooling_method": pooling,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "max_length": int(max_length),
        "num_workers": int(num_workers),
        "graph_k": int(graph_k),
        "perplexity": float(perplexity),
        "truncation_tolerance": float(truncation_tolerance),
        "diffusion_scales": diffusion_scales,
        "row_weight": float(row_weight),
        "row_start_epoch": int(row_start_epoch),
        "direct_temp": float(direct_temp),
        "hard_neg_k": int(hard_neg_k),
        "random_neg_k": int(random_neg_k),
        "eval_every": 0,
        "final_weights_only": True,
        "extra_args": extra_args,
    },
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PYEOF

    if [[ -f "${save_dir}/result.csv" ]]; then
        if ! "${PYTHON_BIN}" - "${save_dir}/result.csv" "${pending_request}" <<'PYEOF'
import csv
import hashlib
import sys
from pathlib import Path

result_path, request_path = map(Path, sys.argv[1:])
with result_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
actual = rows[0].get("request_sha256") if len(rows) == 1 else None
expected = hashlib.sha256(request_path.read_bytes()).hexdigest()
raise SystemExit(0 if actual == expected else 1)
PYEOF
        then
            echo "[fail] ${group}/${arm}/seed${seed}: result.csv belongs to a different protocol" >&2
            echo "       use a new EXPERIMENT_KEY; refusing to reuse or overwrite ${save_dir}" >&2
            rm -f "${pending_request}"
            flock -u "${lock_fd}"
            exec {lock_fd}>&-
            return 2
        fi
        rm -f "${pending_request}"
        echo "[skip] ${group}/${arm}/seed${seed} already complete (CSV protocol verified)"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 0
    fi

    if [[ -f "${save_dir}/.done" ]]; then
        echo "[fail] ${group}/${arm}/seed${seed}: legacy .done run is not compacted" >&2
        echo "       use a new EXPERIMENT_KEY; refusing to mix artifact formats" >&2
        rm -f "${pending_request}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 2
    fi

    mv "${pending_request}" "${save_dir}/request.json"
    mkdir -p "${weights_dir}" "$(dirname "${graph_cache}")" "${graph_log}"
    echo "[run ] ${group}/${arm}/seed${seed} | GPU=${CUDA_VISIBLE_DEVICES} graph=${graph_key}"
    echo "       overrides: $*"

    local started_at
    local elapsed
    local start_seconds=${SECONDS}
    local teacher_cache_warm_before=0
    local graph_cache_warm_before=0
    [[ -f "${TEACHER_CACHE}" ]] && teacher_cache_warm_before=1
    [[ -f "${graph_cache}" ]] && graph_cache_warm_before=1
    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    "${PYTHON_BIN}" main.py \
        --method ggpkd \
        --train_data "${TRAIN_DATA}" \
        --student_model "${STUDENT_MODEL}" \
        --teacher_model "${TEACHER_MODEL}" \
        --pooling_method "${POOLING_METHOD}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --eval_every 0 \
        --lr "${LR}" \
        --max_length "${MAX_LENGTH}" \
        --num_workers "${NUM_WORKERS}" \
        --seed "${seed}" \
        --graph_k "${GRAPH_K}" \
        --perplexity "${PERPLEXITY}" \
        --truncation_tolerance "${TRUNCATION_TOLERANCE}" \
        --diffusion_scales "${DIFFUSION_SCALES}" \
        --row_weight "${ROW_WEIGHT}" \
        --row_start_epoch "${ROW_START_EPOCH}" \
        --direct_temp "${DIRECT_TEMP}" \
        --hard_neg_k "${HARD_NEG_K}" \
        --random_neg_k "${RANDOM_NEG_K}" \
        --support_policy topk \
        --relation_target diffusion \
        --cache_path "${TEACHER_CACHE}" \
        --ggpkd_cache_path "${graph_cache}" \
        --ggpkd_log_dir "${graph_log}" \
        --save_dir "${save_dir}" \
        --weights_dir "${weights_dir}" \
        --final_weights_only \
        "$@" 2>&1 | tee "${save_dir}/train.log"

    elapsed=$(( SECONDS - start_seconds ))
    # main.py historically treats evaluation errors as warnings. A paper run is
    # not complete without the final-model test record, so validate it before
    # compacting the run; this keeps a transient evaluator failure
    # resumable instead of permanently marking an unusable directory complete.
    "${PYTHON_BIN}" - "${save_dir}" <<'PYEOF'
import json
import sys
from pathlib import Path

save_dir = Path(sys.argv[1])
manifest_path = save_dir / "run.json"
if not manifest_path.exists():
    raise SystemExit(f"missing run manifest after training: {manifest_path}")
run_id = json.loads(manifest_path.read_text())["run_id"]
found = False
for filename in ("metrics.jsonl", "epochs.jsonl"):
    path = save_dir / filename
    if not path.exists():
        continue
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_id") == run_id and isinstance(row.get("test"), dict):
            found = True
if not found:
    raise SystemExit(
        f"final-model evaluation is missing for run_id={run_id} in {save_dir}"
    )
PYEOF
    "${PYTHON_BIN}" - "${save_dir}" "${group}" "${arm}" "${seed}" \
        "${graph_key}" "${elapsed}" "${started_at}" \
        "${teacher_cache_warm_before}" "${graph_cache_warm_before}" "$@" <<'PYEOF'
import json
import sys

(
    save_dir, group, arm, seed, graph_key, elapsed, started_at,
    teacher_warm, graph_warm, *extra,
) = sys.argv[1:]
with open(f"{save_dir}/arm.json", "w", encoding="utf-8") as handle:
    json.dump(
        {
            "ablation": group,
            "group": group,
            "arm": arm,
            "seed": int(seed),
            "graph_key": graph_key,
            "wall_clock_seconds": int(elapsed),
            "teacher_cache_warm_before": bool(int(teacher_warm)),
            "graph_cache_warm_before": bool(int(graph_warm)),
            "timing_scope": "launcher end-to-end: init + cache build/load + training + final evaluation",
            "started_at": started_at,
            "extra_args": extra,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
PYEOF
    "${PYTHON_BIN}" scripts/ablation/compact_run.py \
        --run-dir "${save_dir}" \
        --experiment "${EXPERIMENT_KEY}" \
        --graph-log-dir "${graph_log}"
    flock -u "${lock_fd}"
    exec {lock_fd}>&-
    echo "[done] ${group}/${arm}/seed${seed} in ${elapsed}s -> result.csv"
}

run_full() {
    local seed="$1"
    if [[ -n "${CANONICAL_QUOTA:-}" ]]; then
        GRAPH_KEY=base run_arm full full "${seed}" \
            --diffusion_quota "${CANONICAL_QUOTA}"
    else
        GRAPH_KEY=base run_arm full full "${seed}"
    fi
}

canonical_quota() {
    if [[ -n "${CANONICAL_QUOTA:-}" ]]; then
        printf '%s\n' "${CANONICAL_QUOTA}"
        return 0
    fi
    "${PYTHON_BIN}" scripts/ablation/budget.py \
        --artifact "$(base_graph_path)" \
        --hard "${HARD_NEG_K}" \
        --random "${RANDOM_NEG_K}" \
        --format quota
}

canonical_width() {
    if [[ -n "${CANONICAL_QUOTA:-}" ]]; then
        printf '%s\n' "$(( CANONICAL_QUOTA + HARD_NEG_K + RANDOM_NEG_K ))"
        return 0
    fi
    "${PYTHON_BIN}" scripts/ablation/budget.py \
        --artifact "$(base_graph_path)" \
        --hard "${HARD_NEG_K}" \
        --random "${RANDOM_NEG_K}" \
        --format width
}

# Extreme deletion shared by Tables 2 and 3. Candidates are uniform over the
# corpus, the graph objective is absent, and L_row is absent.
# Keeping the canonical candidate width prevents the deletion from winning or
# losing merely because it encoded a different number of columns per anchor.
run_no_graph_support() {
    local seed="$1"
    local width
    width="$(canonical_width)"
    GRAPH_KEY=base run_arm shared no_graph_support "${seed}" \
        --relation_target ambient_only \
        --row_weight 0 \
        --diffusion_quota 0 \
        --hard_neg_k 0 \
        --random_neg_k "${width}"
}

# Shared between the component table and row-weight sensitivity (lambda=0).
run_no_row() {
    local seed="$1"
    GRAPH_KEY=base run_arm components no_row "${seed}" --row_weight 0
}

echo "protocol: ${TEACHER_MODEL} -> ${STUDENT_MODEL} | lr=${LR} epochs=${EPOCHS} bs=${BATCH_SIZE}"
echo "seeds: [${SEEDS}] | output: ${ABL_ROOT}"
