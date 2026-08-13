#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
TORCHRUN="${VENV_DIR}/bin/torchrun"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
MODEL_ROOT="${PROJECT_ROOT}/models"
ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts"
LOG_ROOT="${PROJECT_ROOT}/logs"
MIN_FREE_MB="${MIN_FREE_MB:-100000}"

if [[ ! -x "${TORCHRUN}" ]]; then
    echo "Missing project torchrun: ${TORCHRUN}" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required" >&2
    exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required to keep jobs alive after SSH disconnects" >&2
    exit 1
fi

mkdir -p "${MODEL_ROOT}" "${ARTIFACT_ROOT}" "${LOG_ROOT}"
MANIFEST="${LOG_ROOT}/launch_${RUN_STAMP}.tsv"
printf 'task\tphysical_gpu\tpid\tlog\tmodel_dir\tartifact_dir\n' > "${MANIFEST}"

launch_job() {
    local task="$1"
    local physical_gpu="$2"
    local teacher="$3"
    local student="$4"
    local teacher_pooling="$5"
    local master_port="$6"
    local free_mb
    local model_dir="${MODEL_ROOT}/${task}/${RUN_STAMP}"
    local artifact_dir="${ARTIFACT_ROOT}/${task}"
    local log_path="${LOG_ROOT}/${task}_${RUN_STAMP}.log"
    local tmux_session="heatgeo_${task}_${RUN_STAMP}"

    free_mb="$(
        nvidia-smi -i "${physical_gpu}" \
            --query-gpu=memory.free --format=csv,noheader,nounits \
            | tr -d '[:space:]'
    )"
    if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
        echo "Refusing ${task}: GPU ${physical_gpu} has ${free_mb:-unknown} MiB free; need ${MIN_FREE_MB}" >&2
        return 1
    fi

    mkdir -p "${model_dir}/weights" "${artifact_dir}/graph_logs"
    local command=(
        "${TORCHRUN}"
        --nnodes=1
        --nproc_per_node=1
        --master_addr=127.0.0.1
        --master_port="${master_port}"
        "${PROJECT_ROOT}/main.py"
        --method heatgeo
        --train_data "${PROJECT_ROOT}/data/train_set/merged_3_data_5k_each.csv"
        --student_model "${student}"
        --teacher_model "${teacher}"
        --teacher_pooling "${teacher_pooling}"
        --batch_size 32
        --epochs 5
        --lr 3e-5
        --max_length 256
        --sgc_weight 0.05
        --num_workers 4
        --save_dir "${model_dir}"
        --weights_dir "${model_dir}/weights"
        --cache_path "${artifact_dir}/teacher_embeddings.pt"
        --heatgeo_cache_path "${artifact_dir}/heatgeo_graph.pt"
        --heatgeo_log_dir "${artifact_dir}/graph_logs"
        --no_wandb
    )

    {
        echo "task=${task}"
        echo "timestamp=${RUN_STAMP}"
        echo "physical_gpu=${physical_gpu}"
        echo "gpu_free_mb_before_launch=${free_mb}"
        printf 'command='
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "${log_path}"

    local detached_command
    printf -v detached_command '%q ' \
        env \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="${physical_gpu}" \
        TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 \
        WANDB_MODE=disabled \
        "${command[@]}"
    printf -v detached_command \
        'exec %s >> %q 2>&1 < /dev/null' \
        "${detached_command}" "${log_path}"

    if ! tmux new-session -d -s "${tmux_session}" -c "${PROJECT_ROOT}" \
        bash -lc "${detached_command}"; then
        echo "Failed to create tmux session ${tmux_session}" >&2
        return 1
    fi
    local pid
    pid="$(tmux display-message -p -t "${tmux_session}:0.0" '#{pane_pid}')"
    echo "${pid}" > "${model_dir}/launcher.pid"
    echo "${tmux_session}" > "${model_dir}/tmux.session"
    sleep 3
    if ! tmux has-session -t "${tmux_session}" 2>/dev/null || \
        ! kill -0 "${pid}" 2>/dev/null; then
        echo "${task} exited during startup; see ${log_path}" >&2
        tail -n 30 "${log_path}" >&2 || true
        return 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${task}" "${physical_gpu}" "${pid}" "${log_path}" \
        "${model_dir}" "${artifact_dir}" >> "${MANIFEST}"
    echo "Started ${task} on physical GPU ${physical_gpu}: pid=${pid}, log=${log_path}"
}

failures=0
launch_job \
    qwen3_4b_to_bert_base 1 \
    Qwen/Qwen3-Embedding-4B \
    google-bert/bert-base-uncased \
    last_token 29511 || failures=$((failures + 1))
launch_job \
    bge_m3_to_minilmv2_h768 2 \
    BAAI/bge-m3 \
    nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large \
    cls 29512 || failures=$((failures + 1))
launch_job \
    qwen3_0_6b_to_minilmv2_h384 6 \
    Qwen/Qwen3-Embedding-0.6B \
    nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large \
    last_token 29516 || failures=$((failures + 1))

echo "Launch manifest: ${MANIFEST}"
if (( failures > 0 )); then
    echo "${failures} job(s) failed startup validation" >&2
    exit 1
fi
