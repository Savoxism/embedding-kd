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
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-5}"
DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
REQUESTED_TASKS=("$@")

if [[ ! -x "${TORCHRUN}" ]]; then
    echo "Missing project torchrun: ${TORCHRUN}" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required" >&2
    exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required to keep jobs alive after SSH disconnects" >&2
    exit 1
fi

mkdir -p "${MODEL_ROOT}" "${ARTIFACT_ROOT}" "${LOG_ROOT}"
MANIFEST="${LOG_ROOT}/launch_${RUN_STAMP}.tsv"
printf 'task\tphysical_gpu\tpid\tlog\tmodel_dir\tartifact_dir\n' > "${MANIFEST}"

select_idle_gpus() {
    local needed="$1"
    local active_gpu_uuids
    active_gpu_uuids="$(
        nvidia-smi \
            --query-compute-apps=gpu_uuid \
            --format=csv,noheader,nounits 2>/dev/null || true
    )"
    nvidia-smi \
        --query-gpu=index,uuid,memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F, \
            -v minimum="${MIN_FREE_MB}" \
            -v max_util="${MAX_IDLE_UTIL}" \
            -v active_uuids="${active_gpu_uuids}" '
            BEGIN {
                active_count = split(active_uuids, values, /[[:space:]]+/)
                for (i = 1; i <= active_count; i++) active[values[i]] = 1
            }
            {
                gsub(/[[:space:]]/, "", $1)
                gsub(/[[:space:]]/, "", $2)
                gsub(/[[:space:]]/, "", $3)
                gsub(/[[:space:]]/, "", $4)
                if ($3 >= minimum && $4 <= max_util && !($2 in active)) {
                    print $1 "\t" $3
                }
            }
        ' \
        | sort -t $'\t' -k2,2nr \
        | awk -v needed="${needed}" 'NR <= needed { print $1 }'
}

select_free_gpu() {
    local selected
    selected="$(select_idle_gpus 1)"
    if [[ -z "${selected}" ]]; then
        return 1
    fi
    printf '%s\n' "${selected}"
}

launch_job() {
    local task="$1"
    local physical_gpu="$2"
    local teacher="$3"
    local student="$4"
    local teacher_pooling="$5"
    local master_port="$6"
    local sampling_mode="${7:-diffusion}"
    local artifact_file="${8:-heatgeo_graph.pt}"
    local save_every="${9:-1}"
    local free_mb
    local model_dir="${MODEL_ROOT}/${task}/${RUN_STAMP}"
    local artifact_dir="${ARTIFACT_ROOT}/${task}"
    local log_path="${LOG_ROOT}/${task}_${RUN_STAMP}.log"
    local pid_file="${model_dir}/launcher.pid"

    free_mb="$(
        nvidia-smi -i "${physical_gpu}" \
            --query-gpu=memory.free --format=csv,noheader,nounits \
            | tr -d '[:space:]'
    )"
    if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
        echo "Refusing ${task}: GPU ${physical_gpu} has ${free_mb:-unknown} MiB free; need ${MIN_FREE_MB}" >&2
        return 1
    fi

    mkdir -p "${model_dir}/weights" "${artifact_dir}"
    if [[ "${sampling_mode}" == "diffusion" ]]; then
        mkdir -p "${artifact_dir}/graph_logs"
    fi
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
        --save_every "${save_every}"
        --lr 3e-5
        --max_length 256
        --sgc_weight 0.05
        --heatgeo_sampling_mode "${sampling_mode}"
        --num_workers 4
        --save_dir "${model_dir}"
        --weights_dir "${model_dir}/weights"
        --cache_path "${artifact_dir}/teacher_embeddings.pt"
        --heatgeo_cache_path "${artifact_dir}/${artifact_file}"
        --heatgeo_log_dir "${artifact_dir}/graph_logs"
        --no_wandb
    )

    {
        echo "task=${task}"
        echo "timestamp=${RUN_STAMP}"
        echo "physical_gpu=${physical_gpu}"
        echo "gpu_free_mb_before_launch=${free_mb}"
        echo "sampling_mode=${sampling_mode}"
        echo "save_every=${save_every}"
        if [[ "${sampling_mode}" == "random_hard_direct" ]]; then
            echo "candidate_composition=32_random+24_hard+8_random_negative"
            echo "diffusion_loss=disabled"
        else
            echo "diffusion_loss=enabled"
        fi
        echo "hf_hub_disable_xet=${DISABLE_XET}"
        printf 'command='
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "${log_path}"

    nohup setsid bash -c '
        pid_file=$1
        project_root=$2
        shift 2
        echo $$ > "${pid_file}"
        cd "${project_root}"
        exec "$@"
    ' _ "${pid_file}" "${PROJECT_ROOT}" \
        env \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="${physical_gpu}" \
        TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 \
        WANDB_MODE=disabled \
        HF_HUB_DISABLE_XET="${DISABLE_XET}" \
        "${command[@]}" >> "${log_path}" 2>&1 < /dev/null &
    local pid
    sleep 3
    if [[ ! -s "${pid_file}" ]]; then
        echo "${task} did not write a launcher PID; see ${log_path}" >&2
        tail -n 30 "${log_path}" >&2 || true
        return 1
    fi
    pid="$(cat "${pid_file}")"
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "${task} exited during startup; see ${log_path}" >&2
        tail -n 30 "${log_path}" >&2 || true
        return 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${task}" "${physical_gpu}" "${pid}" "${log_path}" \
        "${model_dir}" "${artifact_dir}" >> "${MANIFEST}"
    echo "Started ${task} on physical GPU ${physical_gpu}: pid=${pid}, log=${log_path}"
}

is_requested() {
    local task="$1"
    local requested
    if (( ${#REQUESTED_TASKS[@]} == 0 )); then
        return 0
    fi
    for requested in "${REQUESTED_TASKS[@]}"; do
        if [[ "${requested}" == "${task}" ]]; then
            return 0
        fi
    done
    return 1
}

for requested in "${REQUESTED_TASKS[@]}"; do
    case "${requested}" in
        qwen3_4b_to_bert_base|bge_m3_to_minilmv2_h768|qwen3_0_6b_to_minilmv2_h384|bge_m3_to_minilmv2_h768_random_hard_direct|qwen3_4b_to_bert_base_random_hard_direct|qwen3_0_6b_to_minilmv2_h384_random_hard_direct) ;;
        *)
            echo "Unknown task: ${requested}" >&2
            exit 1
            ;;
    esac
done

qwen4_random_gpu=""
qwen06_random_gpu=""
if is_requested qwen3_4b_to_bert_base_random_hard_direct || \
    is_requested qwen3_0_6b_to_minilmv2_h384_random_hard_direct; then
    idle_gpu_snapshot="$(select_idle_gpus 2)"
    if is_requested qwen3_4b_to_bert_base_random_hard_direct && \
        is_requested qwen3_0_6b_to_minilmv2_h384_random_hard_direct; then
        if [[ "$(printf '%s\n' "${idle_gpu_snapshot}" | awk 'NF { count++ } END { print count + 0 }')" -lt 2 ]]; then
            echo "Need two distinct idle GPUs with at least ${MIN_FREE_MB} MiB free and utilization <= ${MAX_IDLE_UTIL}%" >&2
            exit 1
        fi
        qwen4_random_gpu="$(printf '%s\n' "${idle_gpu_snapshot}" | sed -n '1p')"
        qwen06_random_gpu="$(printf '%s\n' "${idle_gpu_snapshot}" | sed -n '2p')"
    else
        selected_random_gpu="$(printf '%s\n' "${idle_gpu_snapshot}" | sed -n '1p')"
        if [[ -z "${selected_random_gpu}" ]]; then
            echo "No idle GPU has at least ${MIN_FREE_MB} MiB free and utilization <= ${MAX_IDLE_UTIL}%" >&2
            exit 1
        fi
        if is_requested qwen3_4b_to_bert_base_random_hard_direct; then
            qwen4_random_gpu="${selected_random_gpu}"
        else
            qwen06_random_gpu="${selected_random_gpu}"
        fi
    fi
fi

failures=0
if is_requested qwen3_4b_to_bert_base; then
    launch_job \
        qwen3_4b_to_bert_base 1 \
        Qwen/Qwen3-Embedding-4B \
        google-bert/bert-base-uncased \
        last_token 29511 || failures=$((failures + 1))
fi
if is_requested bge_m3_to_minilmv2_h768; then
    launch_job \
        bge_m3_to_minilmv2_h768 2 \
        BAAI/bge-m3 \
        nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large \
        cls 29512 || failures=$((failures + 1))
fi
if is_requested qwen3_0_6b_to_minilmv2_h384; then
    launch_job \
        qwen3_0_6b_to_minilmv2_h384 3 \
        Qwen/Qwen3-Embedding-0.6B \
        nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large \
        last_token 29516 || failures=$((failures + 1))
fi
if is_requested bge_m3_to_minilmv2_h768_random_hard_direct; then
    random_hard_gpu="${BGE_RANDOM_GPU:-}"
    if [[ -z "${random_hard_gpu}" ]]; then
        if ! random_hard_gpu="$(select_free_gpu)"; then
            echo "No GPU has at least ${MIN_FREE_MB} MiB free" >&2
            failures=$((failures + 1))
            random_hard_gpu=""
        fi
    fi
    if [[ -n "${random_hard_gpu}" ]]; then
        launch_job \
            bge_m3_to_minilmv2_h768_random_hard_direct "${random_hard_gpu}" \
            BAAI/bge-m3 \
            nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large \
            cls 29517 random_hard_direct hard_negative_pool.pt \
            || failures=$((failures + 1))
    fi
fi
if is_requested qwen3_4b_to_bert_base_random_hard_direct; then
    launch_job \
        qwen3_4b_to_bert_base_random_hard_direct "${qwen4_random_gpu}" \
        Qwen/Qwen3-Embedding-4B \
        google-bert/bert-base-uncased \
        last_token 29518 random_hard_direct hard_negative_pool.pt 3 \
        || failures=$((failures + 1))
fi
if is_requested qwen3_0_6b_to_minilmv2_h384_random_hard_direct; then
    launch_job \
        qwen3_0_6b_to_minilmv2_h384_random_hard_direct "${qwen06_random_gpu}" \
        Qwen/Qwen3-Embedding-0.6B \
        nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large \
        last_token 29519 random_hard_direct hard_negative_pool.pt 3 \
        || failures=$((failures + 1))
fi

echo "Launch manifest: ${MANIFEST}"
if (( failures > 0 )); then
    echo "${failures} job(s) failed startup validation" >&2
    exit 1
fi
