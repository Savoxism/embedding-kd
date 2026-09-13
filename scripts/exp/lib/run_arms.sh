#!/usr/bin/env bash
# Shared graph-preparation and GPU-pool runner for scripts/exp/exp{1,2,4}_*.sh.
# This file is sourced; it deliberately does not change the caller's shell
# options or execute anything until run_arms is called.

run_arms() {
    if (( $# != 2 )); then
        echo "usage: run_arms <repo_root> <experiment>" >&2
        return 2
    fi

    local repo_root="$1"
    local experiment="$2"
    local train_script="$repo_root/scripts/ggpkd/train.sh"
    local export_script="$repo_root/scripts/exp/export_runs.py"
    local python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
    local pair="${PAIR:-qwen3_0_6b_to_minilmv2_h384}"
    local cache_root="${CACHE_ROOT:-$repo_root/cache/exp}"
    local run_id="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
    local result_base="${RESULT_BASE:-$repo_root/results/$experiment}"
    local run_root="$result_base/$run_id"
    local csv_out="${CSV_OUT:-$repo_root/runs/$experiment/results.csv}"
    local dry_run="${DRY_RUN:-0}"
    local jobs_per_gpu="${JOBS_PER_GPU:-1}"
    local graph_spec="${GRAPH_SPEC:-}"
    local arms_spec="${ARMS_SPEC:-}"
    local -a seeds=()
    local -a gpu_list=()
    local -a physical_gpu_list=()
    local -a graph_keys=()
    local -a graph_corpora=()
    local -a graph_flags=()
    local -a arm_labels=()
    local -a arm_graphs=()
    local -a arm_methods=()
    local -a arm_flags=()
    local -a pointwise_corpora=()
    local value other key corpus flags label graph_key method extra index i j

    IFS=',' read -r -a seeds <<< "${SEEDS:-42,43,44}"
    if [[ -n "${GPUS:-}" ]]; then
        IFS=',' read -r -a gpu_list <<< "$GPUS"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        while IFS= read -r value; do
            value="${value//[[:space:]]/}"
            [[ -n "$value" ]] && gpu_list+=("$value")
        done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    else
        echo "No nvidia-smi and no GPUS override; set GPUS explicitly" >&2
        return 2
    fi
    physical_gpu_list=("${gpu_list[@]}")

    if [[ ! -x "$python_bin" ]]; then
        echo "Project virtual-environment Python is not executable: $python_bin" >&2
        return 2
    fi
    if [[ ! -f "$train_script" || ! -f "$export_script" ]]; then
        echo "Experiment runner dependencies are missing under $repo_root/scripts" >&2
        return 2
    fi
    if (( ${#seeds[@]} == 0 || ${#gpu_list[@]} == 0 )); then
        echo "At least one seed and one GPU are required" >&2
        return 2
    fi
    if [[ ! "$jobs_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
        echo "JOBS_PER_GPU must be a positive integer, got: $jobs_per_gpu" >&2
        return 2
    fi
    for value in "${seeds[@]}"; do
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            echo "Seed must be a non-negative integer, got: $value" >&2
            return 2
        fi
    done
    for value in "${gpu_list[@]}"; do
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            echo "GPU ID must be a non-negative integer, got: $value" >&2
            return 2
        fi
        for other in "${gpu_list[@]}"; do
            [[ "$value" == "$other" ]] || continue
            index=0
            for key in "${gpu_list[@]}"; do
                [[ "$key" == "$value" ]] && ((index += 1))
            done
            if (( index > 1 )); then
                echo "Duplicate GPU ID: $value" >&2
                return 2
            fi
            break
        done
    done

    # Treat each repetition as an independent scheduler slot. This is opt-in so
    # existing runs remain one-process-per-GPU, while large-memory accelerators
    # can keep more than one small training process resident when requested.
    gpu_list=()
    for ((j = 0; j < jobs_per_gpu; j++)); do
        gpu_list+=("${physical_gpu_list[@]}")
    done

    while IFS='|' read -r key corpus flags; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        if [[ ! "$key" =~ ^[A-Za-z0-9_.-]+$ || -z "$corpus" ]]; then
            echo "Invalid graph record: $key|$corpus|$flags" >&2
            return 2
        fi
        if (( ${#graph_keys[@]} > 0 )); then
            for value in "${graph_keys[@]}"; do
                if [[ "$value" == "$key" ]]; then
                    echo "Duplicate graph_key in GRAPH_SPEC: $key" >&2
                    return 2
                fi
            done
        fi
        graph_keys+=("$key")
        graph_corpora+=("$corpus")
        graph_flags+=("$flags")
    done <<< "$graph_spec"
    if (( ${#graph_keys[@]} == 0 )); then
        echo "No graph records in GRAPH_SPEC" >&2
        return 2
    fi

    while IFS='|' read -r label graph_key method extra; do
        [[ -z "$label" || "$label" == \#* ]] && continue
        if [[ ! "$label" =~ ^[A-Za-z0-9_.-]+$ ]]; then
            echo "Invalid arm label: $label" >&2
            return 2
        fi
        if [[ "$method" != "ggpkd" && "$method" != "pointwise" ]]; then
            echo "Unsupported method for arm $label: $method" >&2
            return 2
        fi
        index=-1
        for ((i = 0; i < ${#graph_keys[@]}; i++)); do
            [[ "${graph_keys[$i]}" == "$graph_key" ]] && index=$i
        done
        if (( index < 0 )); then
            echo "Arm $label references unknown graph_key: $graph_key" >&2
            return 2
        fi
        if (( ${#arm_labels[@]} > 0 )); then
            for value in "${arm_labels[@]}"; do
                if [[ "$value" == "$label" ]]; then
                    echo "Duplicate arm label in ARMS_SPEC: $label" >&2
                    return 2
                fi
            done
        fi
        arm_labels+=("$label")
        arm_graphs+=("$graph_key")
        arm_methods+=("$method")
        arm_flags+=("$extra")
    done <<< "$arms_spec"
    if (( ${#arm_labels[@]} == 0 )); then
        echo "No arms in ARMS_SPEC" >&2
        return 2
    fi

    local total_runs=$(( ${#arm_labels[@]} * ${#seeds[@]} ))
    echo "$experiment sweep $run_id"
    echo "  pair:   $pair"
    echo "  arms:   ${#arm_labels[@]}"
    echo "  graphs: ${#graph_keys[@]}"
    echo "  seeds:  ${seeds[*]}"
    echo "  gpus:   ${physical_gpu_list[*]}"
    echo "  slots:  ${#gpu_list[@]} ($jobs_per_gpu job(s) per GPU)"
    echo "  runs:   $total_runs"
    echo "  root:   $run_root"

    if [[ "$dry_run" == "1" ]]; then
        echo
        echo "DRY_RUN: planned arms"
        for ((i = 0; i < ${#arm_labels[@]}; i++)); do
            printf '  %-34s graph=%-12s method=%-9s flags: %s\n' \
                "${arm_labels[$i]}" "${arm_graphs[$i]}" \
                "${arm_methods[$i]}" "${arm_flags[$i]}"
        done
        return 0
    fi

    if (( BASH_VERSINFO[0] < 5 )); then
        echo "Training scheduler requires Bash 5 or newer (found $BASH_VERSION)" >&2
        return 2
    fi
    if [[ -e "$run_root" ]]; then
        echo "Refusing to overwrite existing experiment run: $run_root" >&2
        return 2
    fi
    for corpus in "${graph_corpora[@]}"; do
        if [[ ! -f "$repo_root/$corpus" && ! -f "$corpus" ]]; then
            echo "Training corpus does not exist: $corpus" >&2
            return 2
        fi
    done

    local status_dir="$run_root/status"
    local log_dir="$run_root/logs"
    local runs_dir="$run_root/runs"
    local manifest="$run_root/manifest.tsv"
    local stats_tsv="$run_root/stats.tsv"
    local pointer="$repo_root/runs/$experiment/last_run_root.txt"
    local corpus_key teacher_cache graph_path log exit_file code
    mkdir -p "$status_dir" "$log_dir" "$runs_dir" "$cache_root"

    # These names are intentionally distinctive: the signal handler is a shell
    # function and Bash's dynamic scoping lets it see these run_arms locals.
    local -a _ra_active_pids=()
    _run_arms_signal() {
        local signal_code="$1" pid
        trap - INT TERM
        for pid in "${_ra_active_pids[@]}"; do
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        done
        printf '%s\n' "$signal_code" > "$run_root/controller.exit"
        exit "$signal_code"
    }
    trap '_run_arms_signal 130' INT
    trap '_run_arms_signal 143' TERM

    # shellcheck source=../../common/run_stats.sh
    source "$repo_root/scripts/common/run_stats.sh"
    export TOKENIZERS_PARALLELISM=false

    printf 'phase\tarm\tseed\tgpu\tpid\tstate\n' > "$manifest"
    printf 'phase\tunit\tseed\twall_seconds\tpeak_host_rss_mib\tpeak_gpu_mib\texit_code\n' > "$stats_tsv"
    {
        printf 'run_id\t%s\n' "$run_id"
        printf 'experiment\t%s\n' "$experiment"
        printf 'pair\t%s\n' "$pair"
        printf 'seeds\t%s\n' "${seeds[*]}"
        printf 'gpus\t%s\n' "${gpu_list[*]}"
        printf 'cache_root\t%s\n' "$cache_root"
        printf 'commit\t%s\n' "$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
    } > "$run_root/run_config.tsv"

    # GGPKD deduplicates identical anchors before caching; pointwise KD does not.
    # They therefore cannot safely share a teacher cache even on the same input
    # CSV. Prepare each pointwise corpus once before parallel runs start.
    for ((i = 0; i < ${#arm_labels[@]}; i++)); do
        [[ "${arm_methods[$i]}" == "pointwise" ]] || continue
        graph_key="${arm_graphs[$i]}"
        index=-1
        for ((j = 0; j < ${#graph_keys[@]}; j++)); do
            [[ "${graph_keys[$j]}" == "$graph_key" ]] && index=$j
        done
        corpus="${graph_corpora[$index]}"
        other=0
        if (( ${#pointwise_corpora[@]} > 0 )); then
            for value in "${pointwise_corpora[@]}"; do
                [[ "$value" == "$corpus" ]] && other=1
            done
        fi
        (( other == 0 )) && pointwise_corpora+=("$corpus")
    done

    for corpus in "${pointwise_corpora[@]}"; do
        corpus_key="$(basename "${corpus%.*}")"
        teacher_cache="$cache_root/$pair/$corpus_key/teacher_pointwise.pt"
        log="$log_dir/cache.pointwise.$corpus_key.log"
        exit_file="$status_dir/cache.pointwise.$corpus_key.exit"
        echo "Building pointwise teacher cache for $corpus on GPU ${gpu_list[0]} (log: $log)"
        set +e
        PAIR_KEY="$pair" METHOD="pointwise" GPU="${gpu_list[0]}" \
            PYTHON_BIN="$python_bin" TRAIN_DATA="$corpus" \
            CACHE_PATH="$teacher_cache" \
            SAVE_DIR="$run_root/cache_setup/pointwise_$corpus_key" \
            bash "$train_script" --prepare_cache_only >"$log" 2>&1
        code=$?
        set -e
        printf '%s\n' "$code" > "$exit_file"
        printf 'cache\tpointwise_%s\t-\t%s\t-\texit_%s\n' \
            "$corpus_key" "${gpu_list[0]}" "$code" >> "$manifest"
        printf 'cache\tpointwise_%s\t-\t%s\n' "$corpus_key" \
            "$(run_stats_row "$run_root/cache_setup/pointwise_$corpus_key/run_stats.json")" >> "$stats_tsv"
        if (( code != 0 )); then
            printf '1\n' > "$run_root/controller.exit"
            trap - INT TERM
            unset -f _run_arms_signal
            echo "Pointwise teacher cache failed for $corpus (exit $code); see $log" >&2
            return 1
        fi
    done

    # Prepare each graph serially. Graph variants for the same corpus share the
    # teacher cache, so concurrent preparation would race on that cache file.
    for ((i = 0; i < ${#graph_keys[@]}; i++)); do
        key="${graph_keys[$i]}"
        corpus="${graph_corpora[$i]}"
        flags="${graph_flags[$i]}"
        corpus_key="$(basename "${corpus%.*}")"
        teacher_cache="$cache_root/$pair/$corpus_key/teacher_train.pt"
        graph_path="$cache_root/$pair/$corpus_key/graph_$key.pt"
        log="$log_dir/graph.$key.log"
        exit_file="$status_dir/graph.$key.exit"
        echo "Building graph $key on GPU ${gpu_list[0]} (log: $log)"
        set +e
        # shellcheck disable=SC2086 -- graph flags are a trusted CLI fragment.
        PAIR_KEY="$pair" GPU="${gpu_list[0]}" PYTHON_BIN="$python_bin" \
            TRAIN_DATA="$corpus" CACHE_PATH="$teacher_cache" \
            GGPKD_CACHE_PATH="$graph_path" \
            GGPKD_LOG_DIR="$run_root/graph_logs/$key" \
            SAVE_DIR="$run_root/graph_setup/$key" \
            bash "$train_script" --prepare_cache_only $flags >"$log" 2>&1
        code=$?
        set -e
        printf '%s\n' "$code" > "$exit_file"
        printf 'graph\t%s\t-\t%s\t-\texit_%s\n' \
            "$key" "${gpu_list[0]}" "$code" >> "$manifest"
        printf 'graph\t%s\t-\t%s\n' "$key" \
            "$(run_stats_row "$run_root/graph_setup/$key/run_stats.json")" >> "$stats_tsv"
        if (( code != 0 )); then
            printf '1\n' > "$run_root/controller.exit"
            trap - INT TERM
            unset -f _run_arms_signal
            echo "Graph build failed for $key (exit $code); see $log" >&2
            return 1
        fi
    done

    local -a task_labels=()
    local -a task_seeds=()
    local -a active_gpus=()
    local -a active_labels=()
    local -a active_seeds=()
    for label in "${arm_labels[@]}"; do
        for value in "${seeds[@]}"; do
            task_labels+=("$label")
            task_seeds+=("$value")
        done
    done

    local next_task=0
    local failed_runs=0
    local completed_pid completed_status completed_index completed_gpu
    local completed_label completed_seed run_dir task pid

    _run_arms_launch() {
        local task_index="$1"
        local gpu="$2"
        local task_label="${task_labels[$task_index]}"
        local task_seed="${task_seeds[$task_index]}"
        local task_graph="" task_method="" task_flags="" task_graph_flags=""
        local task_corpus="" task_graph_index=-1 arm_index=-1

        for ((j = 0; j < ${#arm_labels[@]}; j++)); do
            [[ "${arm_labels[$j]}" == "$task_label" ]] && arm_index=$j
        done
        task_graph="${arm_graphs[$arm_index]}"
        task_method="${arm_methods[$arm_index]}"
        task_flags="${arm_flags[$arm_index]}"
        for ((j = 0; j < ${#graph_keys[@]}; j++)); do
            [[ "${graph_keys[$j]}" == "$task_graph" ]] && task_graph_index=$j
        done
        task_corpus="${graph_corpora[$task_graph_index]}"
        task_graph_flags="${graph_flags[$task_graph_index]}"
        corpus_key="$(basename "${task_corpus%.*}")"
        if [[ "$task_method" == "pointwise" ]]; then
            teacher_cache="$cache_root/$pair/$corpus_key/teacher_pointwise.pt"
        else
            teacher_cache="$cache_root/$pair/$corpus_key/teacher_train.pt"
        fi
        graph_path="$cache_root/$pair/$corpus_key/graph_$task_graph.pt"
        task="$task_label.seed_$task_seed"
        run_dir="$runs_dir/$task_label/seed_$task_seed"
        log="$log_dir/$task.log"
        exit_file="$status_dir/$task.exit"
        mkdir -p "$run_dir"
        (
            set +e
            # shellcheck disable=SC2086 -- both flag strings are trusted CLI fragments.
            PAIR_KEY="$pair" METHOD="$task_method" SEED="$task_seed" GPU="$gpu" \
                PYTHON_BIN="$python_bin" TRAIN_DATA="$task_corpus" \
                CACHE_PATH="$teacher_cache" GGPKD_CACHE_PATH="$graph_path" \
                GGPKD_LOG_DIR="$run_dir/graph_logs" SAVE_DIR="$run_dir" \
                WEIGHTS_DIR="$run_dir/weights" bash "$train_script" \
                --final_weights_only --eval_every 0 \
                $task_graph_flags $task_flags >"$log" 2>&1
            code=$?
            printf '%s\n' "$code" > "$exit_file"
            exit "$code"
        ) &
        pid=$!
        _ra_active_pids+=("$pid")
        active_gpus+=("$gpu")
        active_labels+=("$task_label")
        active_seeds+=("$task_seed")
        printf 'train\t%s\t%s\t%s\t%s\trunning\n' \
            "$task_label" "$task_seed" "$gpu" "$pid" >> "$manifest"
        echo "Launched $task on GPU $gpu (pid $pid)"
    }

    while (( next_task < ${#task_labels[@]} && next_task < ${#gpu_list[@]} )); do
        _run_arms_launch "$next_task" "${gpu_list[$next_task]}"
        ((next_task += 1))
    done

    while (( ${#_ra_active_pids[@]} > 0 )); do
        completed_pid=""
        set +e
        wait -n -p completed_pid "${_ra_active_pids[@]}"
        completed_status=$?
        set -e
        if [[ -z "$completed_pid" ]]; then
            echo "Could not identify completed experiment process" >&2
            failed_runs=1
            break
        fi
        completed_index=-1
        for ((i = 0; i < ${#_ra_active_pids[@]}; i++)); do
            [[ "${_ra_active_pids[$i]}" == "$completed_pid" ]] && completed_index=$i
        done
        if (( completed_index < 0 )); then
            echo "Completed PID $completed_pid is not in the active task table" >&2
            failed_runs=1
            break
        fi
        completed_gpu="${active_gpus[$completed_index]}"
        completed_label="${active_labels[$completed_index]}"
        completed_seed="${active_seeds[$completed_index]}"
        printf 'train\t%s\t%s\t%s\t%s\texit_%s\n' \
            "$completed_label" "$completed_seed" "$completed_gpu" \
            "$completed_pid" "$completed_status" >> "$manifest"
        printf 'train\t%s\t%s\t%s\n' "$completed_label" "$completed_seed" \
            "$(run_stats_row "$runs_dir/$completed_label/seed_$completed_seed/run_stats.json")" >> "$stats_tsv"
        if (( completed_status != 0 )); then
            failed_runs=1
            echo "FAILED: $completed_label.seed_$completed_seed exited $completed_status" >&2
        else
            echo "Completed: $completed_label.seed_$completed_seed ($(run_stats_field "$runs_dir/$completed_label/seed_$completed_seed/run_stats.json" wall_seconds)s)"
        fi

        unset '_ra_active_pids[completed_index]' 'active_gpus[completed_index]'
        unset 'active_labels[completed_index]' 'active_seeds[completed_index]'
        _ra_active_pids=("${_ra_active_pids[@]}")
        active_gpus=("${active_gpus[@]}")
        active_labels=("${active_labels[@]}")
        active_seeds=("${active_seeds[@]}")

        if (( next_task < ${#task_labels[@]} )); then
            _run_arms_launch "$next_task" "$completed_gpu"
            ((next_task += 1))
        fi
    done

    unset -f _run_arms_launch
    trap - INT TERM
    unset -f _run_arms_signal
    if (( failed_runs != 0 )); then
        printf '1\n' > "$run_root/controller.exit"
        echo "At least one $experiment run failed; refusing to aggregate" >&2
        return 1
    fi

    "$python_bin" "$export_script" "$run_root" --experiment "$experiment" \
        --pair "$pair" --status-file "$status_dir" --out "$csv_out"
    mkdir -p "$(dirname "$pointer")"
    local pointer_tmp
    pointer_tmp="$(mktemp "${pointer}.tmp.XXXXXX")"
    printf '%s\n' "$run_root" > "$pointer_tmp"
    mv "$pointer_tmp" "$pointer"
    printf '0\n' > "$run_root/controller.exit"
    echo "$experiment sweep complete: $run_root"
}
