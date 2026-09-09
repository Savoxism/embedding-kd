#!/usr/bin/env bash
# Cost accounting for one training run: wall clock, peak host memory, peak GPU
# memory.
#
# Every launcher routes its python invocation through `run_with_stats`, so the
# cost of a run is written next to its results instead of having to be
# reconstructed afterwards from log timestamps and a guess. Measuring from
# outside the process has two consequences worth relying on: a run that dies
# mid-epoch still leaves its numbers behind, and the figures cover the whole
# process tree -- dataloader workers included -- rather than only what the
# training process reports about itself.
#
# Host memory is read from /proc/<pid>/status:VmHWM, the kernel's own
# high-water mark, summed over that tree. Because each process reports its own
# peak rather than its current size, a slow poll still catches a spike inside a
# process that outlives the poll; only a process born and reaped entirely
# between two polls is missed. GPU memory has no such high-water counter
# exposed by nvidia-smi, so it is genuinely sampled and RUN_STATS_POLL_SECONDS
# bounds the spike it can miss. The GPU figure is per-process (the compute-apps
# table), not per-device, so it stays correct when two runs share one GPU.

if [[ -n "${RUN_STATS_SOURCED:-}" ]]; then
    return 0
fi
RUN_STATS_SOURCED=1

RUN_STATS_POLL_SECONDS="${RUN_STATS_POLL_SECONDS:-2}"

# Every pid descending from $1, including it. Printed space-separated.
_run_stats_pid_tree() {
    local root="$1"
    ps -eo pid=,ppid= 2>/dev/null | awk -v root="$root" '
        { pid[NR] = $1; parent[$1] = $2; n = NR }
        END {
            keep[root] = 1
            changed = 1
            while (changed) {
                changed = 0
                for (i = 1; i <= n; i++) {
                    p = pid[i]
                    if (!(p in keep) && (parent[p] in keep)) {
                        keep[p] = 1
                        changed = 1
                    }
                }
            }
            for (p in keep) printf "%s ", p
        }'
}

# Polls until the root pid exits, then writes "peak_rss_kb peak_gpu_mib samples
# gpu_seen" to $2. Runs as a background subshell, so errexit is turned off here
# rather than in the caller: a transient read failure against a pid that has
# just exited must not take the sampler down with it.
_run_stats_sampler() {
    set +e
    local root_pid="$1" out_file="$2"
    local peak_rss_kb=0 peak_gpu_mib=0 samples=0 have_gpu=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        have_gpu=1
    fi

    while kill -0 "$root_pid" 2>/dev/null; do
        local pids rss_kb=0 pid hwm
        pids="$(_run_stats_pid_tree "$root_pid")"
        for pid in $pids; do
            hwm="$(awk '/^VmHWM:/ { print $2 }' "/proc/$pid/status" 2>/dev/null)"
            if [[ "$hwm" =~ ^[0-9]+$ ]]; then
                rss_kb=$(( rss_kb + hwm ))
            fi
        done
        if (( rss_kb > peak_rss_kb )); then
            peak_rss_kb=$rss_kb
        fi

        if (( have_gpu )); then
            local gpu_mib
            gpu_mib="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory \
                --format=csv,noheader,nounits 2>/dev/null | tr ',' ' ' \
                | awk -v pidlist="$pids" '
                    BEGIN { n = split(pidlist, a, " "); for (i = 1; i <= n; i++) keep[a[i]] = 1 }
                    ($1 in keep) && ($2 ~ /^[0-9]+$/) { total += $2 }
                    END { print total + 0 }')"
            if [[ "$gpu_mib" =~ ^[0-9]+$ ]] && (( gpu_mib > peak_gpu_mib )); then
                peak_gpu_mib=$gpu_mib
            fi
        fi

        samples=$(( samples + 1 ))
        sleep "$RUN_STATS_POLL_SECONDS"
    done

    printf '%s %s %s %s\n' "$peak_rss_kb" "$peak_gpu_mib" "$samples" "$have_gpu" > "$out_file"
}

_run_stats_hms() {
    awk -v s="$1" 'BEGIN {
        t = int(s + 0.5)
        printf "%d:%02d:%02d", t / 3600, (t % 3600) / 60, t % 60
    }'
}

_run_stats_mem() {
    awk -v mib="$1" 'BEGIN {
        if (mib < 1024) printf "%.0f MiB", mib
        else printf "%.2f GiB", mib / 1024
    }'
}

# run_with_stats <stats.json> <command...>
#
# Runs the command, then writes the JSON stats file and echoes a one-line
# summary into the run log. Returns the command's own exit code, so a launcher
# can keep using it as its last statement.
run_with_stats() {
    local stats_file="$1"
    shift
    if (( $# == 0 )); then
        echo "run_with_stats: no command given" >&2
        return 2
    fi
    if [[ "${RUN_STATS_DISABLE:-0}" == "1" ]]; then
        "$@"
        return $?
    fi

    mkdir -p "$(dirname -- "$stats_file")"
    local sample_file
    sample_file="$(mktemp "${TMPDIR:-/tmp}/run_stats.XXXXXX")"

    local start_epoch start_iso
    start_epoch="$(date +%s.%N)"
    start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    "$@" &
    local cmd_pid=$!
    _run_stats_sampler "$cmd_pid" "$sample_file" &
    local sampler_pid=$!

    # `wait` on a failing child must not take the launcher down before the
    # stats file is written; the launchers run under `set -e`.
    local errexit=0
    case "$-" in *e*) errexit=1 ;; esac
    set +e
    wait "$cmd_pid"
    local code=$?
    wait "$sampler_pid"
    if (( errexit )); then
        set -e
    fi

    local end_epoch end_iso
    end_epoch="$(date +%s.%N)"
    end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    local peak_rss_kb=0 peak_gpu_mib=0 samples=0 have_gpu=0
    read -r peak_rss_kb peak_gpu_mib samples have_gpu < "$sample_file" 2>/dev/null
    rm -f "$sample_file"
    : "${peak_rss_kb:=0}" "${peak_gpu_mib:=0}" "${samples:=0}" "${have_gpu:=0}"

    local wall
    wall="$(awk -v a="$start_epoch" -v b="$end_epoch" 'BEGIN { printf "%.1f", b - a }')"
    local rss_mib
    rss_mib="$(awk -v kb="$peak_rss_kb" 'BEGIN { printf "%.1f", kb / 1024 }')"

    # A newline inside an argument would otherwise split the JSON string.
    local cmd_json
    cmd_json="$(printf '%s ' "$@" | sed 's/ $//; s/\\/\\\\/g; s/"/\\"/g' \
        | sed ':a;N;$!ba;s/\n/\\n/g')"

    local gpu_json="$peak_gpu_mib"
    if (( ! have_gpu )); then
        gpu_json="null"
    fi

    cat > "$stats_file" <<EOF
{
  "command": "$cmd_json",
  "host": "$(hostname)",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES:-}",
  "started_at": "$start_iso",
  "finished_at": "$end_iso",
  "wall_seconds": $wall,
  "exit_code": $code,
  "peak_host_rss_mib": $rss_mib,
  "peak_gpu_mib": $gpu_json,
  "poll_seconds": $RUN_STATS_POLL_SECONDS,
  "samples": $samples
}
EOF

    local gpu_human="n/a (no nvidia-smi)"
    if (( have_gpu )); then
        gpu_human="$(_run_stats_mem "$peak_gpu_mib")"
    fi
    printf '[stats] wall %s (%ss) | peak host RSS %s | peak GPU %s | exit %s -> %s\n' \
        "$(_run_stats_hms "$wall")" "$wall" \
        "$(_run_stats_mem "$rss_mib")" "$gpu_human" "$code" "$stats_file"

    return "$code"
}

# One field out of a stats file, or "NA" when the run left none behind. The
# multi-run launchers use this to build their per-run cost table without
# needing a JSON parser.
run_stats_field() {
    local file="$1" key="$2"
    if [[ ! -f "$file" ]]; then
        printf 'NA'
        return 0
    fi
    awk -F'"' -v key="$key" '
        $2 == key {
            line = $0
            sub(/^[^:]*:[[:space:]]*/, "", line)
            sub(/,$/, "", line)
            sub(/^"/, "", line)
            sub(/"$/, "", line)
            print (line == "" ? "NA" : line)
            found = 1
            exit
        }
        END { if (!found) print "NA" }' "$file"
}

# The wall / host / GPU / exit columns of one run, tab-separated, for the
# aggregate cost table the multi-run launchers write. "NA" in every column when
# the run left no stats file behind -- a run killed by the OOM killer before
# `run_with_stats` could write one is exactly the case worth seeing in the
# table rather than silently omitting.
run_stats_row() {
    local file="$1"
    printf '%s\t%s\t%s\t%s' \
        "$(run_stats_field "$file" wall_seconds)" \
        "$(run_stats_field "$file" peak_host_rss_mib)" \
        "$(run_stats_field "$file" peak_gpu_mib)" \
        "$(run_stats_field "$file" exit_code)"
}
