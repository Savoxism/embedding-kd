#!/usr/bin/env bash
# Runs every stage in experiments.md, in the order that file prescribes.
#
# It deliberately STOPS after Stage 0 when GRAPH_K is unset. Stage 0 chooses the
# operating point and the rule for choosing it is a judgement call -- smallest
# graph_k within one seed-sd of the best Avg -- so a script must not guess it.
# Read runs/stage0_graph_k/results.csv, then re-invoke with GRAPH_K=<winner>.
#
#   GPUS=0,1,2,3            bash scripts/exp/run_all.sh   # stage 0, then stop
#   GRAPH_K=50 GPUS=0,1,2,3 bash scripts/exp/run_all.sh   # stages 1-3
#   DRY_RUN=1 GRAPH_K=50    bash scripts/exp/run_all.sh   # print the plan only
#
# Env: FROM / TO restrict the range (0, 1, 2b, 2c, 2d, 3). Everything else is
# forwarded to the stage scripts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAGES=(0 1 2b 2c 2d 3)
FROM="${FROM:-}"
TO="${TO:-}"

in_range() {
    local stage="$1" seen_from=0 i
    [[ -z "$FROM" && -z "$TO" ]] && return 0
    for i in "${STAGES[@]}"; do
        [[ -n "$FROM" && "$i" == "$FROM" ]] && seen_from=1
        [[ -z "$FROM" ]] && seen_from=1
        if [[ "$i" == "$stage" ]]; then
            (( seen_from )) || return 1
            return 0
        fi
        [[ -n "$TO" && "$i" == "$TO" ]] && return 1
    done
    return 1
}

script_for() {
    case "$1" in
        0)  echo "stage0_graph_k.sh" ;;
        1)  echo "stage1_deletions.sh" ;;
        2b) echo "stage2b_ladder.sh" ;;
        2c) echo "stage2c_dose_response.sh" ;;
        2d) echo "stage2d_components.sh" ;;
        3)  echo "stage3_main_table.sh" ;;
    esac
}

failed=()
for stage in "${STAGES[@]}"; do
    in_range "$stage" || continue
    if [[ "$stage" != "0" && -z "${GRAPH_K:-}" ]]; then
        cat <<'EOF'

==============================================================
Stage 0 is done. Stopping here on purpose.

Pick the operating point before anything else runs:

    column -s, -t runs/stage0_graph_k/results.csv | less -S

Rule: the smallest graph_k within one seed-sd of the best Avg. Small beats
tied-and-large -- it is the whole cost argument. Then:

    GRAPH_K=<winner> bash scripts/exp/run_all.sh
==============================================================
EOF
        exit 0
    fi
    script="$(script_for "$stage")"
    echo
    echo "=============================================================="
    echo "Stage $stage -- $script"
    echo "=============================================================="
    # Second pass: the operating point is already known, so Stage 0 has nothing
    # left to sweep. It still runs at that one setting, because its seeds are
    # the full-model reference Stages 1, 2D and 3 compare against.
    if [[ "$stage" == "0" && -n "${GRAPH_K:-}" ]]; then
        echo "GRAPH_K=$GRAPH_K is already chosen: running Stage 0 at that setting"
        echo "only, to materialise the full-model reference (3 runs, not 12)."
        if GRAPH_KS="$GRAPH_K" bash "$SCRIPT_DIR/$script"; then
            continue
        fi
        code=$?
        failed+=("$stage")
        echo "Stage $stage failed with exit $code" >&2
        break
    fi
    if bash "$SCRIPT_DIR/$script"; then
        :
    else
        code=$?
        failed+=("$stage")
        echo "Stage $stage failed with exit $code" >&2
        # Later stages read earlier ones, so a failure is not something to run
        # past: stop and let the operator look.
        break
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo
    echo "Failed stages: ${failed[*]}" >&2
    exit 1
fi

if [[ "${DRY_RUN:-0}" != "1" && -n "${GRAPH_K:-}" ]]; then
    cat <<'EOF'

==============================================================
Training done. Two things left, neither of which needs a GPU:

  A  the count: (B-1) * k / (N-1). Analytic, plus one exposure curve from
     scripts/exp/coverage.py.
  E  held-out pairs, post-hoc on Stage 2D's checkpoints:
         bash scripts/exp/exp3_heldout_geometry.sh
     Read spearman_anchor beside the pooled spearman.
==============================================================
EOF
fi
