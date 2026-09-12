#!/usr/bin/env bash
# Shared preamble for the staged sweeps in experiments.md.
#
# Sourced, never executed. It adds three things on top of `run_arms.sh`:
#
#   require_graph_k   the operating point is chosen by a human in Stage 0 and
#                     every later stage inherits it, so no stage may quietly
#                     default it.
#   has_flag          whether main.py accepts a CLI flag. Three arms in
#                     experiments.md need patches that do not exist yet; without
#                     this check the sweep would build caches, launch, and fail
#                     one run at a time with an argparse error.
#   note / warn       one prefix for everything a stage says about itself.

stage_common_init() {
    STAGE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")/../.." && pwd)"
    PYTHON_BIN="${PYTHON_BIN:-$STAGE_REPO_ROOT/.venv/bin/python}"
    PAIR="${PAIR:-qwen3_0_6b_to_minilmv2_h384}"
    CORPUS="${CORPUS:-data/train_set/merged_3_data_5k_each.csv}"
    SEEDS="${SEEDS:-42,43,44}"
    CACHE_ROOT="${CACHE_ROOT:-$STAGE_REPO_ROOT/cache/exp}"
    export PAIR SEEDS CACHE_ROOT PYTHON_BIN
    STAGE_MISSING=()
}

note() { printf '  %s\n' "$*"; }
warn() { printf '  WARNING: %s\n' "$*" >&2; }

# Cached `main.py --help`, so a stage with several guards pays for it once.
_stage_help() {
    if [[ -z "${STAGE_HELP_TEXT:-}" ]]; then
        STAGE_HELP_TEXT="$("$PYTHON_BIN" "$STAGE_REPO_ROOT/main.py" --help 2>&1 || true)"
    fi
    printf '%s' "$STAGE_HELP_TEXT"
}

# has_flag --some_flag  -> 0 when main.py accepts it
has_flag() {
    _stage_help | grep -q -- "$1"
}

# Register an arm that cannot run yet. The stage still runs everything else.
skip_arm() {
    local arm="$1" flag="$2" why="$3"
    STAGE_MISSING+=("$arm")
    warn "skipping arm '$arm': main.py has no $flag."
    warn "  $why"
}

report_missing() {
    (( ${#STAGE_MISSING[@]} == 0 )) && return 0
    echo
    echo "  ${#STAGE_MISSING[@]} arm(s) were skipped because the code does not"
    echo "  support them yet: ${STAGE_MISSING[*]}"
    echo "  See the patch table at the top of experiments.md."
}

# Stage 0 picks the operating point; nothing after it may guess one.
require_graph_k() {
    if [[ -z "${GRAPH_K:-}" ]]; then
        cat >&2 <<'EOF'
GRAPH_K is not set.

Every stage after Stage 0 runs at the operating point Stage 0 selected, and
picking it is a judgement call (smallest graph_k within one seed-sd of the best
Avg), so it is not defaulted here. Run Stage 0, read
runs/stage0_graph_k/results.csv, then:

    GRAPH_K=<winner> bash scripts/exp/<this script>
EOF
        exit 2
    fi
    if [[ ! "$GRAPH_K" =~ ^[0-9]+$ ]] || (( GRAPH_K < 2 )); then
        echo "GRAPH_K must be an integer >= 2, got: $GRAPH_K" >&2
        exit 2
    fi
}

# The shipped method at the operating point. Graph-defining flags only; the
# objective is whatever config/ggpkd_config.py ships.
method_graph_flags() {
    printf -- '--graph_k %s' "$GRAPH_K"
}

# The minimal per-anchor objective the comparison ladder is run on: one KL per
# anchor over that anchor's own columns, at that anchor's own temperature. No
# calibration term and no extra anchors, so nothing of ours can be credited with
# a gap that belongs to the choice of compared texts.
minimal_objective() {
    printf -- '--relation_target direct --no_ambient --row_weight 0 --batch_size %s' \
        "${BATCH_SIZE:-64}"
}
