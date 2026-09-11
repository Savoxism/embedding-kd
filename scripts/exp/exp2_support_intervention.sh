#!/usr/bin/env bash
# E2 -- Support intervention: is teacher relevance the thing that matters?
#
# E1 shows batch composition changes what an in-batch relational objective learns.
# The obvious reply is "then just take relations from outside the batch". This
# experiment answers that: it holds the objective, the support size, the
# temperature and the budget fixed, and changes only *which* relations each anchor
# is scored against.
#
#   in_batch          the batch, as conventional relational KD has it
#   corpus_uniform    drawn uniformly from the whole corpus -- beyond the batch,
#                     but not teacher-informed
#   rewired           degree-matched rewiring: each anchor keeps the number of
#                     columns its own transition row has, the endpoints are
#                     redrawn. Graph structure without graph semantics
#   student_knn       the *base student's* own nearest neighbours: semantic-
#                     looking support that the teacher did not choose. Optional,
#                     see WITH_STUDENT_KNN below
#   teacher_topk      teacher top-k, no mutual filter
#   teacher_mutual    the method's support: teacher mutual-kNN
#
# The reading that carries the paper:
#
#   in_batch ~ corpus_uniform ~ rewired  <<  teacher_topk ~ teacher_mutual
#
# because it excludes, in one table, "more relations", "cross-batch relations"
# and "graph structure" as the explanation, leaving teacher relevance.
#
# Every arm runs the same minimal objective as E1 -- one KL per anchor over that
# anchor's own columns, at that anchor's own tau_i, with no ambient scale and no
# auxiliary row loss. Three consequences worth knowing before reading the CSV:
#
# * `--relation_target direct` is not optional for the random-support arms. A
#   column the graph puts no mass on has diffusion target exactly zero, so under
#   the method's own target those arms would have no objective at all. The
#   teacher's raw cosine is defined for every pair, so it is what makes them
#   controls rather than deletions -- and the teacher arms run at `direct` too,
#   so the comparison stays one-factor.
# * `--row_weight 0` everywhere. L_row derives its row set from the support draw,
#   so leaving it on would make the arms differ in how much supervision they
#   carry as well as in which relations.
# * The arms are matched on support *size*, not on compute. A corpus-uniform draw
#   deduplicates far worse across a batch than a teacher draw does, so it encodes
#   more texts per step. The CSV carries `train_encoded_texts_cum` and
#   `cost_wall_seconds` for exactly this reason: the cost difference is a
#   consequence of the design and belongs in the table, not hidden by shrinking
#   the budget.
#
# HOLDOUT_FRAC withholds a symmetric random subset of teacher edges from every
# arm's support, so E3 can score all of these checkpoints on relations no arm was
# trained on. It defaults to 0.2, which means this family's absolute numbers are
# not comparable with the paper's main table -- it is a controlled study, and the
# comparison that matters is between its own arms. Set HOLDOUT_FRAC=0 for a
# family whose numbers line up with the main table, at the cost of E3's
# masked-neighbour metric.
#
# Usage:
#   bash scripts/exp/exp2_support_intervention.sh
#   DRY_RUN=1 bash scripts/exp/exp2_support_intervention.sh
#   PAIRS=qwen3_0_6b_to_minilmv2_h384,bge_m3_to_minilmv2_h768 bash scripts/exp/exp2_support_intervention.sh
#   WITH_STUDENT_KNN=1 bash scripts/exp/exp2_support_intervention.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

EXPERIMENT="exp2_support"
CORPUS="${CORPUS:-data/train_set/merged_3_data_5k_each.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SUPPORT_SIZE="${SUPPORT_SIZE:-$((BATCH_SIZE - 1))}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.2}"
HOLDOUT_SEED="${HOLDOUT_SEED:-12345}"
WITH_STUDENT_KNN="${WITH_STUDENT_KNN:-0}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache/exp}"
BASE_RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"

IFS=',' read -r -a PAIR_LIST <<< "${PAIRS:-qwen3_0_6b_to_minilmv2_h384}"

HOLDOUT_FLAGS="--holdout_edge_frac $HOLDOUT_FRAC --holdout_seed $HOLDOUT_SEED"
# Two graph keys because knn_mode is a property of the artifact, not of the run:
# the two teacher arms must not share one, or the second would load the first's
# graph and the "mutual vs directed" row of the table would be a duplicate.
GRAPH_SPEC="mutual|$CORPUS|--knn_mode mutual $HOLDOUT_FLAGS
directed|$CORPUS|--knn_mode directed $HOLDOUT_FLAGS"

MINIMAL="--relation_target direct --no_ambient --row_weight 0 --batch_size $BATCH_SIZE"
BUDGET="--diffusion_quota $SUPPORT_SIZE"

ARMS_SPEC="in_batch|mutual|ggpkd|--batch_local $MINIMAL
corpus_uniform|mutual|ggpkd|$MINIMAL $BUDGET --support_policy corpus_uniform
rewired|mutual|ggpkd|$MINIMAL $BUDGET --support_policy rewired
teacher_topk|directed|ggpkd|$MINIMAL $BUDGET
teacher_mutual|mutual|ggpkd|$MINIMAL $BUDGET"

overall=0
for PAIR in "${PAIR_LIST[@]}"; do
    echo
    echo "##############################################"
    echo "# E2 support intervention -- pair $PAIR"
    echo "##############################################"

    CORPUS_KEY="$(basename "${CORPUS%.*}")"
    PAIR_ARMS="$ARMS_SPEC"

    if [[ "$WITH_STUDENT_KNN" == "1" ]]; then
        # The student-kNN arm needs a graph built from the *base* student's own
        # embeddings. It is built here rather than declared in GRAPH_SPEC because
        # it comes from a different encoder entirely, and it is opt-in because it
        # carries a caveat the other arms do not: the artifact's row bandwidths
        # tau_i are then the student's, not the teacher's, so this arm differs
        # from the teacher arms in temperature as well as in support. Read it as
        # indicative, and say so in the paper.
        STUDENT_GRAPH="$CACHE_ROOT/$PAIR/$CORPUS_KEY/graph_student_knn.pt"
        if [[ ! -f "$STUDENT_GRAPH" ]]; then
            echo "Building the base-student kNN graph for $PAIR"
            "$PYTHON_BIN" "$SCRIPT_DIR/student_graph.py" \
                --pair "$PAIR" \
                --train-data "$CORPUS" \
                --out "$STUDENT_GRAPH" \
                --holdout-frac "$HOLDOUT_FRAC" \
                --holdout-seed "$HOLDOUT_SEED"
        fi
        # It joins as its own graph key whose "build" is a no-op: the artifact
        # already exists on disk, and the runner's prepare step will load it
        # rather than rebuild, because the metadata matches.
        PAIR_ARMS+="
student_knn|student_knn|ggpkd|$MINIMAL $BUDGET"
        GRAPH_SPEC_PAIR="$GRAPH_SPEC
student_knn|$CORPUS|--knn_mode mutual $HOLDOUT_FLAGS"
    else
        GRAPH_SPEC_PAIR="$GRAPH_SPEC"
    fi

    GRAPH_SPEC="$GRAPH_SPEC_PAIR" ARMS_SPEC="$PAIR_ARMS" \
    RUN_ID="$BASE_RUN_ID/$PAIR" \
    CSV_OUT="$REPO_ROOT/runs/$EXPERIMENT/results_$PAIR.csv" \
    PAIR="$PAIR" \
        run_arms "$REPO_ROOT" "$EXPERIMENT" || overall=$?
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit $overall
fi

# One frame for the analysis, regardless of how many pairs ran.
"$PYTHON_BIN" - "$REPO_ROOT/runs/$EXPERIMENT" <<'PY'
import csv, sys
from pathlib import Path

root = Path(sys.argv[1])
parts = sorted(root.glob("results_*.csv"))
if not parts:
    raise SystemExit("no per-pair CSVs to merge")
rows, fields = [], []
for part in parts:
    with part.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for name in reader.fieldnames or []:
            if name not in fields:
                fields.append(name)
        rows.extend(reader)
out = root / "results.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"merged {len(parts)} pair CSVs -> {out} ({len(rows)} rows)")
PY

echo
echo "E2 outputs:"
echo "  $REPO_ROOT/runs/$EXPERIMENT/results.csv"
echo "Next: bash scripts/exp/exp3_heldout_geometry.sh"
exit $overall
