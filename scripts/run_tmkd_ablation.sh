#!/bin/bash
set -euo pipefail

for mode in within full; do
    for coefficient in 0.25 0.5 1 2 4; do
        TMKD_MODE="$mode" \
        LAMBDA_TMKD="$coefficient" \
        SAVE_DIR="checkpoints/tmkd_${mode}_lambda_${coefficient}" \
        ./train_tmkd.sh
    done
done
