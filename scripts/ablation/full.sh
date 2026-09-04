#!/usr/bin/env bash
# Build the shared teacher/graph caches and train the canonical method.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"
done
