#!/usr/bin/env bash
# The unablated model, one run per seed. Every P0 script starts with these, so
# running this first is also the warm-up that builds the teacher-embedding cache
# and the base graph artifact exactly once, before anything runs in parallel.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for seed in ${SEEDS}; do
    run_full "${seed}"
done
