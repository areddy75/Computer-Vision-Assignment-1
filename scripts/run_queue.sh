#!/bin/bash
# Runs a list of configs sequentially: scripts/run_queue.sh e1_scenecnn e2_scenecnn_aug ...
cd "$(dirname "$0")/.."
for c in "$@"; do
  .venv/bin/python train.py --config configs/$c.yaml > results/logs/$c.log 2>&1 || echo "FAILED $c"
  tail -1 results/logs/$c.log
done
