#!/bin/bash
# Re-runs configs with extra training seeds (same val split): scripts/run_seeds.sh "1 2" e4_resnet18_pretrained ...
cd "$(dirname "$0")/.."
seeds=$1; shift
for c in "$@"; do for s in $seeds; do
  caffeinate -is .venv/bin/python train.py --config configs/$c.yaml --set train.seed=$s --tag seed$s --no-save \
    > results/logs/${c}_seed$s.log 2>&1 || echo "FAILED $c seed $s"
  tail -1 results/logs/${c}_seed$s.log
done; done
