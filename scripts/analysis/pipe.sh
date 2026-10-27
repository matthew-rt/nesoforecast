#!/bin/zsh
# usage: pipe.sh <NWP_DAY>  -> runs the fair pipeline with the learned unconstrained target
setopt pipefail
cd /Users/matt/code/nesoforecast/scripts
export NWP_DAY=$1 UNC_TARGET=1
for s in "residual.py --rebuild --pooled-only" nn_baselines.py fleet_nn.py benchmark.py; do
  echo "=== $s ($(date +%H:%M))"
  python3 ${=s} 2>&1 | grep -v -i warn || { echo "FAILED $s"; exit 1; }
done
echo "=== done $(date +%H:%M)"
