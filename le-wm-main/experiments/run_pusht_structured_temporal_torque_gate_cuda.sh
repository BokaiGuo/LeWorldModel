#!/usr/bin/env bash
# Fresh, immutable GPU protocol for the low-dimensional temporal torque gate.
# This is deliberately separate from the failed v1 outcome panel.  Do not
# change root bands, model capacity, or confirmation thresholds after launch.
set -euo pipefail

if [[ "${PWD##*/}" != "le-wm-main" ]]; then
  echo "Run this launcher from the le-wm-main repository root." >&2
  exit 2
fi

output="results/pusht_structured_temporal_torque_gate_fresh_v2_20261021_cuda"
if [[ -e "$output" ]]; then
  echo "Refusing to reuse immutable protocol output: $output" >&2
  exit 3
fi

bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m \
  experiments.run_pusht_structured_temporal_torque_gate \
  --device cuda \
  --output "$output" \
  --train-per-class 128 \
  --calibration-per-class 64 \
  --test-per-class 128 \
  --pool-transitions 8192 \
  --grid-size 4 \
  --pca-rank 24 \
  --epochs 180 \
  --learning-rate 0.01 \
  --weight-decay 0.10 \
  --bootstrap 10000 \
  --minimum-calibration-ba 0.65 \
  --minimum-calibration-coverage 0.25 \
  --minimum-calibration-per-class 12 \
  --minimum-outcome-ba 0.70 \
  --minimum-outcome-ci-lower 0.60 \
  --minimum-outcome-coverage 0.25
