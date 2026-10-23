#!/usr/bin/env bash
# Fresh CUDA-only signed torque-response confirmation.  It never reopens the
# absolute-response panel and refuses a CPU fallback through the common wrapper.
set -euo pipefail

if [[ "${PWD##*/}" != "le-wm-main" ]]; then
  echo "Run this launcher from the le-wm-main repository root." >&2
  exit 2
fi

output="results/pusht_signed_temporal_torque_confirmation_contact_frame_v2_20261022_cuda"
if [[ -e "$output" ]]; then
  echo "Refusing to reuse immutable protocol output: $output" >&2
  exit 3
fi

bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m \
  experiments.run_pusht_signed_temporal_torque_confirmation \
  --device cuda --output "$output" \
  --train-per-class 64 --calibration-per-class 32 --test-per-class 64 \
  --pool-transitions 8192 --history-steps 4 --history-magnitude 0.30 \
  --contact-inward-magnitude 0.18 --contact-tangent-magnitude 0.55 \
  --image-size 16 --epochs 240 --learning-rate 0.02 --weight-decay 0.02 \
  --bootstrap 10000 --minimum-balanced-accuracy 0.70 --minimum-ci-lower 0.60
