#!/usr/bin/env bash
# Fresh, immutable host-CUDA experiment.  The Codex sandbox is not a CUDA
# execution environment; run this launcher from the host via systemd-run.
set -euo pipefail

if [[ "${PWD##*/}" != "le-wm-main" ]]; then
  echo "Run this launcher from the le-wm-main repository root." >&2
  exit 2
fi

output="results/pusht_action_conditioned_signed_torque_confirmation_v3_bodyframe_20261022_cuda"
if [[ -e "$output" ]]; then
  echo "Refusing to reuse immutable protocol output: $output" >&2
  exit 3
fi

bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m experiments.gpu_preflight --device cuda --recurrent-smoke
bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m \
  experiments.run_pusht_action_conditioned_signed_torque_confirmation \
  --device cuda --output "$output" \
  --train-per-class 128 --calibration-per-class 64 --test-per-class 128 \
  --train-seed-band 1706210000 --calibration-seed-band 1706220000 --test-seed-band 1706230000 \
  --pool-transitions 8192 --history-steps 4 --history-magnitude 0.30 \
  --contact-inward-magnitude 0.18 --contact-tangent-magnitude 0.55 \
  --epochs 240 --learning-rate 0.02 --weight-decay 0.10 --bootstrap 10000 \
  --minimum-balanced-accuracy 0.70 --minimum-ci-lower 0.60 \
  --confirmation-representation geometry_plus_rigid_body_mechanics
