#!/usr/bin/env bash
# Fresh CUDA-only candidate-bank signed torque confirmation.  It tests the
# exact boundary-frame pulse semantics used by the learned PushT candidate bank.
set -euo pipefail

if [[ "${PWD##*/}" != "le-wm-main" ]]; then
  echo "Run this launcher from the le-wm-main repository root." >&2
  exit 2
fi

output="results/pusht_candidate_bank_signed_temporal_torque_confirmation_v3_20261022_cuda"
if [[ -e "$output" ]]; then
  echo "Refusing to reuse immutable protocol output: $output" >&2
  exit 3
fi

attempts="${GPU_PREFLIGHT_ATTEMPTS:-3}"
if [[ "$attempts" -lt 1 ]]; then
  echo "GPU_PREFLIGHT_ATTEMPTS must be positive" >&2
  exit 2
fi

for ((attempt = 1; attempt <= attempts; attempt++)); do
  if .venv/bin/python -m experiments.gpu_preflight --device cuda; then
    exec bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m \
      experiments.run_pusht_candidate_bank_signed_temporal_torque_confirmation \
      --device cuda --output "$output" \
      --train-per-class 64 --calibration-per-class 32 --test-per-class 64 \
      --pool-transitions 8192 --history-steps 4 --history-magnitude 0.30 \
      --contact-inward-magnitude 0.08 --contact-tangent-magnitude 0.16 \
      --deployed-tangent-magnitudes 0.06 0.10 0.16 \
      --deployed-contact-normal-magnitudes 0.02 0.04 0.08 \
      --image-size 16 --epochs 240 --learning-rate 0.02 --weight-decay 0.02 \
      --bootstrap 10000 --minimum-balanced-accuracy 0.70 --minimum-ci-lower 0.60 \
      --minimum-gate-calibration-ba 0.70 --minimum-gate-coverage 0.25 --minimum-gate-per-class 8
  fi
  echo "CUDA preflight attempt ${attempt}/${attempts} failed; no CPU fallback will be used." >&2
done

echo "CUDA preflight failed after ${attempts} attempts; candidate-bank signed confirmation was not run." >&2
exit 1
