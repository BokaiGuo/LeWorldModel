#!/usr/bin/env bash
# CUDA-only V7 entry point.  The Python driver keeps the process that passed
# GPU preflight alive through build/calibrate/score, avoiding shell-level
# library rewrites that can hide managed NVIDIA device nodes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" -m \
  experiments.decision_benchmark.run_pointmaze_counterfactual_contrast_v7_cuda
