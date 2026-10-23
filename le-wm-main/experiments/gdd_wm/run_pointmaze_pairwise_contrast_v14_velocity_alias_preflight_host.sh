#!/usr/bin/env bash
# Host-only static gate.  It must finish before the V14 CUDA runner is started.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" -m experiments.gdd_wm.run_pointmaze_pairwise_contrast_v14_velocity_alias_preflight
