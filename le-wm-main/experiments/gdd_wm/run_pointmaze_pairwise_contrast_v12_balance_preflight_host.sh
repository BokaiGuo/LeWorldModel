#!/usr/bin/env bash
# Host-only static C1 admission. This creates fresh PointMaze roots and tests
# the balanced-reversal candidate bank; it does not fit a model or open labels.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
cd "${ROOT}"
[[ -x "${PYTHON}" ]] || { echo "missing project Python: ${PYTHON}" >&2; exit 5; }
exec "${PYTHON}" -m experiments.gdd_wm.run_pointmaze_pairwise_contrast_v12_balance_preflight
