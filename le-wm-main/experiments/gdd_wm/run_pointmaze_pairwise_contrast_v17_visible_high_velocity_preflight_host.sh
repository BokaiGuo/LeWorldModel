#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" -m experiments.gdd_wm.run_pointmaze_pairwise_contrast_v17_visible_high_velocity_preflight
