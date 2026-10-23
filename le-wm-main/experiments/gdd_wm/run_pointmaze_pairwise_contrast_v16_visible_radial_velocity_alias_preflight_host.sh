#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" -m experiments.gdd_wm.run_pointmaze_pairwise_contrast_v16_visible_radial_velocity_alias_preflight
