#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" -m experiments.decision_benchmark.build_pointmaze_goal_radial_visible_high_velocity_pool
