#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" -m experiments.decision_benchmark.build_pointmaze_v20_budgeted_mixed_velocity_pool
