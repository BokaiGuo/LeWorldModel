#!/usr/bin/env bash
# Host-only fresh-root static audit; no learned model and no outcome access.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
cd "${ROOT}"
[[ -x "${PYTHON}" ]] || { echo "missing project Python: ${PYTHON}" >&2; exit 5; }
exec "${PYTHON}" -m experiments.gdd_wm.audit_pointmaze_pairwise_contrast_v13_position_alias_preflight
