#!/usr/bin/env bash
# Wait only for the independently launched official-source validator, then
# exec the CUDA-only official repair pipeline.  This avoids accidentally
# starting a training run from a partial archive while preserving fail-closed
# behavior if source acquisition terminates without a validation manifest.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SOURCE_ROOT="${1:-${ROOT}/data/reacher_official_source_20261024}"
RUN_ROOT="${2:-${ROOT}/results/reacher_d2p_official_causal_repair_20261024_cuda}"
SOURCE_UNIT="${3:-lewm-reacher-official-source-20261024.service}"
MANIFEST="${SOURCE_ROOT}/official_source_manifest.json"

while [[ ! -f "${MANIFEST}" ]]; do
  if ! systemctl --user is-active --quiet "${SOURCE_UNIT}"; then
    echo "official Reacher source did not validate; refusing CUDA repair without ${MANIFEST}" >&2
    exit 3
  fi
  sleep 30
done

exec bash "${ROOT}/experiments/spectral_reachability/run_reacher_d2p_official_causal_repair_cuda.sh" \
  "${SOURCE_ROOT}" "${RUN_ROOT}"
