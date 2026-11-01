#!/usr/bin/env bash
# Wait for the pre-existing host download of the official Cube HDF5, then run
# its source receipt and CUDA-only P0.  The HDF5 open is a completion gate;
# a truncated stream never reaches the model invocation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/../data/cube_source_20261025/cube_single_expert.h5"
RECEIPT="${ROOT}/results/cube_d2p_training_source_20261025/cube_training_source.json"
OUTPUT="${ROOT}/results/cube_d2p_p0_preflight_source_native_20261025_cuda"

if [[ "${CUDA_VISIBLE_DEVICES+x}" == x && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty; refusing a CPU substitute." >&2
  exit 2
fi
if [[ -e "${RECEIPT}" || -e "${OUTPUT}" ]]; then
  echo "immutable Cube receipt or P0 output already exists; refusing overwrite" >&2
  exit 2
fi

for attempt in $(seq 1 720); do
  # ``tar`` creates a sparse file with the final logical length while the
  # stream is still arriving.  Do not let an HDF5 header open alone certify
  # completion: wait for the producer to leave and the receipt's allocation
  # check to confirm that HDF5 EOF is physically present.
  if pgrep -f '[w]get -O - https://hf-mirror.com/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst' >/dev/null \
    || pgrep -f 't[a]r --use-compress-program=unzstd -xvf - -C data/cube_source_20261025' >/dev/null; then
    echo "[$(date -Is)] official Cube stream producer is still active (${attempt}/720)" >&2
    sleep 60
    continue
  fi
  if "${ROOT}/.venv/bin/python" -c "from pathlib import Path; from experiments.spectral_reachability.cube_d2p_training_source import require_complete_hdf5; require_complete_hdf5(Path(r'${SOURCE}'))" >/dev/null 2>&1; then
    break
  fi
  if [[ "${attempt}" == 720 ]]; then
    echo "official Cube HDF5 did not become readable within 12 hours" >&2
    exit 3
  fi
  echo "[$(date -Is)] waiting for complete official Cube HDF5 (${attempt}/720)" >&2
  sleep 60
done

cd "${ROOT}"
"${ROOT}/.venv/bin/python" -m experiments.spectral_reachability.cube_d2p_training_source \
  --dataset "${SOURCE}" \
  --output "${RECEIPT}"

exec "${ROOT}/.venv/bin/python" -m experiments.spectral_reachability.preflight_cube_d2p_p0 \
  --device cuda \
  --output "${OUTPUT}" \
  --training-dataset "${SOURCE}" \
  --training-source-receipt "${RECEIPT}" \
  --policy /home/gbk/.stable_worldmodel/checkpoints/cube/lewm \
  --roots 4 \
  --seed 202610250 \
  --history-len 3 \
  --goal-offset-packets 1 \
  --future-horizon 2 \
  --probe-dims 8 \
  --fd-pairs 8 \
  --fd-epsilon 0.001
