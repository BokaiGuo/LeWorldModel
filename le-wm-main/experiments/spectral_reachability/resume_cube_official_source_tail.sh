#!/usr/bin/env bash
# Rebuild a truncated official Cube HDF5 without ever materialising its 46-GB
# compressed archive.  The archive uses GNU sparse members: stdout extraction
# omits holes, so byte-tail append is invalid.  The verified-incomplete target
# is therefore replaced at the file level by GNU tar extraction.  We
# deliberately *do not* pass --sparse: the project admission contract requires
# a fully allocated HDF5 extent, not a logically correct sparse reconstruction.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/../data/cube_source_20261025/cube_single_expert.h5"
REMOTE="https://hf-mirror.com/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst?download=true"
RESERVE_BYTES=1073741824
LOG="${ROOT}/../data/cube_source_20261025/cube_sparse_rebuild.log"

if [[ ! -f "${SOURCE}" ]]; then
  echo "Cube source is absent: ${SOURCE}" >&2
  exit 2
fi

read -r CURRENT EXPECTED < <(
  "${ROOT}/.venv/bin/python" -c '
from pathlib import Path
import h5py
p = Path("'"${SOURCE}"'")
with h5py.File(p, "r", swmr=True) as handle:
    print(p.stat().st_size, int(handle.id.get_filesize()))
'
)
if (( CURRENT >= EXPECTED )); then
  if (( CURRENT != EXPECTED )); then
    echo "Cube source length exceeds its HDF5 EOF; refusing to append" >&2
    exit 3
  fi
  if "${ROOT}/.venv/bin/python" -c '
from pathlib import Path
from experiments.spectral_reachability.cube_d2p_training_source import require_complete_hdf5
require_complete_hdf5(Path("'"${SOURCE}"'"))
'; then
    echo "Cube source already passes its HDF5 extent gate; no rebuild needed." >&2
    exit 0
  fi
  echo "Cube source reaches its logical EOF but has incomplete sparse extents; rebuilding." >&2
fi

AVAILABLE=$(df -B1 --output=avail "${SOURCE}" | tail -n 1 | tr -d ' ')
# ``st_blocks`` is released when tar opens the exact target with --overwrite.
# Reserve the full logical HDF5 size, not merely its current missing tail,
# because sparse archive placement is not a continuous byte suffix.
ALLOCATED=$(stat -c '%b' "${SOURCE}")
ALLOCATED=$((ALLOCATED * 512))
if (( AVAILABLE + ALLOCATED < EXPECTED + RESERVE_BYTES )); then
  echo "Rebuild needs ${EXPECTED} bytes plus ${RESERVE_BYTES}-byte reserve after releasing ${ALLOCATED}; only $((AVAILABLE + ALLOCATED)) available." >&2
  exit 4
fi

echo "[$(date -Is)] replacing verified-truncated Cube source: current=${CURRENT}, hdf5_eof=${EXPECTED}" | tee -a "${LOG}"
# This is intentionally not ``tar -xOf``: an archive sparse map requires
# seekable output so tar can place extents at their declared offsets.  No
# compressed archive is retained on disk.  Omitting --sparse makes GNU tar
# materialise holes so the extent check detects no false completion.
wget --progress=dot:giga -O - "${REMOTE}" 2>>"${LOG}" | \
  tar --use-compress-program=unzstd --overwrite -xvf - -C "$(dirname "${SOURCE}")" cube_single_expert.h5 >>"${LOG}" 2>&1
PIPE_STATUS=("${PIPESTATUS[@]}")
FINAL=$(stat -c '%s' "${SOURCE}")
if (( FINAL != EXPECTED )); then
  echo "[$(date -Is)] incomplete full-extent rebuild: final=${FINAL}, expected=${EXPECTED}, pipeline=${PIPE_STATUS[*]}" | tee -a "${LOG}" >&2
  exit 6
fi
"${ROOT}/.venv/bin/python" -c '
from pathlib import Path
from experiments.spectral_reachability.cube_d2p_training_source import require_complete_hdf5
print(require_complete_hdf5(Path("'"${SOURCE}"'")))
'
echo "[$(date -Is)] Cube full-extent rebuild complete; final=${FINAL}, pipeline=${PIPE_STATUS[*]}" | tee -a "${LOG}"
