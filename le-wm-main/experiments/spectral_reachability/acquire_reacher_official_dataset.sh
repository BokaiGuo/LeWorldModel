#!/usr/bin/env bash
# Acquire and validate the public Reacher training source from an HF mirror.
# Extraction/validation are data-preparation operations only; learned-model
# training and all P0/P1 evaluation remain CUDA-only in the paired launcher.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SOURCE_ROOT="${1:-${ROOT}/data/reacher_official_source_20261024}"
ARCHIVE="${SOURCE_ROOT}/reacher.tar.zst"
EXTRACTED="${SOURCE_ROOT}/extracted"
MANIFEST="${SOURCE_ROOT}/official_source_manifest.json"

if [[ -f "${MANIFEST}" ]]; then
  echo "official Reacher source is already validated: ${MANIFEST}"
  exit 0
fi

mkdir -p "${SOURCE_ROOT}"
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download quentinll/lewm-reacher reacher.tar.zst \
  --repo-type dataset \
  --local-dir "${SOURCE_ROOT}"

if [[ ! -s "${ARCHIVE}" ]]; then
  echo "official Reacher archive is missing or empty: ${ARCHIVE}" >&2
  exit 2
fi
if [[ -e "${EXTRACTED}" ]]; then
  echo "refusing to mix a new extraction with an unvalidated extracted directory: ${EXTRACTED}" >&2
  exit 2
fi
mkdir -p "${EXTRACTED}"
tar --zstd -xf "${ARCHIVE}" -C "${EXTRACTED}"

mapfile -t H5_FILES < <(find "${EXTRACTED}" -type f -name '*.h5' -print | sort)
if [[ "${#H5_FILES[@]}" -ne 1 ]]; then
  echo "expected exactly one HDF5 dataset in official archive, found ${#H5_FILES[@]}" >&2
  exit 2
fi
DATASET="${H5_FILES[0]}"
.venv/bin/python - "${ARCHIVE}" "${DATASET}" "${MANIFEST}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import h5py

archive, dataset, manifest = map(Path, sys.argv[1:])
with h5py.File(dataset, "r") as source:
    names = []
    source.visit(names.append)
if not any(name.split("/")[-1] == "pixels" for name in names):
    raise RuntimeError("official Reacher HDF5 lacks a pixels field")
if not any(name.split("/")[-1] == "action" for name in names):
    raise RuntimeError("official Reacher HDF5 lacks an action field")

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

manifest.write_text(json.dumps({
    "schema": "reacher_official_source.v1",
    "repository": "quentinll/lewm-reacher",
    "mirror": "https://hf-mirror.com",
    "archive": str(archive.resolve()),
    "archive_bytes": archive.stat().st_size,
    "archive_sha256": digest(archive),
    "dataset": str(dataset.resolve()),
    "dataset_bytes": dataset.stat().st_size,
    "dataset_sha256": digest(dataset),
    "status": "archive_and_hdf5_validated",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "official Reacher dataset validated: ${MANIFEST}"
