#!/usr/bin/env bash
# Run a CUDA experiment with the cuDNN shipped alongside this virtualenv.
# Some hosts export an older system cuDNN first in LD_LIBRARY_PATH: tensor
# allocation then succeeds, but cuDNN RNN/GRU kernels fail at runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLED_CUDNN="${ROOT}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [[ ! -f "${BUNDLED_CUDNN}/libcudnn.so.9" ]]; then
  echo "Bundled cuDNN not found at ${BUNDLED_CUDNN}" >&2
  exit 2
fi
export LD_LIBRARY_PATH="${BUNDLED_CUDNN}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
