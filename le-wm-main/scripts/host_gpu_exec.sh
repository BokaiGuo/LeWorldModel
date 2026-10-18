#!/usr/bin/env bash
set -euo pipefail

# Permanent host-side boundary for formal GPU experiments.
# Run this script from a host terminal, not from a managed sandbox that hides
# /dev/nvidia*. It intentionally has no CPU fallback.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

required_nodes=(
  /dev/nvidia0
  /dev/nvidiactl
  /dev/nvidia-uvm
  /dev/nvidia-uvm-tools
  /dev/nvidia-modeset
)
missing=()
for node in "${required_nodes[@]}"; do
  [[ -e "$node" ]] || missing+=("$node")
done
if (( ${#missing[@]} )); then
  printf 'GPU host guard failed: missing device nodes:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf 'Run from the host terminal or configure the sandbox device mapping; CPU fallback is forbidden.\n' >&2
  exit 20
fi

command -v nvidia-smi >/dev/null 2>&1 || {
  echo 'GPU host guard failed: nvidia-smi is not on PATH' >&2
  exit 21
}
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader >/dev/null

if [[ ! -r /proc/driver/nvidia/version ]]; then
  echo 'GPU host guard failed: /proc/driver/nvidia/version is not readable' >&2
  echo 'Expose /proc/driver/nvidia read-only to the execution environment.' >&2
  exit 22
fi
if ! command -v ldconfig >/dev/null 2>&1 || ! ldconfig -p 2>/dev/null | grep -q 'libcuda.so.1'; then
  echo 'GPU host guard failed: libcuda.so.1 is not visible through the dynamic linker' >&2
  echo 'Expose the host NVIDIA driver libraries or enable the NVIDIA container runtime.' >&2
  exit 23
fi

cd "$ROOT"
"$PYTHON" -m experiments.gpu_preflight \
  --device cuda \
  --render-smoke \
  --output "${GPU_PREFLIGHT_OUTPUT:-$ROOT/results/host_gpu_preflight.json}"

if (($# == 0)); then
  echo 'GPU host guard passed. No experiment command was supplied.'
  exit 0
fi

echo "[host-gpu] executing: $*"
exec "$@"
