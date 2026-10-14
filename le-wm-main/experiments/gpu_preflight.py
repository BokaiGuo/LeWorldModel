"""Fail-closed CUDA preflight for learned-world-model experiments.

``torch.cuda.is_available()`` is not sufficient on hosts with an NVML/kernel
driver mismatch.  This probe performs the operation that the experiment
actually needs: CUDA initialization, device-name lookup, and a real tensor
allocation.  A failed allocation exits non-zero and must stop the caller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("gpu_preflight requires a CUDA device; use no probe for explicit CPU runs")

    torch.cuda.init()
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    probe = torch.ones((2, 2), device=device)
    result = {
        "status": "passed",
        "device": str(device),
        "index": int(index),
        "name": name,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "allocation_sum": float(probe.sum().item()),
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
