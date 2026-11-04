"""Infrastructure-only checks before creating a future experiment manifest.

This module does not start a panel or persist results. A caller must invoke
``run_before_artifacts`` before creating output directories or manifests.
It does not certify model inputs, task feasibility, or scientific claims.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Callable, TypeVar

T = TypeVar("T")


def cuda_probe(expected_uuid: str) -> dict:
    from experiments.gpu_preflight import require_cuda_allocation
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal the declared GPU UUID")
    result = require_cuda_allocation("cuda:0")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.removeprefix("GPU-").lower() != expected_uuid.removeprefix("GPU-").lower():
        raise RuntimeError(f"CUDA device UUID mismatch: {actual}")
    return {**result, "uuid": actual}


def isolation_probe() -> dict:
    # Same namespace requirements as the isolated JEPA workers; no network
    # request and no changes to the worker's own filesystem policy.
    command = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-net",
               "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
               "/bin/true"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if completed.returncode:
        raise RuntimeError(f"bwrap namespace preflight failed: {completed.stderr.strip()}")
    return {"status": "passed", "scope": "namespace creation only"}


def reacher_render_probe() -> dict:
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl before launch")
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup

    setup.install_missing_field_compat()
    env = gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
        if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[-1] != 3:
            raise RuntimeError("Reacher render did not return an RGB frame")
        return {"status": "passed", "frame_shape": list(frame.shape), "dtype": str(frame.dtype)}
    finally:
        env.close()


def run_before_artifacts(probes: list[Callable[[], object]], start: Callable[[], T]) -> T:
    """Call start only if all checks succeed; start owns all experiment writes."""
    for probe in probes:
        probe()
    return start()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--reacher-render", action="store_true")
    parser.add_argument("--isolation", action="store_true")
    args = parser.parse_args()
    result = {"cuda": cuda_probe(args.gpu_uuid)}
    if args.isolation:
        result["isolation"] = isolation_probe()
    if args.reacher_render:
        result["reacher_render"] = reacher_render_probe()
    print(json.dumps({"status": "passed", "checks": result}, sort_keys=True))


if __name__ == "__main__":
    main()
