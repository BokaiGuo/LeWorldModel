"""Fail-closed CUDA preflight for learned-world-model experiments.

``torch.cuda.is_available()`` is not sufficient on hosts with an NVML/kernel
driver mismatch.  This probe performs the operation that the experiment
actually needs: CUDA initialization, device-name lookup, and a real tensor
allocation.  A failed allocation exits non-zero and must stop the caller.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def visible_nvidia_nodes() -> list[str]:
    """Return NVIDIA device nodes visible to this process.

    A host can have a healthy driver while a managed sandbox hides the device
    namespace.  Reporting this separately makes that failure actionable and
    prevents confusing it with a broken PyTorch/CUDA installation.
    """
    candidates = (
        "/dev/nvidia0",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-modeset",
    )
    return [path for path in candidates if os.path.exists(path)]


def render_smoke() -> dict[str, object]:
    """Verify the EGL/MuJoCo render path used by visual GPU panels."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401 - registers environments

    env = gym.make(
        "swm/OGBCube-v0", ob_type="pixels", render_mode="rgb_array",
        terminate_at_goal=False,
    )
    try:
        observation, _info = env.reset(
            seed=20261900, options={"task_id": 1, "render_goal": True},
        )
        frame = env.render()
        if frame is None or getattr(frame, "ndim", 0) != 3:
            raise RuntimeError("EGL render returned no RGB frame")
        return {
            "status": "passed",
            "observation_type": type(observation).__name__,
            "frame_shape": [int(v) for v in frame.shape],
            "frame_dtype": str(frame.dtype),
        }
    finally:
        env.close()


def require_cuda_allocation(device: str | torch.device = "cuda") -> dict[str, object]:
    """Require a real CUDA allocation and return resolved runtime metadata.

    NVML visibility and ``torch.cuda.is_available()`` are not sufficient when
    device nodes or CUDA context creation are broken. Formal experiments call
    this before collecting scientific rows, so they fail closed instead of
    silently producing CPU or partial-GPU evidence.
    """
    requested = torch.device(device)
    if requested.type != "cuda":
        raise RuntimeError(
            "formal learned-world-model experiments require CUDA; "
            f"received {requested}"
        )
    device_nodes = visible_nvidia_nodes()
    if not device_nodes:
        raise RuntimeError(
            "CUDA was requested but no NVIDIA device nodes are visible to this "
            "process. The host may have a working GPU, but this sandbox does "
            "not expose /dev/nvidia*. Run from a host terminal or map the "
            "NVIDIA device nodes into the sandbox and restart the app."
        )
    try:
        torch.cuda.init()
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        probe = torch.ones((2, 2), device=requested)
        torch.cuda.synchronize(index)
        allocation_sum = float(probe.sum().item())
    except Exception as exc:  # pragma: no cover - exercised on broken hosts
        raise RuntimeError(
            "CUDA was requested but a real CUDA allocation failed; refusing "
            "CPU fallback. Check /dev/nvidia*, driver state, and host runtime. "
            f"Original error: {exc}"
        ) from exc
    return {
        "device": str(requested),
        "index": int(index),
        "name": str(name),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "allocation_sum": allocation_sum,
        "visible_nvidia_nodes": device_nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--render-smoke", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("gpu_preflight requires a CUDA device; use no probe for explicit CPU runs")

    try:
        runtime = require_cuda_allocation(device)
        result = {
            "status": "passed",
            **runtime,
        }
    except Exception as exc:
        # Preserve a machine-readable failure artifact even when the host
        # mapping is broken.  This is deliberately still a non-zero exit:
        # callers must not interpret a recorded failure as GPU evidence or
        # fall back to CPU for a formal run.
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "visible_nvidia_nodes": visible_nvidia_nodes(),
        }
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(1)
    if args.render_smoke:
        result["render_smoke"] = render_smoke()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
