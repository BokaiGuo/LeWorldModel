"""Bind the same v7 compiler to the existing PointMaze RGB feedback harness."""
import os
from pathlib import Path

import torch

from experiments import pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex as prior
from experiments import pc_v7_reacher_rgb_live_20261104_codex as adapter
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT, live = prior.ROOT, prior.live
PROTOCOL = ROOT.parent / "docs/PC_V7_POINTMAZE_RGB_LIVE_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_pointmaze_rgb_live_dev_20261104_codex"
GPU_UUID = prior.GPU_UUID


def configure():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prior.configure()
    previous = live.dependencies
    live.PROTOCOL, live.PREFIX = PROTOCOL, PREFIX
    live.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    live.ARMS, live.feedback = ("response",), adapter

    def dependencies():
        files = seal.capture([Path(__file__), PROTOCOL, Path(adapter.__file__),
                              Path(adapter.v7.__file__), Path(adapter.v6.__file__)])["files"]
        return {**previous(), **files}

    live.dependencies = dependencies


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("physical device 0 is required")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.lower().removeprefix("gpu-") != GPU_UUID.lower().removeprefix("gpu-"):
        raise ValueError("unexpected GPU UUID")
    configure()
    live.main()
