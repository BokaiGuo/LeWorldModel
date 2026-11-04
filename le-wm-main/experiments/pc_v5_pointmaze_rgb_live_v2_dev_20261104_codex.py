"""New prefix for legacy MuJoCo's integer CUDA device environment."""
import os
from pathlib import Path

import torch

from experiments import pc_v5_pointmaze_rgb_live_dev_20261104_codex as prior

ROOT, live = prior.ROOT, prior.live
PROTOCOL = ROOT.parent / "docs/PC_V5_POINTMAZE_RGB_LIVE_V2_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex"
GPU_UUID = "GPU-4e8973ff-0278-53f3-2bae-2615d461e441"


def configure():
    prior.configure()
    previous = live.dependencies
    live.PROTOCOL, live.PREFIX = PROTOCOL, PREFIX
    live.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")

    def dependencies():
        paths = (Path(__file__), PROTOCOL)
        return {**previous(), **{str(p.resolve()): live.old.prior.sha(p) for p in paths}}

    live.dependencies = dependencies


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("physical device 0 is required")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.lower().removeprefix("gpu-") != GPU_UUID.lower().removeprefix("gpu-"):
        raise ValueError("unexpected GPU UUID")
    configure()
    live.main()
