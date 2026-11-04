"""Run own-carry two-start CEM on the exposed v7 PointMaze episode."""
import json
import os
from pathlib import Path

import torch

from experiments import pc_v7_pointmaze_rgb_live_dev_20261104_codex as v7
from experiments.spectral_reachability import pc_response_temporal_two_start_cem_v1 as baseline
from experiments.spectral_reachability import pc_response_temporal_warm_cem_v1 as single
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT, live = v7.ROOT, v7.live
PROTOCOL = ROOT.parent / "docs/PC_V7_POINTMAZE_MATCHED_WARM_CEM_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex"
V7_MANIFEST = v7.PREFIX.with_name(v7.PREFIX.name + "_manifest.json")
V7_AUDIT = v7.PREFIX.with_name(v7.PREFIX.name + "_audit.json")
METHOD = baseline.METHOD


def decide(arm, root, step, target, predict, meter, carry=None):
    if arm != "response":
        raise ValueError("single CEM harness arm required")
    return baseline.decide("smooth_cem", root, step, target, predict, meter, carry)


def configure():
    v7.configure()
    previous = live.dependencies
    source = json.loads(V7_AUDIT.read_text())
    if source["manifest_sha256"] != live.old.prior.sha(V7_MANIFEST):
        raise ValueError("v7 audit binding mismatch")
    live.PROTOCOL, live.PREFIX = PROTOCOL, PREFIX
    live.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    live.ARMS, live.feedback = ("response",), __import__(__name__, fromlist=["decide"])

    def dependencies():
        files = seal.capture([Path(__file__), PROTOCOL, V7_MANIFEST, V7_AUDIT,
                              Path(baseline.__file__), Path(single.__file__)])["files"]
        return {**previous(), **files}

    live.dependencies = dependencies


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("physical device 0 is required")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.lower().removeprefix("gpu-") != v7.GPU_UUID.lower().removeprefix("gpu-"):
        raise ValueError("unexpected GPU UUID")
    configure()
    live.main()
