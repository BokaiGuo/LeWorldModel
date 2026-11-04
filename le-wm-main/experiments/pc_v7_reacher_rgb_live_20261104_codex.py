"""v7 fixed-terminal response compiler on the existing Reacher live harness."""

import sys
from pathlib import Path

import torch

from experiments import pc_v5_reacher_rgb_live_20261104_codex as prior
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_target_reach_paired_abstention_v7 as v7
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT, live = prior.ROOT, prior.live
PROTOCOL = ROOT.parent / "docs/PC_V7_REACHER_RGB_LIVE_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_reacher_rgb_live_dev_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
METHOD = v7.METHOD


def decide(arm, root, step, target, predict, resource, carry=None):
    return v7.decide("response", arm, root, step, target, predict, resource, core.NOMINALS, carry)


def configure():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prior.configure()
    old_dependencies = live.dependencies
    live.PROTOCOL, live.PREFIX, live.MANIFEST = PROTOCOL, PREFIX, MANIFEST
    live.method, live.ARMS = sys.modules[__name__], ("response",)

    def dependencies(source):
        extra = seal.capture([Path(__file__), PROTOCOL, Path(v7.__file__), Path(v6.__file__)])["files"]
        return {**old_dependencies(source), **extra}

    live.dependencies = dependencies


if __name__ == "__main__":
    configure()
    live.main()
