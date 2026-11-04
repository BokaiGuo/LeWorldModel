"""Two-start CEM with the same numerical settings as the v7 Reacher run."""

import json
from pathlib import Path

import torch

from experiments import pc_v4_reacher_two_start_cem_rgb_20261104_codex as prior
from experiments import pc_v7_reacher_rgb_live_20261104_codex as v7
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

live, ROOT = prior.live, prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_REACHER_MATCHED_WARM_CEM_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_reacher_matched_warm_cem_dev_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")


def configure():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prior.configure()
    previous = live.dependencies
    sha = live.host.bridge.sha_file
    v7_audit = v7.PREFIX.with_name(v7.PREFIX.name + "_audit.json")
    v7_manifest = json.loads(v7.MANIFEST.read_text())
    if json.loads(v7_audit.read_text())["manifest_sha256"] != sha(v7.MANIFEST):
        raise ValueError("v7 record audit mismatch")
    for name, digest in v7_manifest["dependencies"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"v7 dependency changed: {name}")
    live.PROTOCOL, live.PREFIX, live.MANIFEST = PROTOCOL, PREFIX, MANIFEST

    def dependencies(source):
        return {**previous(source), **seal.capture(
            [Path(__file__), PROTOCOL, v7.MANIFEST, v7_audit])["files"]}

    live.dependencies = dependencies


if __name__ == "__main__":
    configure()
    live.main()
