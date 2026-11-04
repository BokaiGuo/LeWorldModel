"""Single-process v4 configuration of the sealed Reacher RGB executor."""
import json
from pathlib import Path

from experiments import pc_response_temporal_v3_reacher_rgb_live_20261104_codex as live
from experiments import pc_v4_reacher_saved_context_20261104_codex as saved
from experiments.spectral_reachability import d2p_local_compiler as local
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as method
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = live.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_REACHER_RGB_LIVE_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_reacher_rgb_live_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
original_dependencies = live.dependencies


def configure():
    """Configure only this isolated process; do not use concurrently in-process."""
    sha = live.host.bridge.sha_file
    prior_manifest = json.loads(saved.MANIFEST.read_text())
    prior_audit = json.loads(saved.PREFIX.with_name(saved.PREFIX.name + "_audit.json").read_text())
    if prior_audit["result_sha256"] != sha(saved.RESULT):
        raise ValueError("saved-context audit mismatch")
    seal.verify(prior_manifest["seal"])
    live.PROTOCOL, live.PREFIX, live.MANIFEST = PROTOCOL, PREFIX, MANIFEST
    live.ARMS, live.method = ("response",), method

    def dependencies(source):
        files = seal.capture([Path(__file__), PROTOCOL, Path(method.__file__), Path(local.__file__),
                              saved.MANIFEST, saved.RESULT,
                              saved.PREFIX.with_name(saved.PREFIX.name + "_audit.json")])["files"]
        return {**original_dependencies(source), **files}

    live.dependencies = dependencies


if __name__ == "__main__":
    configure()
    live.main()
