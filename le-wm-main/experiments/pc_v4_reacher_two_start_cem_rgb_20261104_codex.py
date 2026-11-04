"""Isolated two-start baseline through the sealed RGB feedback runner."""
import json
from pathlib import Path

from experiments import pc_v4_reacher_rgb_live_20261104_codex as v4
from experiments.spectral_reachability import pc_response_temporal_two_start_cem_v1 as method
from experiments.spectral_reachability import pc_response_temporal_warm_cem_v1 as warm
from experiments.spectral_reachability import pc_response_temporal_feedback_v1 as old_feedback
from experiments.spectral_reachability import pc_response_temporal_feedback_v3 as old_v3
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

live, ROOT = v4.live, v4.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_REACHER_TWO_START_CEM_RGB_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_reacher_two_start_cem_rgb_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")


def configure():
    v4.configure()
    prior_dependencies = live.dependencies
    source_manifest = json.loads(v4.MANIFEST.read_text())
    if source_manifest["dependencies"] != prior_dependencies(json.loads(live.run.SOURCE.read_text())):
        raise ValueError("v4 dependency mismatch")
    sha = live.host.bridge.sha_file
    audit = v4.PREFIX.with_name(v4.PREFIX.name + "_audit.json")
    if json.loads(audit.read_text())["manifest_sha256"] != sha(v4.MANIFEST):
        raise ValueError("v4 audit binding mismatch")
    live.PROTOCOL, live.PREFIX, live.MANIFEST = PROTOCOL, PREFIX, MANIFEST
    live.ARMS, live.method = ("smooth_cem",), method

    def dependencies(source):
        files = seal.capture([Path(__file__), PROTOCOL, Path(method.__file__), Path(warm.__file__),
                              Path(old_feedback.__file__), Path(old_v3.__file__), Path(core.__file__),
                              v4.MANIFEST, audit])["files"]
        return {**prior_dependencies(source), **files}

    live.dependencies = dependencies


if __name__ == "__main__":
    configure()
    live.main()
