"""Execute the unmodified canonical V5 response core in the Reacher RGB runner."""
import json
from pathlib import Path

from experiments import pc_v4_reacher_rgb_live_20261104_codex as prior
from experiments import pc_v5_reacher_saved_context_20261104_codex as adapter
from experiments import pc_v5_reacher_saved_context_v2_20261104_codex as saved
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT, live = prior.ROOT, prior.live
PROTOCOL = ROOT.parent / "docs/PC_V5_REACHER_RGB_LIVE_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_reacher_rgb_live_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")


def configure():
    prior.configure()
    manifest = saved.PREFIX.with_name(saved.PREFIX.name + "_manifest.json")
    result = saved.PREFIX.with_name(saved.PREFIX.name + "_result.json")
    audit = saved.PREFIX.with_name(saved.PREFIX.name + "_audit.json")
    sha = live.host.bridge.sha_file
    source_manifest = json.loads(manifest.read_text())
    if (json.loads(result.read_text())["manifest_sha256"] != sha(manifest)
            or json.loads(audit.read_text())["result_sha256"] != sha(result)):
        raise ValueError("v5 saved-context audit binding mismatch")
    seal.verify(source_manifest["seal"])
    previous_dependencies = live.dependencies
    live.PROTOCOL, live.PREFIX, live.MANIFEST = PROTOCOL, PREFIX, MANIFEST
    live.ARMS, live.method = ("response",), adapter

    def dependencies(source):
        paths = [Path(__file__), PROTOCOL, manifest, result, audit, Path(adapter.__file__),
                 Path(adapter.canonical.__file__), Path(adapter.response.__file__),
                 Path(adapter.core.__file__), Path(adapter.allocation.__file__), Path(adapter.meter.__file__)]
        return {**previous_dependencies(source), **seal.capture(paths)["files"]}

    live.dependencies = dependencies


if __name__ == "__main__":
    configure()
    live.main()
