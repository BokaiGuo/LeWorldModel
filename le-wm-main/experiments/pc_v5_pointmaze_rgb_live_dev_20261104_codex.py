"""Bind the unmodified V5 compiler to the established PointMaze RGB runner."""
from pathlib import Path

from experiments import pc_response_temporal_v3_rgb_live_20261104_codex as prior
from experiments import pc_v5_reacher_saved_context_20261104_codex as adapter

live = prior.live
ROOT = live.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_POINTMAZE_RGB_LIVE_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_pointmaze_rgb_live_dev_20261104_codex"


def configure():
    prior.configure()
    previous = live.dependencies
    live.PROTOCOL, live.PREFIX = PROTOCOL, PREFIX
    live.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    live.ARMS, live.feedback = ("response",), adapter

    def dependencies():
        files = [Path(__file__), PROTOCOL, Path(adapter.__file__), Path(adapter.canonical.__file__),
                 Path(adapter.response.__file__), Path(adapter.core.__file__),
                 Path(adapter.allocation.__file__), Path(adapter.meter.__file__)]
        return {**previous(), **{str(p.resolve()): live.old.prior.sha(p) for p in files}}

    live.dependencies = dependencies


if __name__ == "__main__":
    configure()
    live.main()
