"""Isolated output prefix for retry after a sandbox device-access failure."""
from pathlib import Path

from experiments import pc_v5_reacher_saved_context_20261104_codex as prior
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PREFIX = prior.ROOT / "results/pc_v5_reacher_saved_context_v2_20261104_codex"
PROTOCOL = prior.ROOT.parent / "docs/PC_V5_REACHER_SAVED_CONTEXT_V2_PROTOCOL_20261104_CODEX.md"


class RetrySeal:
    @staticmethod
    def capture(files):
        return prior.SealAdapter.capture([*files, Path(__file__), PROTOCOL])

    verify = staticmethod(seal.verify)


def configure():
    prior.configure()
    saved = prior.saved
    saved.PREFIX, saved.PROTOCOL = PREFIX, PROTOCOL
    saved.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    saved.RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
    saved.seal = RetrySeal


if __name__ == "__main__":
    configure()
    prior.saved.main()
