"""V5 signature adapter on saved PushT latent contexts."""
from pathlib import Path

from experiments import pc_v4_pusht_saved_context_adapter_20261104_codex as saved
from experiments import pc_v5_reacher_saved_context_20261104_codex as adapter

ROOT = saved.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_PUSHT_SAVED_CONTEXT_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_pusht_saved_context_20261104_codex"


class SealAdapter:
    @staticmethod
    def capture(files):
        return adapter.SealAdapter.capture([*files, Path(__file__)])

    verify = staticmethod(adapter.SealAdapter.verify)


def configure():
    saved.PROTOCOL, saved.PREFIX = PROTOCOL, PREFIX
    saved.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    saved.RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
    saved.method = adapter
    saved.seal = SealAdapter


if __name__ == "__main__":
    configure()
    saved.main()
