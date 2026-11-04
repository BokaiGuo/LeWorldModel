"""Run unmodified V5 response using the existing guarded PushT RGB executor."""
from pathlib import Path

from experiments import pc_v4_pusht_rgb_feedback_dev_20261104_codex as live
from experiments import pc_v5_pusht_saved_context_20261104_codex as saved

ROOT = live.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_PUSHT_RGB_FEEDBACK_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_pusht_rgb_feedback_dev_20261104_codex"


class SealAdapter:
    @staticmethod
    def capture(files):
        return saved.SealAdapter.capture([*files, Path(__file__), PROTOCOL])

    verify = staticmethod(saved.SealAdapter.verify)


def configure():
    saved.configure()
    live.PROTOCOL, live.PREFIX = PROTOCOL, PREFIX
    live.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    live.RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
    live.ARMS = ("response",)
    live.seal = SealAdapter


if __name__ == "__main__":
    configure()
    live.main()
