"""Reuse executed-packet diagnostic against the sealed V5 PushT run."""
from pathlib import Path

from experiments import pc_v4_pusht_executed_packet_diag_20261104_codex as diag
from experiments import pc_v5_pusht_rgb_feedback_dev_20261104_codex as run

PROTOCOL = run.ROOT.parent / "docs/PC_V5_PUSHT_EXECUTED_PACKET_DIAG_PROTOCOL_20261104_CODEX.md"
OUT = run.ROOT / "results/pc_v5_pusht_executed_packet_diag_20261104_codex.json"


if __name__ == "__main__":
    run.configure()
    previous = diag.run.seal

    class BoundSeal:
        @staticmethod
        def capture(files):
            return previous.capture([*files, Path(__file__)])

        verify = staticmethod(previous.verify)

    diag.run.seal = BoundSeal
    diag.PROTOCOL, diag.OUT = PROTOCOL, OUT
    diag.AUDIT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
    diag.main()
