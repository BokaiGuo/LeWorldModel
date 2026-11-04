"""Run the preserved Cube decision harness with v7 paired abstention."""

from pathlib import Path

from experiments import pc_v6_cube_decision_dev_20261104_codex as harness
from experiments.spectral_reachability import pc_target_reach_paired_abstention_v7 as v7


def main():
    harness.v6 = v7
    harness.PROTOCOL = harness.ROOT.parent / "docs/PC_V7_PAIRED_ABSTENTION_DEV_PROTOCOL_20261104_CODEX.md"
    harness.OUTPUT = harness.ROOT / "results/pc_v7_cube_decision_dev_20261104_codex.json"
    harness.TRACE = harness.OUTPUT.with_suffix(".npz")
    capture = harness.seal.capture

    def capture_with_v7(paths=()):
        return capture((*paths, Path(__file__), Path(v7.v6.__file__)))

    harness.seal.capture = capture_with_v7
    harness.main()


if __name__ == "__main__":
    main()
