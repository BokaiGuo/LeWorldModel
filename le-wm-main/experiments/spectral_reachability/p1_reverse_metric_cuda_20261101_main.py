"""Goal-only co-metric for the audited opposite-side task."""
import argparse
from pathlib import Path

from . import p1_goal_cometric_cuda_20261101_main as measurement

NAME = "p1_reverse_metric_cuda_20261101_main"
TARGET = measurement.source.ROOT / "results/p1_reverse_target_cuda_20261101_main_MAIN01.json"
TARGET_SHA = "a669269652bb677d6921ab2d76fe0fb8847de3461509092522d4b803bd54318f"


def run(output):
    measurement.NAME = NAME
    measurement.TARGET, measurement.TARGET_SHA = TARGET, TARGET_SHA
    measurement.DOC = measurement.source.ROOT.parent / "docs/D2P_P1_REVERSE_TARGET_PROTOCOL_20261101_MAIN.md"
    old_bindings = measurement.source.base.bindings

    def bindings():
        value = old_bindings()
        path = Path(__file__).resolve()
        value[str(path)] = measurement.source.base.sha(path)
        return value

    measurement.source.base.bindings = bindings
    return measurement.run(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
