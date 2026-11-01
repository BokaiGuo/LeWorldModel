"""Unchanged goal-only co-metric on the fixed centered task."""
import argparse
from pathlib import Path

from . import p1_goal_cometric_cuda_20261101_main as measurement
from . import p1_centered_target_cuda_20261101_main as target

NAME = "p1_centered_metric_cuda_20261101_main"
TARGET = measurement.source.ROOT / "results/p1_centered_target_cuda_20261101_main_MAIN01.json"
TARGET_SHA = "801c3b089cb9af71901c638b9197b25aa84573e0befefec8abe8a3c068d62ef3"


def run(output):
    # Recheck source frames and physical eligibility, but fit only to goal RGB.
    _, _, sources = target.collect()
    measurement.NAME = NAME
    measurement.TARGET, measurement.TARGET_SHA = TARGET, TARGET_SHA
    measurement.DOC = target.witness.DOC
    old_bindings = measurement.source.base.bindings

    def bindings():
        value = old_bindings()
        value.update(sources)
        path = Path(__file__).resolve()
        value[str(path)] = measurement.source.base.sha(path)
        return value

    measurement.source.base.bindings = bindings
    return measurement.run(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", required=True, action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
