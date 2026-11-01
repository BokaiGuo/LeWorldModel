"""Plan all uphill/zero/greedy real-cp5 suffixes on host CUDA."""
import argparse
from pathlib import Path

from . import p1_joint_cp5_cuda20_20261030 as prior
from . import p1_uphill_cp5_bundle_20261030 as bundle
from . import p1_uphill_h10_20261030 as h10

planner = prior.planner
NAME = "p1_uphill_cp5_cuda20_20261030"
PROTOCOL = bundle.ROOT.parent / "docs/D2P_P1_UPHILL_CP5_PROTOCOL_20261030_MAIN.md"


def bindings():
    result = prior.baseline.original_bindings()
    for path in (Path(__file__).resolve(), Path(bundle.__file__).resolve(),
                 Path(h10.__file__).resolve(), PROTOCOL, h10.PROTOCOL, bundle.PREP):
        result[str(path)] = planner.sha(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    args = parser.parse_args()
    planner.BUNDLE, planner.BUNDLE_SHA = bundle.OUT, args.bundle_sha256
    planner.NAME = NAME
    planner.CONFIG = {"steps": 20, "learning_rate": 0.05, "history_size": 3}
    planner.bindings = bindings
    planner.make_contract = prior.make_contract
    print("Plan 24 fixed real-RGB suffixes with equal optimizer caps on host CUDA.", flush=True)
    return planner.run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
