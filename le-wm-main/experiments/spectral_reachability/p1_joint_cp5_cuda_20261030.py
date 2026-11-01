"""Run the common cp5 optimizer on the joint/zero/greedy blind bundle."""
import argparse
from pathlib import Path

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as planner
from . import p1_joint_cp5_bundle_20261030 as bundle

NAME = "p1_joint_cp5_cuda_20261030"
BUNDLE_SHA = "5c87b873ea0db613fb98ad4a9bfd93fde0eec92aeb2223dc7acb53528624dc1e"
PROTOCOL = bundle.ROOT.parent / "docs/D2P_P1_JOINT_CP5_PLANNING_PROTOCOL_20261030_MAIN.md"
original_bindings = planner.bindings


def bindings():
    result = original_bindings()
    for path in (Path(__file__).resolve(), Path(bundle.__file__).resolve(), PROTOCOL):
        result[str(path)] = planner.sha(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    planner.BUNDLE, planner.BUNDLE_SHA, planner.NAME = bundle.OUT, BUNDLE_SHA, NAME
    planner.bindings = bindings
    print("Independently plan ten-step suffix for all 24 real RGB anchors on host CUDA.", flush=True)
    return planner.run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
