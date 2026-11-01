"""Twenty-step matched per-anchor suffix optimization on host CUDA."""
import argparse
from pathlib import Path

from . import p1_joint_cp5_cuda_20261030 as baseline
from . import p1_joint_cp5_bundle_20261030 as bundle

planner = baseline.planner
NAME = "p1_joint_cp5_cuda20_20261030"
PROTOCOL = bundle.ROOT.parent / "docs/D2P_P1_JOINT_CP5_20STEP_PROTOCOL_20261030_MAIN.md"
original_contract = planner.make_contract


def bindings():
    result = baseline.bindings()
    for path in (Path(__file__).resolve(), PROTOCOL):
        result[str(path)] = planner.sha(path)
    return result


def make_contract(row, source_hashes):
    policy, contract = original_contract(row, source_hashes)
    caps = contract["levels"][0]["caps"]
    caps.update(action_encode_calls=42, predict_calls=42, predict_samples=42,
                backward_calls=20, vjp_rows=20)
    contract["levels"][1]["caps"] = {key: value + 1 for key, value in caps.items()}
    return policy, contract


def configure():
    planner.BUNDLE, planner.BUNDLE_SHA = bundle.OUT, baseline.BUNDLE_SHA
    planner.NAME = NAME
    planner.CONFIG = {"steps": 20, "learning_rate": 0.05, "history_size": 3}
    planner.bindings = bindings
    planner.make_contract = make_contract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    configure()
    print("Plan 24 independent suffixes at fixed 20-step budget on host CUDA.", flush=True)
    return planner.run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
