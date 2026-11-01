"""Common original-tail initialization for all three fixed prefix roles."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import p1_uphill_cp5_cuda20_20261030 as zero

planner, bundle, h10 = zero.planner, zero.bundle, zero.h10
NAME = "p1_uphill_cp5_warm_cuda20_20261030"
PROTOCOL = bundle.ROOT.parent / "docs/D2P_P1_UPHILL_CP5_WARMSTART_PROTOCOL_20261030_MAIN.md"
PREP_SHA = "890e38ecf7758ae739c25e9d355533e4d3b31be91a8c063c9625d1886c77d56d"
BUNDLE_SHA = "376bb8276e1544010ea2767d9c46f141c2665e223154a12b2e6c28786636676a"


def initializers():
    h10.require(planner.sha(bundle.PREP) == PREP_SHA, "H10 prepare hash")
    value = json.loads(bundle.PREP.read_bytes())
    h10.require(len(value["anchors"]) == 24 and len(value["probe_banks"]) == 8, "roster")
    result = []
    for i, anchor in enumerate(value["anchors"]):
        h10.require(anchor["index"] == i and anchor["logical_pair"] == i // 3, "anchor identity")
        matches = [x for x in value["probe_banks"][i // 3] if x["name"] == "original_selected_tail"]
        h10.require(len(matches) == 1, "original tail identity")
        raw = bytes.fromhex(matches[0]["raw10_float32_hex"])
        h10.require(len(raw) == 80 and hashlib.sha256(raw).hexdigest() == matches[0]["raw10_sha256"],
                    "original tail bytes")
        actions = np.frombuffer(raw, dtype="<f4")
        h10.require(np.isfinite(actions).all() and (abs(actions) <= 1).all(), "initializer bounds")
        result.append(raw)
    return result


def bindings():
    result = zero.bindings()
    for path in (Path(__file__).resolve(), PROTOCOL):
        result[str(path)] = planner.sha(path)
    h10.require(result[str(bundle.PREP)] == PREP_SHA, "prepare source hash")
    return result


def configure():
    starts = initializers()
    planner.BUNDLE, planner.BUNDLE_SHA, planner.NAME = bundle.OUT, BUNDLE_SHA, NAME
    planner.CONFIG = {"steps": 20, "learning_rate": 0.05, "history_size": 3}
    planner.bindings = bindings
    cp5 = planner.bridge.cp5

    def make_contract(row, source_hashes):
        policy, contract = zero.prior.make_contract(row, source_hashes)
        identity = planner.spec.digest({"raw10_float32_hex": starts[row["index"]].hex()})
        for arm in contract["arms"].values():
            arm["initializer_sha256"] = identity
        return policy, contract

    def warm_plan(row, cp5_latent, leaf, config):
        states, actions, goal, mean, std = cp5.context(row, cp5_latent, device=cp5_latent.device)
        raw = np.frombuffer(starts[row["index"]], dtype="<f4").copy().reshape(1, 2, 5, 2)
        initial = torch.as_tensor(raw, dtype=torch.float32, device=states.device)
        return cp5.optimizer.optimize(leaf, states, actions, initial, goal, mean, std, config)

    planner.make_contract = make_contract
    cp5.plan = warm_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure()
    print("Plan all 24 RGB-conditioned suffixes from fixed original-tail initializers on host CUDA.", flush=True)
    return planner.run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
