"""Explore independent random starts after the original DLS first packet."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import e3_cp10_dls_sourcefix_20261031 as sourcefix
from . import e3_cp10_goal_dls_20261031 as dls
from . import p1_four_arm_extra32_budget_20261031 as control

NAME = "e3_cp10_dls_feedback_mix_20261031"
PROTOCOL = Path(__file__).resolve().parents[3] / "docs/D2P_E3_DLS_FEEDBACK_MIX_PROTOCOL_20261031.md"


def mix_bank(original_bank, boundary):
    bank = np.asarray(original_bank)
    if bank.shape != (8, 3, 5, 2) or bank.dtype != np.float32 or not np.isfinite(bank).all():
        raise ValueError("invalid original bank")
    if boundary not in (5, 10, 15):
        raise ValueError("unsupported feedback boundary")
    generator = np.random.default_rng(20261031 + boundary)
    random = np.clip(.4 * generator.standard_normal((4, 3, 5, 2)), -1, 1).astype(np.float32)
    return np.concatenate((bank[[3, 5, 6, 7]], random))


def setup(root):
    experiment = sourcefix.setup(root)
    runner = experiment.runner
    old_build, old_protocol = runner.build, runner.protocol
    old_contract, old_node = control.contract_for, runner.bridge.plan_node
    source_sha = runner.sha(Path(__file__).resolve())

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = old_contract(task, model_sha, hardware_sha)
        return policy, sourcefix.bind_initializer(contract, source_sha), config

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), PROTOCOL):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(structured_initializer_sha256=source_sha,
                     feedback_initializer="four_dls_four_seeded_gaussian",
                     feedback_seed="20261031_plus_boundary",
                     independent_confirmation=False)
        return value

    def plan_node(task, arm, boundary, store, runtime):
        if arm != "structured" or boundary == 0:
            return old_node(task, arm, boundary, store, runtime)
        original_bank = dls.goal_bank

        def bank(jac, projection, goal, nominal):
            candidates, singular = original_bank(jac, projection, goal, nominal)
            return mix_bank(candidates, boundary), singular

        with patch.object(dls, "goal_bank", bank):
            return old_node(task, arm, boundary, store, runtime)

    control.contract_for = contract_for
    runner.bridge.plan_node = plan_node
    runner.NAME = f"{NAME}_root{root}"
    runner.build, runner.protocol = build, protocol
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    args = parser.parse_args()
    experiment = setup(args.root)
    if args.command == "prepare":
        result = experiment.runner.prepare(args.output)
    else:
        print("Run fixed feedback mix on host CUDA; no hindsight actions.", flush=True)
        with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
            result = experiment.runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
