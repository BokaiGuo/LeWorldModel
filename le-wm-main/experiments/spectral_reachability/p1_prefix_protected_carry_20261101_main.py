"""Test fixed initial prefixes instead of jointly optimizing them away."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from . import p1_response_diverse_carry_20261101_main as diverse
from . import p1_first_feedback_carry_20261101_main as carry

NAME = "p1_prefix_protected_carry_20261101_main"
DOC = Path(__file__).resolve().parents[3] / "docs/D2P_P1_PREFIX_PROTECTED_PROTOCOL_20261101_MAIN.md"


def suffix_only_gradient(gradient):
    if gradient.ndim != 4 or gradient.shape[1:] != (3, 5, 2):
        raise ValueError("H3 action gradient")
    result = gradient.clone()
    result[:, 0] = 0
    return result


def setup(root):
    experiment = diverse.setup(root)
    runner, control = experiment.runner, carry.control
    old_node, old_contract = runner.bridge.plan_node, control.contract_for
    old_build, old_protocol = runner.build, runner.protocol
    source_sha = runner.sha(Path(__file__).resolve())

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = old_contract(task, model_sha, hardware_sha)
        contract["identity"]["optimizer"] = source_sha
        for arm in contract["arms"].values():
            arm["optimizer_sha256"] = source_sha
        control.spec.preflight(contract, control.spec.digest(contract))
        return policy, contract, config

    def node(task, arm, boundary, store, runtime):
        original = control.base.suffix.optimize

        def optimize(leaf, states, history, raw, goal, mean, std, config):
            gradient = leaf.gradient

            def restricted(loss, actions, *, phase):
                return suffix_only_gradient(gradient(loss, actions, phase=phase))

            with patch.object(leaf, "gradient", restricted):
                result = original(leaf, states, history, raw, goal, mean, std, config)
            runner.require(torch.equal(result[:, 0], raw[:, 0]), "prefix changed during suffix optimization")
            return result

        if boundary != 0:
            return old_node(task, arm, boundary, store, runtime)
        with patch.object(control.base.suffix, "optimize", optimize):
            return old_node(task, arm, boundary, store, runtime)

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), DOC):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(initial_prefix_fixed=True, protected_boundaries=[0],
                     optimizer_sha256=source_sha, all_arms_same_prefix_protection=True)
        return value

    control.contract_for = contract_for
    runner.bridge.plan_node, runner.build, runner.protocol = node, build, protocol
    runner.NAME = f"{NAME}_root{root}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    exp = setup(args.root)
    prefix = exp.runner.ROOT / "results" / f"{NAME}_root{args.root}"
    prepared, output = Path(str(prefix) + f"_PREP_{args.tag}.json"), Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt = exp.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with exp.extra.gpu.locking.gpu_lock(exp.extra.gpu.LOCK):
        result = exp.runner.run(str(prepared), receipt["sha256"], str(output),
                               "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
