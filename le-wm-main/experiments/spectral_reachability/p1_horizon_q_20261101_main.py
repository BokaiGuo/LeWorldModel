"""Align fresh optimization/Jacobian/replay with the exact carry horizon."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from . import p1_equal_candidate_q_20261101_main as equal

parent = equal.parent
control = parent.carry.control
base, spec = control.base, control.spec
NAME = "p1_horizon_q_20261101_main"
DOC = parent.ROOT.parent / "docs/D2P_P1_HORIZON_Q_PROTOCOL_20261101_MAIN.md"


def horizon():
    value = base.CONFIG.get("planning_horizon_packets", 3)
    if type(value) is not int or value not in (1, 2, 3):
        raise ValueError("planning horizon")
    return value


def cost(arm):
    value = base.cost(arm)
    # Eight candidates, each with steps forwards + one final replay.
    # Structured additionally measures one H-packet Jacobian.
    removed = (3 - horizon()) * (8 * (base.CONFIG["steps"] + 1) + int(arm == "structured"))
    for key in ("action_encode_calls", "predict_calls", "predict_samples"):
        value[key] -= removed
    return value


def rollout(leaf, states, history, raw, mean, std, packets):
    z, past = states.detach(), history.detach()
    for step in range(packets):
        normalized = ((raw[:, step] - mean) / std).reshape(1, 1, 10)
        context = torch.cat((past, normalized), dim=1)
        prediction = leaf.predict(z, leaf.action_encode(context, phase="common_suffix"),
                                  phase="common_suffix")
        spec.require(prediction.shape == z.shape and bool(torch.isfinite(prediction).all()),
                     "native prediction")
        z, past = torch.cat((z, prediction[:, -1:]), dim=1)[:, -3:], context[:, -2:]
    return z[:, -1]


def optimize(leaf, states, history, initial, goal, mean, std, config):
    spec.require(type(leaf) is base.leafmod.LeafAdapter and spec.digest(config) == leaf.config_sha,
                 "optimizer contract")
    packets = horizon()
    spec.require(config["planning_horizon_packets"] == packets
                 and initial.shape == (1, 3, 5, 2) and states.shape == (1, 3, 192)
                 and history.shape == (1, 2, 10) and goal.shape == (1, 192)
                 and bool(torch.isfinite(initial).all()) and bool((initial.abs() <= 1).all())
                 and bool((std > 0).all()), "horizon optimizer inputs")
    value = initial.detach().clone()
    for _ in range(config["steps"]):
        raw = value.detach().requires_grad_(True)
        prediction = rollout(leaf, states, history, raw, mean, std, packets)
        gradient = leaf.gradient(parent.kernels.loss(prediction, goal.detach()), raw,
                                 phase="common_suffix")
        spec.require(gradient.shape == raw.shape and bool(torch.isfinite(gradient).all()),
                     "finite horizon gradient")
        spec.require(torch.count_nonzero(gradient[:, packets:]).item() == 0,
                     "discarded tail must have zero gradient")
        value = (raw - config["learning_rate"] * gradient).clamp(-1, 1).detach()
    return value


def setup(arm):
    # Install before first-feedback setup captures its underlying cost callable.
    control.cost = cost
    experiment = equal.setup(arm)
    runner = experiment.runner
    original_node, original_contract = runner.bridge.plan_node, control.contract_for
    original_build, original_protocol = runner.build, runner.protocol
    original_normal = base.raw._normal
    optimizer_sha = runner.sha(Path(__file__).resolve())

    def node(task, which, boundary, store, runtime):
        packets, _ = parent.carry.dimensions(boundary)

        def normal(leaf, phase, states, history, raw, mean, std):
            return original_normal(leaf, phase, states, history, raw[:, :packets], mean, std)

        with patch.dict(base.CONFIG, planning_horizon_packets=packets), \
                patch.object(base.raw, "_normal", normal):
            return original_node(task, which, boundary, store, runtime)

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = original_contract(task, model_sha, hardware_sha)
        contract["identity"]["optimizer"] = optimizer_sha
        for entry in contract["arms"].values():
            entry["optimizer_sha256"] = optimizer_sha
        spec.preflight(contract, spec.digest(contract))
        return policy, contract, config

    def build():
        value = original_build()
        for path in (Path(__file__).resolve(), DOC):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = original_protocol()
        value.update(optimization_horizons=parent.carry.horizon_protocol(),
                     align_jacobian_optimization_replay_carry=True,
                     backward_count_unchanged=True, forward_count_reduced=True)
        return value

    base.suffix.optimize = optimize
    runner.bridge.plan_node, control.contract_for = node, contract_for
    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_{arm}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=parent.parent.ARMS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    experiment = setup(args.arm)
    prefix = parent.ROOT / "results" / f"{NAME}_{args.arm}"
    prepared = Path(str(prefix) + f"_PREP_{args.tag}.json")
    output = Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt = experiment.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        result = experiment.runner.run(
            str(prepared), receipt["sha256"], str(output),
            "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
