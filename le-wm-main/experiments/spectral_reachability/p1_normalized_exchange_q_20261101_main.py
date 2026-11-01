"""Fixed action-space gradient steps plus complete prefix/tail exchange."""
import argparse
import json
from pathlib import Path

import torch

from . import p1_suffix_exchange_q_20261101_main as exchange

h = exchange.parent
NAME = "p1_normalized_exchange_q_20261101_main"
DOC = exchange.ROOT.parent / "docs/D2P_P1_NORMALIZED_EXCHANGE_PROTOCOL_20261101_MAIN.md"


def normalized_update(raw, gradient, peak):
    if (raw.shape != gradient.shape or not bool(torch.isfinite(gradient).all())
            or not bool(torch.isfinite(raw).all()) or not 0 < peak <= 1):
        raise ValueError("finite bounded-step inputs")
    norm = gradient.abs().amax()
    direction = gradient / norm.clamp_min(torch.finfo(gradient.dtype).tiny)
    return (raw - peak * direction).clamp(-1, 1).detach()


def optimize(leaf, states, history, initial, goal, mean, std, config):
    h.spec.require(type(leaf) is h.base.leafmod.LeafAdapter and h.spec.digest(config) == leaf.config_sha,
                   "optimizer contract")
    packets = h.horizon()
    h.spec.require(config["gradient_rule"] == "linf_normalized_action_peak"
                   and config["planning_horizon_packets"] == packets
                   and initial.shape == (1, 3, 5, 2) and states.shape == (1, 3, 192)
                   and history.shape == (1, 2, 10) and goal.shape == (1, 192)
                   and bool(torch.isfinite(initial).all()) and bool((initial.abs() <= 1).all())
                   and bool((std > 0).all()), "normalized optimizer inputs")
    value = initial.detach().clone()
    updates = []
    for iteration in range(config["steps"]):
        raw = value.detach().requires_grad_(True)
        prediction = h.rollout(leaf, states, history, raw, mean, std, packets)
        loss = h.parent.kernels.loss(prediction, goal.detach())
        gradient = leaf.gradient(loss, raw, phase="common_suffix")
        h.spec.require(gradient.shape == raw.shape and bool(torch.isfinite(gradient).all())
                       and torch.count_nonzero(gradient[:, packets:]).item() == 0, "bounded horizon gradient")
        value = normalized_update(raw, gradient, config["learning_rate"])
        updates.append(dict(iteration=iteration, loss=float(loss.detach()),
                            gradient_linf=float(gradient.abs().amax()),
                            action_step_linf=float((value - raw.detach()).abs().amax())))
    leaf.normalized_update_records.append(updates)
    return value


def setup(arm):
    original = exchange.BASE_PLAN

    def instrumented(leaf, which, task, config, policy, contract, *, device, emit):
        leaf.normalized_update_records = []
        result = original(leaf, which, task, config, policy, contract, device=device, emit=emit)
        h.spec.require(len(leaf.normalized_update_records) == 8, "eight optimized candidates")
        emit("normalized_gradient_updates", dict(candidates=leaf.normalized_update_records,
                                                 physical_labels_used=False))
        return result

    exchange.BASE_PLAN = instrumented
    experiment = exchange.setup(arm)
    h.base.CONFIG = dict(h.base.CONFIG, gradient_rule="linf_normalized_action_peak")
    h.base.suffix.optimize = optimize
    runner = experiment.runner
    old_build, old_protocol, old_contract = runner.build, runner.protocol, h.control.contract_for
    source_sha = runner.sha(Path(__file__).resolve())

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = old_contract(task, model_sha, hardware_sha)
        contract["identity"]["optimizer"] = source_sha
        for entry in contract["arms"].values():
            entry["optimizer_sha256"] = source_sha
        h.spec.preflight(contract, h.spec.digest(contract))
        return policy, contract, config

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), DOC):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(gradient_rule="linf_normalized_action_peak", action_peak_step=.05,
                     neural_calls_unchanged_from_exchange=True,
                     no_backtracking_or_descent_guarantee=True)
        return value

    h.control.contract_for = contract_for
    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_{arm}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=h.parent.parent.ARMS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    experiment = setup(args.arm)
    prefix = exchange.ROOT / "results" / f"{NAME}_{args.arm}"
    prepared, output = Path(str(prefix) + f"_PREP_{args.tag}.json"), Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt = experiment.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        result = experiment.runner.run(str(prepared), receipt["sha256"], str(output),
                                      "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
