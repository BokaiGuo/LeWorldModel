"""Known-root CUDA feedback experiment retaining previously selected plans."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import e3_cp10_dls_feedback_mix_20261031 as mix
from . import p1_four_arm_extra32_budget_20261031 as control
from . import p1_rgb_carry_replan_codex_20261101 as selector
from . import p1_carry_binding_20261101 as binding

NAME = "p1_carry_host_20261101_main"
PROTOCOL = Path(__file__).resolve().parents[3] / "docs/D2P_P1_CARRY_HOST_PROTOCOL_20261101_MAIN.md"


def node_extra(boundary):
    if boundary not in (0, 5, 10, 15):
        raise ValueError("unsupported decision boundary")
    return (binding.extra_cost(8, (20 - boundary) // 5, has_carry=True)
            if boundary >= 10 else {})


def first_packet(result):
    plan = result["selected_raw"]
    bank = result["carry_bank"]
    index = result["selected_index"]
    if (plan.ndim != 3 or plan.shape[1:] != (5, 2) or plan.shape[0] not in (1, 2)
            or bank.shape != (9, *plan.shape) or not torch.equal(plan, bank[index])
            or not bool(torch.isfinite(bank).all()) or not bool((bank.abs() <= 1).all())):
        raise ValueError("carry selected packet binding")
    return plan[0].detach().clone()


def setup(root):
    experiment = mix.setup(root)
    runner = experiment.runner
    rgb = runner.rgb
    original_episode = rgb.episode
    original_plan = control.plan
    original_cost = control.cost
    old_build, old_protocol = runner.build, runner.protocol

    def episode(env, ref, data, layer, base, row, nodes, visual, runtime, meter, store, plan_node):
        previous = None

        def carried_node(task, arm, boundary, sink, rt):
            nonlocal previous
            issued = np.asarray(row["executed_raw"], np.float32).reshape(-1, 2)
            runner.require(len(issued) == boundary, "actual command boundary")
            feedback = None
            if boundary >= 10:
                runner.require(previous is not None, "missing prior selected plan")
                # The observation and action history are produced by the existing
                # metered RGB episode, not reconstructed from predicted states.
                observation = row["last_observation"]
                feedback = previous.bind_feedback(
                    episode=row["episode_id"], boundary=boundary,
                    issued_packet=issued[-5:], rgb_sha256=observation["pixel_sha256"])
                raw10 = np.concatenate((ref["raw_history"], issued))[-10:]
                runner.require(task["root_context"]["action_history"] ==
                               rgb.moving.normalized(base, raw10),
                               "actual commands do not match planner history")

            def cost(which):
                result = original_cost(which)
                for key, value in node_extra(boundary).items():
                    result[key] += value
                return result

            def plan(leaf, which, local_task, config, policy, contract, *, device, emit):
                result = original_plan(leaf, which, local_task, config, policy, contract,
                                       device=device, emit=emit)
                if boundary >= 10:
                    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                    bank = result["optimized_raw"]
                    rescored = selector.select(
                        leaf, states=tensor(local_task["root_context"]["latent_history"])[None],
                        history=tensor(local_task["root_context"]["action_history"])[None],
                        goal=tensor(local_task["goal"])[None],
                        mean=tensor(local_task["action_mean"]), std=tensor(local_task["action_std"]),
                        new_candidates=bank, previous_selected=tensor(previous.plan.copy()))
                    horizon = rescored["horizon_packets"]
                    result.update(rescored)
                    result["carry_bank"] = torch.cat(
                        (bank[:, :horizon], tensor(feedback["tail"])[None]))
                    emit("carry_selection", {
                        "feedback": {k: v for k, v in feedback.items() if k != "tail"},
                        "task_sha256": runner.digest(local_task),
                        "scores": rescored["scores"], "selected_source": rescored["selected_source"],
                        "selected_index": rescored["selected_index"],
                        "horizon_packets": horizon,
                        "incremental_cost": rescored["incremental_cost"],
                        "candidate_plans": result["carry_bank"].detach().cpu().tolist()})
                full = result["selected_raw"].detach().cpu().numpy().astype(np.float32)
                emit("selected_full_plan", {"raw": full.tolist(), "boundary": boundary})
                captured["plan"] = full
                return result

            captured = {}
            old_first = control.base.first_packet

            def extract(result):
                return first_packet(result) if "carry_bank" in result else old_first(result)

            with patch.object(control, "cost", cost), patch.object(control, "plan", plan), \
                    patch.object(control.base, "first_packet", extract):
                result = plan_node(task, arm, boundary, sink, rt)
            selected = binding.BoundPlan.seal(row["episode_id"], boundary, captured["plan"])
            runner.require(np.array_equal(np.asarray(result["commands"], np.float32), selected.plan[0]),
                           "bridge returned wrong first packet")
            result["selected_full_plan"] = selected.plan.tolist()
            result["selected_full_plan_sha256"] = selected.plan_sha256
            result["carry_feedback"] = None if feedback is None else {
                k: v for k, v in feedback.items() if k != "tail"}
            previous = selected
            return result

        return original_episode(env, ref, data, layer, base, row, nodes, visual,
                                runtime, meter, store, carried_node)

    def protocol():
        value = old_protocol()
        value.update(carry_boundaries=[10, 15],
                     carry_policy="all_arms_same_context_equal_tail_horizon_ties_keep_carry",
                     node_extra_costs={str(b): node_extra(b) for b in (0, 5, 10, 15)},
                     carry_scientific_admission=False)
        return value

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), Path(selector.__file__).resolve(),
                     Path(binding.__file__).resolve(), PROTOCOL):
            value["sources"][str(path)] = runner.sha(path)
        return value

    rgb.episode = episode
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
        print("Known-root carry exploration on host CUDA; not P1 confirmation.", flush=True)
        with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
            result = experiment.runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
