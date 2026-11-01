"""Carry from the FIRST RGB feedback; fixed known-root exploration."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import p1_carry_host_20261101_main as prior

NAME = "p1_first_feedback_carry_20261101_main"
DOC = Path(__file__).resolve().parents[3] / "docs/D2P_P1_FIRST_FEEDBACK_CARRY_PROTOCOL_20261101_MAIN.md"
control, binding, selector = prior.control, prior.binding, prior.selector


def dimensions(boundary):
    # Do not invent padded actions to extend a previously selected plan.
    if boundary not in (0, 5, 10, 15):
        raise ValueError("unsupported boundary")
    return {0: (3, False), 5: (2, True), 10: (1, True), 15: (1, False)}[boundary]


def horizon_protocol():
    return {str(b): list(dimensions(b)) for b in (0, 5, 10, 15)}


def setup(root):
    experiment = prior.mix.setup(root)
    runner, rgb = experiment.runner, experiment.runner.rgb
    old_episode, old_plan, old_cost = rgb.episode, control.plan, control.cost
    old_build, old_protocol = runner.build, runner.protocol

    def episode(env, ref, data, layer, base, row, nodes, visual, runtime, meter, store, plan_node):
        previous = None

        def node(task, arm, boundary, sink, rt):
            nonlocal previous
            horizon, has_carry = dimensions(boundary)
            extra = binding.extra_cost(8, horizon, has_carry=has_carry) if boundary else {}
            feedback = None
            if boundary:
                runner.require(previous is not None, "prior plan required")
                issued = np.asarray(row["executed_raw"], np.float32).reshape(-1, 2)
                runner.require(len(issued) == boundary, "command boundary")
                feedback = previous.bind_feedback(
                    episode=row["episode_id"], boundary=boundary,
                    issued_packet=issued[-5:], rgb_sha256=row["last_observation"]["pixel_sha256"])
                runner.require(len(feedback["tail"]) == (horizon if has_carry else 0),
                               "exact carry/exhaustion horizon")
                raw_history = np.concatenate((ref["raw_history"], issued))[-10:]
                runner.require(task["root_context"]["action_history"] ==
                               rgb.moving.normalized(base, raw_history), "actual normalized history")
            captured = {}

            def cost(which):
                value = old_cost(which)
                for key, count in extra.items():
                    value[key] += count
                return value

            def plan(leaf, which, local_task, config, policy, contract, *, device, emit):
                result = old_plan(leaf, which, local_task, config, policy, contract,
                                  device=device, emit=emit)
                if boundary:
                    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                    fresh = result["optimized_raw"][:, :horizon]
                    rescored = selector.select(
                        leaf, states=tensor(local_task["root_context"]["latent_history"])[None],
                        history=tensor(local_task["root_context"]["action_history"])[None],
                        goal=tensor(local_task["goal"])[None], mean=tensor(local_task["action_mean"]),
                        std=tensor(local_task["action_std"]), new_candidates=fresh,
                        previous_selected=tensor(previous.plan.copy()) if has_carry else None)
                    result.update(rescored)
                    result["feedback_bank"] = (torch.cat((fresh, tensor(feedback["tail"])[None]))
                                               if has_carry else fresh)
                    runner.require(rescored["incremental_cost"] == extra, "selector exact cost")
                    emit("first_feedback_selection", dict(
                        boundary=boundary, horizon=horizon, has_carry=has_carry,
                        feedback={k: v for k, v in feedback.items() if k != "tail"},
                        scores=rescored["scores"], selected_index=rescored["selected_index"],
                        selected_source=rescored["selected_source"], incremental_cost=extra,
                        bank=result["feedback_bank"].detach().cpu().tolist()))
                full = result["selected_raw"].detach().cpu().numpy().astype(np.float32)
                captured["plan"] = full
                emit("selected_full_plan", dict(boundary=boundary, raw=full.tolist()))
                return result

            old_first = control.base.first_packet

            def first(result):
                if "feedback_bank" not in result:
                    return old_first(result)
                value = result["selected_raw"]
                runner.require(torch.equal(value, result["feedback_bank"][result["selected_index"]]),
                               "selected first packet identity")
                return value[0].detach().clone()

            with patch.object(control, "cost", cost), patch.object(control, "plan", plan), \
                    patch.object(control.base, "first_packet", first):
                receipt = plan_node(task, arm, boundary, sink, rt)
            previous = binding.BoundPlan.seal(row["episode_id"], boundary, captured["plan"])
            runner.require(np.array_equal(np.asarray(receipt["commands"], np.float32), previous.plan[0]),
                           "bridge command identity")
            receipt["selected_full_plan"] = previous.plan.tolist()
            receipt["selected_full_plan_sha256"] = previous.plan_sha256
            return receipt
        return old_episode(env, ref, data, layer, base, row, nodes, visual, runtime, meter, store, node)

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), DOC, Path(prior.__file__).resolve(),
                     Path(binding.__file__).resolve(), Path(selector.__file__).resolve()):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(feedback_horizons=horizon_protocol(),
                     rule="all_four_arms_first_feedback_carry_equal_horizon_exhausted_restarts_H1",
                     P1_admission=False, independent_confirmation=False)
        return value

    rgb.episode, runner.build, runner.protocol = episode, build, protocol
    runner.NAME = f"{NAME}_root{root}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple output tag")
    exp = setup(args.root)
    prefix = exp.runner.ROOT / "results" / f"{NAME}_root{args.root}"
    prepared, output = Path(str(prefix) + f"_PREP_{args.tag}.json"), Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive outputs")
    receipt = exp.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with exp.extra.gpu.locking.gpu_lock(exp.extra.gpu.LOCK):
        result = exp.runner.run(str(prepared), receipt["sha256"], str(output),
                               "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
