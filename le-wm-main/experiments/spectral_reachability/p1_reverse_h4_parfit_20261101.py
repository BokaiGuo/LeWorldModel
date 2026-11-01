"""Exclusive reverse-task H4 launcher; dry-prepare is CPU and read-only.

Run in a fresh process: setup uses process-local monkeypatches inherited from
the H3 chain.  Never import the privileged witness diagnostic or its actions.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import p1_h4_parfit_20261101 as h4
from . import p1_node_h4_parfit_20261101 as node_impl
from . import p1_carry_binding_20261101 as binding
from . import p1_reverse_control_20261101_main as reverse
from . import p1_four_arm_extra32_root0_20261031 as extra

NAME = "p1_reverse_h4_parfit_20261101"
_SET_UP = False


def boundary_config(boundary):
    horizon, _ = h4.dimensions(boundary)
    return {"planning_horizon_packets": horizon, "allocated_horizon_packets": 4,
            "decision_boundary": boundary}


def setup(arm):
    global _SET_UP
    if _SET_UP or arm not in h4.ARMS:
        raise ValueError("one valid arm per fresh process")
    _SET_UP = True
    runner = extra.runner
    # Capture the physical RGB driver before H3 installs its carry wrapper.
    physical_episode = runner.rgb.episode
    experiment = reverse.setup(arm)
    control = reverse.Q.carry.control
    kernels = reverse.Q.kernels
    old_contract, old_build, old_protocol = control.contract_for, runner.build, runner.protocol
    wrapper_sha = runner.sha(Path(__file__).resolve())
    node_sha = runner.sha(Path(node_impl.__file__).resolve())
    primitive_sha = runner.sha(Path(h4.__file__).resolve())
    initializer_sha = control.spec.digest({"response_source_sha256": primitive_sha,
                                            "feedback_mix_source_sha256": node_sha})
    runner.require(node_impl.SEEDS == control.base.SEEDS, "H3 candidate seeds changed")

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = old_contract(task, model_sha, hardware_sha)
        boundary = config.get("decision_boundary")
        runner.require(config.items() >= boundary_config(boundary).items(), "H4 boundary config")
        policy["horizon"] = config["planning_horizon_packets"]
        contract["phase_policy_sha256"] = control.spec.digest(policy)
        contract["identity"]["optimizer_config"] = control.spec.digest(config)
        contract["identity"]["optimizer"] = node_sha
        for entry in contract["arms"].values():
            entry["optimizer_sha256"] = node_sha
            entry["optimizer_config_sha256"] = contract["identity"]["optimizer_config"]
        contract["arms"]["structured"]["initializer_sha256"] = initializer_sha
        control.spec.preflight(contract, control.spec.digest(contract))
        return policy, contract, config

    def episode(env, ref, data, layer, base, row, nodes, visual, runtime, meter, store, plan_node):
        previous = None

        def physical_node(task, which, boundary, sink, rt):
            nonlocal previous
            horizon, carry = h4.dimensions(boundary)
            issued = np.asarray(row["executed_raw"], dtype=np.float32).reshape(-1, 2)
            runner.require(len(issued) == boundary, "returned command boundary")
            feedback = None
            if carry:
                runner.require(previous is not None, "missing prior plan")
                feedback = previous.bind_feedback(
                    episode=row["episode_id"], boundary=boundary,
                    issued_packet=issued[-5:], rgb_sha256=row["last_observation"]["pixel_sha256"])
                runner.require(len(feedback["tail"]) == horizon, "carry horizon")
                raw10 = np.concatenate((ref["raw_history"], issued))[-10:]
                runner.require(task["root_context"]["action_history"] ==
                               runner.rgb.moving.normalized(base, raw10), "returned action history")
            captured = {}

            def cost(which_arm):
                return h4.node_cost(which_arm, boundary)

            def plan(leaf, which_arm, local_task, config, policy, contract, *, device, emit):
                runner.require(policy["metric"] == "goal_encoder_cometric_v1"
                               and policy["horizon"] == horizon
                               and config.items() >= boundary_config(boundary).items()
                               and local_task["goal"] == kernels.GOAL
                               and config["steps"] == 6
                               and config["learning_rate"] == .05
                               and control.spec.digest(config) == leaf.config_sha
                               and control.spec.digest(policy) == contract["phase_policy_sha256"]
                               and control.spec.digest(contract) == leaf.contract_sha,
                               "frozen Q/config binding")
                projection = np.asarray(local_task["projection"], dtype=np.float64)
                runner.require(projection.shape == (192, 192) and
                               np.array_equal(projection, np.eye(192)), "identity projection required")
                emit("h4_contract_binding", {
                    "boundary": boundary, "config": config, "policy": policy,
                    "config_sha256": control.spec.digest(config),
                    "policy_sha256": control.spec.digest(policy),
                    "contract_sha256": leaf.contract_sha,
                    "projection": "identity_192_exact",
                    "candidate_seeds": dict(node_impl.SEEDS)})
                tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
                result = node_impl.run_node(
                    leaf, arm=which_arm, boundary=boundary,
                    states=tensor(local_task["root_context"]["latent_history"])[None],
                    history=tensor(local_task["root_context"]["action_history"])[None],
                    goal=tensor(local_task["goal"])[None], mean=tensor(local_task["action_mean"]),
                    std=tensor(local_task["action_std"]),
                    matrix=torch.as_tensor(kernels.MATRIX, dtype=torch.float64, device=device),
                    previous=tensor(previous.plan.copy()) if carry else None,
                    issued=tensor(issued[-5:]) if carry else None, emit=emit)
                captured["plan"] = result["selected_raw"].detach().cpu().numpy().astype(np.float32)
                emit("h4_feedback_binding", {"boundary": boundary, "horizon": horizon,
                     "feedback": None if feedback is None else
                     {key: value for key, value in feedback.items() if key != "tail"},
                     "current_rgb_sha256": row["last_observation"]["pixel_sha256"],
                     "task_sha256": runner.digest(local_task)})
                return result

            def first(result):
                selected = result["selected_raw"]
                runner.require(isinstance(selected, torch.Tensor)
                               and selected.shape == (horizon, 5, 2), "H4 selected plan shape")
                return selected[0].detach().clone()

            with patch.dict(control.base.CONFIG, boundary_config(boundary)), \
                    patch.object(control, "cost", cost), patch.object(control, "plan", plan), \
                    patch.object(control.base, "first_packet", first):
                receipt = plan_node(task, which, boundary, sink, rt)
            runner.require("plan" in captured, "missing full selected plan")
            sealed = binding.BoundPlan.seal(row["episode_id"], boundary, captured["plan"])
            runner.require(np.array_equal(np.asarray(receipt["commands"], np.float32), sealed.plan[0]),
                           "bridge selected packet mismatch")
            receipt["selected_full_plan"] = sealed.plan.tolist()
            receipt["selected_full_plan_sha256"] = sealed.plan_sha256
            receipt["carry_feedback"] = (None if feedback is None else
                                          {key: value for key, value in feedback.items() if key != "tail"})
            previous = sealed
            return receipt

        return physical_episode(env, ref, data, layer, base, row, nodes, visual,
                                runtime, meter, store, physical_node)

    def protocol():
        value = old_protocol()
        costs = {str(b): {a: h4.node_cost(a, b) for a in h4.ARMS} for b in (0, 5, 10, 15)}
        for stale in ("feedback_initializer", "feedback_seed", "dls_scales",
                      "neural_calls_unchanged_from_exchange", "forward_count_reduced",
                      "exploratory_structured_extra32", "initializer"):
            value.pop(stale, None)
        caps = {str(b): {k: max(costs[str(b)][a][k] for a in h4.ARMS)
                         for k in h4.AXES} for b in (0, 5, 10, 15)}
        value.update(schema=NAME + ".protocol.v1", arms=[arm],
                     optimization_horizons={str(b): list(h4.dimensions(b)) for b in (0, 5, 10, 15)},
                     feedback_horizons={str(b): list(h4.dimensions(b)) for b in (0, 5, 10, 15)},
                     node_costs_by_boundary=costs, node_costs=costs["0"],
                     node_caps_by_boundary=caps, node_caps=caps["0"],
                     exchange_boundaries=[0, 5, 10], initial_prefix_fixed=True,
                     rule="H4_to_H1_fresh_vs_actual_carried_tail",
                     extra_predict_calls_full_trajectory=64 * (4 + 3 + 2),
                     witness_actions_used_in_planner=False, P1_admission=False,
                     independent_confirmation=False, fairness_proven=False,
                     structured_initializer_sha256=initializer_sha, optimizer_sha256=node_sha,
                     candidate_seeds=dict(node_impl.SEEDS),
                     gaussian_rng="torch_CPU_h4_shape_versus_h3_sequence_differs",
                     feedback_initializer="four_response_four_seeded_gaussian",
                     feedback_seed="20261031_plus_boundary",
                     initializer_labels={"structured": "Q_DLS_plus_minus_top3_H4",
                                         "native": "zero_centered_gaussian_native_style",
                                         "random": "random_raw_gaussian_bank",
                                         "immediate_progress": "one_packet_progress_top8_of16"},
                     h4_runtime_wrapper_sha256=wrapper_sha)
        return value

    def build():
        value = old_build()
        value["schema"] = NAME + ".prepare.v1"
        value["protocol"] = protocol()
        for path in (Path(__file__), Path(node_impl.__file__), Path(h4.__file__)):
            path = path.resolve()
            value["sources"][str(path)] = runner.sha(path)
        return value

    control.contract_for = contract_for
    runner.rgb.episode = episode
    runner.bridge.plan_node = extra.plan_node
    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_{arm}"
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=h4.ARMS, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    execute = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        execute.add_argument("--" + key, required=True)
    execute.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    experiment = setup(args.arm)
    runner = experiment.runner
    if args.command == "dry-prepare":
        value = runner.build()
        control = reverse.Q.carry.control
        task = value["source"]["tasks"][0]
        contracts = {}
        for boundary in (0, 5, 10, 15):
            with patch.dict(control.base.CONFIG, boundary_config(boundary)), \
                    patch.object(control, "cost", lambda a: h4.node_cost(a, boundary)):
                policy, contract, config = control.contract_for(task, "a" * 64, "b" * 64)
            expected = value["protocol"]["node_caps_by_boundary"][str(boundary)]
            if (contract["levels"][0]["caps"] != expected
                    or policy["horizon"] != h4.dimensions(boundary)[0]
                    or not config.items() >= boundary_config(boundary).items()
                    or control.spec.digest(config) != contract["identity"]["optimizer_config"]
                    or control.spec.digest(policy) != contract["phase_policy_sha256"]):
                raise ValueError("dry node contract/caps mismatch")
            contracts[str(boundary)] = control.spec.digest(contract)
        result = {"status": "dry_prepared_not_written", "schema": value["schema"],
                  "protocol": value["protocol"], "source_count": len(value["sources"]),
                  "build_sha256": runner.digest(value), "dry_contract_sha256": contracts,
                  "cuda_used": False}
    elif args.command == "prepare":
        result = runner.prepare(args.output)
    else:
        with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
            result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
