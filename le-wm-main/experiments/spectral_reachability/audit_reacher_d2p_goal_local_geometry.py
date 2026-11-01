"""All24 known-root CUDA residual-anchor isolation and physical replay.

Preserve all three task conditions, all endpoint winners and both controls.
Reproduce old common-residual metrics before changing only the residual
anchor. Seal ALL model decisions before measuring any new branch successor.
No threshold fitting, new training, physical-state planning or P1 override.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import audit_reacher_d2p_certified_task_matrix as matrix
from experiments.spectral_reachability import audit_reacher_d2p_actuator_limit_tasks as tasks
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_geometry_dispatch as geometry
from experiments.spectral_reachability import reacher_d2p_goal_local_geometry as local
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_packet_interpolation import load_historical_p0


def validate_inputs(paths):
    reports, bindings = {}, []
    for path in paths:
        report = json.loads(path.read_text())
        task = report["sealed_protocol"]["task"]
        if task in reports:
            raise ValueError("duplicate task reference")
        reports[task] = report
        bindings.append({"task": task, "path": str(path.resolve()), "sha256": digest(path)})
    matrix.audit(reports)  # Includes root-identity join, all three tasks and old decisions.
    return reports, bindings


def collect_responses(model, pixels, goal, history, bank, required, *, p0, prefix, lower, upper):
    responses, endpoints = {}, {}
    target = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach().to(torch.float64)
    for index in required:
        emb, actions = prefix._state_after_prefix(model, pixels, history, bank[index:index + 1], action_lower=lower, action_upper=upper)

        def terminal(flat):
            value = prefix._terminal_from_state(model, emb, actions, flat.reshape(1, 2, history.shape[-1]), action_lower=lower, action_upper=upper)[0]
            return p0._planning_coordinate(model, value, role="predicted")

        origin = torch.zeros(2 * history.shape[-1], device=history.device, dtype=history.dtype, requires_grad=True)
        endpoints[index] = terminal(origin).detach().to(torch.float64)
        responses[index] = torch.autograd.functional.jacobian(terminal, origin, vectorize=True).detach().to(torch.float64)
    return responses, endpoints, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports, bindings = validate_inputs(args.reference)
    reference = reports["full_drive"]["sealed_protocol"]
    reference_path = next(Path(item["path"]) for item in bindings if item["task"] == "full_drive")
    p0, p0_path = load_historical_p0(reference_path, reference)
    prefix_path = reference_path.with_suffix(".implementation") / "prefix_suffix_backend.py"
    if digest(prefix_path) != reference["implementation_sha256"]["prefix_suffix_backend"]:
        raise ValueError("historical prefix/suffix snapshot changed")
    spec = importlib.util.spec_from_file_location("historical_prefix_goal_local_audit", prefix_path)
    prefix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefix)
    prefix.p0 = p0
    if digest(args.policy / "weights.pt") != reference["weights_sha256"] or digest(args.adapter_artifact) != reference["adapter_sha256"]:
        raise ValueError("frozen checkpoint/action statistics changed")
    modules = {"auditor": Path(__file__), "local_geometry": Path(local.__file__), "qp": Path(qp.__file__),
        "historical_p0": p0_path, "historical_prefix_suffix": prefix_path, "context": Path(execution.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename), "shared": Path(shared.__file__),
        "geometry": Path(geometry.__file__), "base_compiler": Path(geometry.base.__file__),
        "solver": Path(solver.__file__), "rescue": Path(rescue.__file__), "solver_p0": Path(solver.p0.__file__),
        "solver_prefix": Path(solver.p1.__file__), "task_contract": Path(tasks.__file__), "reference_validator": Path(matrix.__file__)}
    for key, old_key in (("context", "context"), ("source_context", "source_context"), ("geometry", "response_backend"),
                         ("qp", "certified_qp"), ("solver", "solver"), ("rescue", "rescue"), ("base_compiler", "base_refinement_compiler")):
        if digest(modules[key]) != reference["implementation_sha256"][old_key]:
            raise ValueError("reference computation/execution drift: " + key)
    output = args.output.resolve()
    sidecars = [output, output.with_suffix(".protocol.json"), output.with_suffix(".selection.json"),
                output.with_suffix(".roots"), output.with_suffix(".implementation")]
    if any(path.exists() for path in sidecars):
        raise FileExistsError("new immutable output and sidecars required")
    protocol = {"inputs": bindings, "root_seeds": reference["root_seeds"], "tasks": tasks.TASKS,
        "weights_sha256": reference["weights_sha256"], "adapter_sha256": reference["adapter_sha256"], "source_sha256": reference["source_sha256"],
        "target_delta": reference["target_delta"], "damping": .01, "bootstrap_seed": 202641911,
        "policy": local.POLICY, "rule": "original endpoint winners AND own-response feasible gain beats both swapped-control gain upper bounds, holding candidate residual fixed",
        "selection_law": "min original fine8 endpoint cost among qualified; else original greedy; no new tuning",
        "physical_budget": "one five-step prefix plus ten-step suffix for every branch",
        "controls": "replay immutable common-gate, same-bank greedy, zero and CEM witnesses; new prefixes receive observed-RGB identical fine8 suffix",
        "all_choices_sealed_before_any_new_successor": True,
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "All-known-root residual-anchor causal diagnostic. Unequal model compute, not fresh efficacy, P1 admission or full research closure."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with sidecars[1].open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    sidecars[3].mkdir()
    sidecars[4].mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, sidecars[4] / (key + ".py"))
    print("All24 fixed task/root units, 113 original responses. Exact old-metric replay, candidate-local response swaps, then physical controls; host CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != reference["source_sha256"]:
        raise ValueError("official source receipt changed")
    print("Verifying full official-source SHA and loading original proposal support.", flush=True)
    pool, support = shared.load_packet_pool(dataset, expected_sha256=source_sha)
    if any(report["source_pool_receipt"] != support for report in reports.values()):
        raise ValueError("original proposal pool changed")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    predict, samples = model.predict, 0

    def counted_predict(emb, act):
        nonlocal samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return predict(emb, act)

    model.predict = counted_predict
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(reference["target_delta"]))
    low, high = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    full_lower, full_upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    pending, selections, rows = [], [], []

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)

    try:
        # First pass contains no candidate execution. All24 choices are sealed together.
        for task in tasks.TASKS:
            lower, upper = tasks.task_bounds(mean, std, tasks.TASKS[task], device=device)
            for source in sorted(reports[task]["rows"], key=lambda row: row["seed"]):
                seed, receipt = source["seed"], source["compilation"]
                context = shared._source_state_context(env, dataset=dataset, seed=seed)
                expected = source["methods"]["dispatch"]["execution_identity"]["initial"]
                if (any(context[key] != source[key] for key in ("source_episode", "source_start_row"))
                        or context["root_rgb_sha256"] != expected["reset_rgb_sha256"] or context["goal_rgb_sha256"] != expected["goal_rgb_sha256"]):
                    raise RuntimeError("original source/root/goal identity changed")
                context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
                bank, bank_support = shared.proposal_parameters(pool, proposals=128, horizon=1, seed=seed, mean=mean, std=std, lower=full_lower, upper=full_upper)
                if bank_support != source["proposal_support"]:
                    raise RuntimeError("original proposal bank changed")
                pixels, goal, history = prepare(context)
                greedy, winners = geometry.endpoint_winners(receipt)
                required = sorted(set(winners) | {0, greedy})
                start_samples = samples
                responses, endpoints, target = collect_responses(model, pixels, goal, history, bank, required, p0=p0, prefix=prefix, lower=lower, upper=upper)
                residual = target - endpoints[greedy]
                common = {index: qp.certified_response_gain(responses[index], residual) for index in required}
                original = receipt["common_task_response"]["candidate_metrics"]
                gates = {"old_common_residual_exact": residual.tolist() == receipt["common_task_response"]["residual"],
                    "old_endpoints_exact": all(endpoints[i].tolist() == original[str(i)]["zero_suffix_endpoint"] for i in required),
                    "old_gains_exact": all(common[i]["bounded_quadratic_gain"] == original[str(i)]["bounded_quadratic_gain"] for i in required)}
                old_decision = qp.qualify_certified_decision(receipt, common)
                old_decision["policy"] = receipt["dispatch"]["policy"]
                gates["old_decision_exact"] = old_decision == receipt["dispatch"]
                if not all(gates.values()):
                    raise RuntimeError("old common-residual computation failed exact replay: " + str(gates))
                metrics = local.response_swap_metrics(responses, endpoints, target, greedy=greedy)
                decision = local.select_goal_local(receipt, metrics)
                selection = {"task": task, "seed": seed, "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                    "decision": decision, "old_decision": receipt["dispatch"], "metrics": metrics, "gates": gates,
                    "response_matrices": {i: value.tolist() for i, value in responses.items()},
                    "endpoints": {i: value.tolist() for i, value in endpoints.items()}, "goal_coordinate": target.tolist(),
                    "actual_response_predict_samples": samples - start_samples, "backward_vjp_rows": sum(value.shape[0] for value in responses.values())}
                selections.append(selection)
                pending.append((source, context, bank, selection, lower, upper))
                print(json.dumps({"phase": "model_only", "task": task, "seed": seed, "old_index": receipt["dispatch"]["selected_program_index"],
                    "new_index": decision["selected_program_index"], "qualified": decision["geometry_qualified_indices"], "gates": gates}), flush=True)
        with sidecars[2].open("x") as handle:
            handle.write(json.dumps({"protocol_sha256": digest(sidecars[1]), "rows": selections}, indent=2, sort_keys=True) + "\n")
        selection_sha = digest(sidecars[2])
        print("All model choices now sealed; starting physical replay without changing any choice.", flush=True)
        for source, context, bank, selection, lower, upper in pending:
            seed, task = selection["seed"], selection["task"]
            limits = np.asarray(tasks.TASKS[task], np.float32)
            selected = selection["decision"]["selected_program_index"]
            raw_prefix = p0._to_raw_actions(bank[selected:selected + 1], mean, std, lower, upper, -limits, limits)[0]
            # If already deployed, preserve its immutable action witness byte-for-byte.
            known = next((value for value in source["methods"].values() if np.array_equal(raw_prefix, np.asarray(value["first_action_witness_raw"], np.float32))), None)
            methods, gates = {}, dict(selection["gates"])
            for name in ("dispatch", "same_bank_greedy", "zero", "cem_mpc", "goal_local"):
                old = source["methods"].get(name) if name != "goal_local" else known
                actions = None if old is None else np.asarray(old["action_witness_raw"], np.float32)
                first = raw_prefix if name == "goal_local" else actions[:5]
                root, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                identity = p0._verify_execution_identity(root, goal_rgb, context["root_frame"], context["goal"])
                advanced = execution.advance_executed_context(env, context=context, raw_action_block=first, action_block=5, verified_root_frame=root)
                pixels, goal, history = prepare(advanced)
                emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                observed_cost = float((emb[0, -1] - goal_latent).square().mean())
                compiled_receipt = None
                if actions is None:
                    params, compiled_receipt = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=2,
                        action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                    suffix = p0._to_raw_actions(params, mean, std, lower, upper, -limits, limits)[0]
                    actions = np.concatenate((first, suffix))
                # Execute the entire exact suffix directly at this observed state.
                execution._execute_raw_steps(env, actions[5:])
                physics = env.unwrapped.env.physics
                distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                methods[name] = {"physical": {"final_qpos_distance": distance}, "first_action_witness_raw": first.tolist(),
                    "action_witness_raw": actions.tolist(), "observed_first_rgb_goal_cost": observed_cost,
                    "execution_identity": identity, "reused_sealed_suffix": old is not None, "new_suffix_receipt": compiled_receipt}
                gates[name + "_task_bounds"] = tasks.action_limit_gate(methods[name], limits)
                if old is not None:
                    gates[name + "_distance_exact"] = distance == old["physical"]["final_qpos_distance"]
                    gates[name + "_successor_cost_exact"] = observed_cost == old["observed_first_rgb_goal_cost"]
            row = {"task": task, "seed": seed, "source_episode": source["source_episode"], "decision": selection["decision"],
                "old_decision": selection["old_decision"], "methods": methods, "gates": gates,
                "actual_rgb_non_greedy": methods["goal_local"]["observed_first_rgb_goal_cost"] >= methods["zero"]["observed_first_rgb_goal_cost"] + 1e-6}
            with (sidecars[3] / f"{task}_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"phase": "physical", "task": task, "seed": seed, "gates": gates,
                "actual_final": {name: value["physical"]["final_qpos_distance"] for name, value in methods.items()}}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("exact physical replay failed; all partial evidence retained")
    finally:
        model.predict = predict
        env.close()
    aggregates = {}
    for task in tasks.TASKS:
        subset = [row for row in rows if row["task"] == task]
        eligible = [row for row in subset if row["decision"]["reconfiguration_witness_found"]]
        aggregates[task] = {"roots": len(subset), "model_witnesses": len(eligible),
            "decisions_changed": sum(row["decision"]["selected_program_index"] != row["old_decision"]["selected_program_index"] for row in subset),
            "witness_actual_rgb_non_greedy": sum(row["actual_rgb_non_greedy"] for row in eligible),
            "mean_final_distance": {name: float(np.mean([row["methods"][name]["physical"]["final_qpos_distance"] for row in subset]))
                for name in ("goal_local", "dispatch", "same_bank_greedy", "zero", "cem_mpc")},
            "paired": {name: matrix._paired(subset, "goal_local", name, protocol["bootstrap_seed"]) for name in ("dispatch", "same_bank_greedy", "zero", "cem_mpc")}}
    gates = {"all_task_root_identities_retained": {(row["task"], row["seed"]) for row in rows} == {(task, seed) for task in tasks.TASKS for seed in reference["root_seeds"]},
        "all_replay_gates": all(all(row["gates"].values()) for row in rows), "cuda_only": True,
        "weights_unchanged": before == shared._parameter_hash(model), "selection_unchanged": digest(sidecars[2]) == selection_sha,
        "inputs_unchanged": all(digest(Path(item["path"])) == item["sha256"] for item in bindings),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_goal_local_geometry_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "selection_sha256": selection_sha,
        "actual_model_predict_samples": samples, "aggregate": aggregates, "rows": rows, "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregates}), flush=True)


if __name__ == "__main__":
    main()
