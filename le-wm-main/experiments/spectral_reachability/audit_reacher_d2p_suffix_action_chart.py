"""All-root fixed-prefix audit of tanh saturation vs a direct command box.

Every reference dispatch prefix is fixed. Start legacy and box refinements
from the EXACT same incumbent action, compare 8/32 extra iterations, and add
same-prefix CEM. All new suffix choices are sealed before terminal outcomes.
This known-root intervention neither admits P1 nor changes its prefix gate.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_action_words as words
from experiments.spectral_reachability import reacher_d2p_box_action_suffix as box
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def validate_reference(report, selections):
    required_gates = {"all_declared_roots", "all_execution_gates", "cuda_only", "implementation_unchanged",
        "selection_unchanged", "source_episodes_disjoint", "weights_unchanged"}
    if (report.get("schema") != "reacher_d2p_suffix_continuation_confirmation.v1" or report.get("status") != "completed"
            or not required_gates.issubset(report.get("gates", {}))
            or not all(value is True for value in report["gates"].values())):
        raise ValueError("complete strict integrity-passing continuation reference required")
    seeds = sorted(report["sealed_protocol"]["root_seeds"])
    rows = sorted(report["rows"], key=lambda row: row["seed"])
    choices = {row["seed"]: row for row in selections["rows"]}
    if len(seeds) != 8 or len(seeds) != len(set(seeds)) or [row["seed"] for row in rows] != seeds or len(choices) != len(selections["rows"]) or sorted(choices) != seeds:
        raise ValueError("complete duplicate-free root identity join required")
    for row in rows:
        choice = choices[row["seed"]]
        if any(row[key] != choice[key] for key in ("source_episode", "source_start_row")) or row["decision"] != choice["receipt"]["dispatch"]:
            raise ValueError("source or sealed decision mismatch")
    return [(row, choices[row["seed"]]) for row in rows]


def terminal_rgb_batch(image, goal, action_dim):
    """Renderer views may have negative strides; preserve pixels exactly."""
    return {"pixels": np.ascontiguousarray(image, dtype=np.uint8)[None, None],
        "goal": np.ascontiguousarray(goal, dtype=np.uint8)[None, None],
        "action": np.zeros((1, 1, action_dim), np.float32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.reference.read_text())
    old_selection = args.reference.with_suffix(".selection.json")
    if digest(old_selection) != report["selection_sha256"]:
        raise ValueError("historical selection SHA changed")
    joined = validate_reference(report, json.loads(old_selection.read_text()))
    ref = report["sealed_protocol"]
    old_protocol = args.reference.with_suffix(".protocol.json")
    if (json.loads(old_protocol.read_text()) != ref
            or digest(old_protocol) != json.loads(old_selection.read_text())["protocol_sha256"]):
        raise ValueError("historical protocol seal changed")
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("frozen weights or normalizer changed")
    p0, p1, solver, rescue = words.p0, words.p1, words.solver, words.rescue
    modules = {"auditor": Path(__file__), "box": Path(box.__file__), "qp": Path(qp.__file__), "words": Path(words.__file__),
        "p0": Path(p0.__file__), "prefix": Path(p1.__file__), "solver": Path(solver.__file__), "rescue": Path(rescue.__file__),
        "context": Path(execution.__file__), "cem": Path(cem.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename)}
    for key, path in modules.items():
        if key in ref["implementation_sha256"] and key != "auditor" and digest(path) != ref["implementation_sha256"][key]:
            raise ValueError("historical implementation changed: " + key)
    output = args.output.resolve()
    paths = {name: output.with_suffix(suffix) for name, suffix in
        (("protocol", ".protocol.json"), ("selection", ".selection.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(path.exists() for path in [output, *paths.values()]):
        raise FileExistsError("new immutable output and sidecars required")
    protocol = {"reference_path": str(args.reference.resolve()), "reference_sha256": digest(args.reference),
        "root_seeds": ref["root_seeds"], "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"],
        "source_sha256": ref["source_sha256"], "target_delta": ref["target_delta"], "damping": .01,
        "extra_iteration_checkpoints": [8, 32], "fd_epsilon": .001, "fd_coordinates": 8, "fd_median_relative_error_max": .1,
        "forward_atol": 0., "chain_rule_relative_error_max": 1e-5,
        "cem": {"horizon": 2, "samples": 128, "steps": 3, "elite": 16, "seed_offset": 70000},
        "all_roots_sealed_before_terminal_execution": True, "physical_labels_used_for_planning": False,
        "prefix_fixed": True, "physical_raw_steps": 30,
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Known-root numerical chart/iteration audit, unequal compute, not fresh efficacy or P1 admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (key + ".py"))
    print("Fixed dispatch prefixes on ALL8 roots: incumbent, extra tanh8/32, direct-box8/32, same-prefix CEM. Seal every suffix before outcomes; host CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    from experiments.spectral_reachability import reacher_d2p_packet_support as support
    if source_sha != ref["source_sha256"] or support._sha256(dataset) != source_sha:
        raise ValueError("complete official source SHA changed")
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
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))
    lo, hi = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)

    def raw_params(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    def raw_commands(commands):
        normalized = ((lower + upper) / 2 + (upper - lower) / 2 * commands)[0].detach().cpu().numpy()
        decoded = p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)
        if np.any(decoded < env.action_space.low - 1e-6) or np.any(decoded > env.action_space.high + 1e-6):
            raise RuntimeError("direct commands decode outside the physical box")
        return decoded

    pending, selections, rows = [], [], []
    try:
        for old, choice in joined:
            seed = old["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if (any(context[key] != old[key] for key in ("source_episode", "source_start_row"))
                    or any(context[key] != choice[key] for key in ("root_rgb_sha256", "goal_rgb_sha256"))):
                raise RuntimeError("source/RGB root changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
            identity = p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
            first = np.asarray(old["methods"]["dispatch_incumbent"]["action_witness_raw"], np.float32)[:20]
            advanced, boundaries = words.advance_program(env, context, first, verified_root_frame=frame)
            pixels, goal, history = prepare(advanced)
            emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            prefix_cost = float((emb[0, -1] - goal_latent).square().mean())
            old_suffix = args.reference.with_suffix(".roots") / f"suffix_{seed}_dispatch.selection.json"
            if digest(old_suffix) != old["branches"]["dispatch"]["suffix_selection_sha256"]:
                raise ValueError("historical suffix seal changed")
            old_record = json.loads(old_suffix.read_text())
            incumbent = torch.tensor(old_record["suffix_params"]["dispatch_incumbent"], device=device, dtype=history.dtype)
            initial = incumbent.tanh().detach()

            def predicted_commands(value):
                z = box.terminal_commands(model, emb, history, value.reshape_as(initial), action_lower=lower, action_upper=upper)[0]
                return p0._planning_coordinate(model, z, role="predicted")

            def predicted_params(value):
                z = p1._terminal_from_state(model, emb, history, value.reshape_as(incumbent), action_lower=lower, action_upper=upper)[0]
                return p0._planning_coordinate(model, z, role="predicted")

            with torch.no_grad():
                old_pred, command_pred = predicted_params(incumbent), predicted_commands(initial)
            j_command = torch.autograd.functional.jacobian(predicted_commands, initial.reshape(-1).requires_grad_(True), vectorize=True)
            j_params = torch.autograd.functional.jacobian(predicted_params, incumbent.reshape(-1).requires_grad_(True), vectorize=True)
            chain = j_command * (1 - initial.reshape(-1).square())[None]
            chain_error = float((chain - j_params).norm() / j_params.norm().clamp_min(1e-8))
            fd_errors = []
            indices = np.random.default_rng(seed).choice(initial.numel(), size=8, replace=False)
            with torch.no_grad():
                for index in indices:
                    left, right = initial.reshape(-1).clone(), initial.reshape(-1).clone()
                    left[index] = (left[index] - .001).clamp(-1, 1)
                    right[index] = (right[index] + .001).clamp(-1, 1)
                    finite = (predicted_commands(right) - predicted_commands(left)) / (right[index] - left[index])
                    fd_errors.append(float((finite - j_command[:, index]).norm() / finite.norm().clamp_min(1e-8)))
            goal_coordinate = p0._planning_coordinate(model, goal_latent, role="goal")
            incumbent_cost = float((old_pred - goal_coordinate).square().mean())
            gates = {"root_identity_exact": identity == old["methods"]["dispatch_incumbent"]["execution_identity"],
                "prefix_history_exact": boundaries == old["methods"]["dispatch_incumbent"]["prefix_boundaries"],
                "prefix_rgb_cost_exact": prefix_cost == old["methods"]["dispatch_incumbent"]["observed_prefix_rgb_cost"],
                "incumbent_actions_exact": np.array_equal(raw_params(incumbent), np.asarray(old["methods"]["dispatch_incumbent"]["action_witness_raw"], np.float32)[20:]),
                "command_actions_exact": np.array_equal(raw_commands(initial), raw_params(incumbent)),
                "command_forward_exact": bool(torch.equal(old_pred, command_pred)), "chain_rule": chain_error <= 1e-5,
                "direct_command_fd": float(np.median(fd_errors)) <= .1,
                "incumbent_cost_exact": incumbent_cost == old_record["receipt"]["model_costs"][old_record["receipt"]["selected"]]}
            write_new(paths["roots"] / f"preflight_{seed}.json", {"seed": seed, "gates": gates,
                "chain_rule_relative_error": chain_error, "fd_relative_errors": fd_errors})
            if not all(gates.values()):
                raise RuntimeError(f"chart equivalence failed {seed}: {gates}; chain={chain_error}, FD={fd_errors}")
            start = samples
            legacy = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=2,
                action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8, 32),
                accept_update=rescue.accept_with_fine_rescue, initial_params=incumbent)
            if samples - start != legacy[32][1]["model_predict_sample_evaluations"]:
                raise RuntimeError("legacy accounting drift")
            start = samples
            boxed = box.compile_box_snapshots(model, goal_latent, emb, history, initial_commands=initial,
                action_lower=lower, action_upper=upper, checkpoints=(8, 32))
            if samples - start != boxed[32][1]["model_predict_sample_evaluations"]:
                raise RuntimeError("box accounting drift")
            start = samples
            cem_params, _, cem_receipt = cem.search_with_incumbent(lambda params: p0._cost_fn(model, pixels, goal, history, params,
                action_lower=lower, action_upper=upper), future_horizon=2, action_dim=history.shape[-1], device=device,
                samples=128, steps=3, elite=16, seed=seed + 70000)
            if samples - start != 768:
                raise RuntimeError("fixed-prefix CEM accounting drift")
            raw = {"incumbent": raw_params(incumbent), "same_prefix_cem": raw_params(cem_params)}
            costs = {"incumbent": incumbent_cost}
            receipts = {"same_prefix_cem": cem_receipt}
            for step in (8, 32):
                raw[f"tanh_extra_{step}"] = raw_params(legacy[step][0])
                raw[f"box_extra_{step}"] = raw_commands(boxed[step][0])
                costs[f"tanh_extra_{step}"] = legacy[step][1]["trace"][-1]["cost_after"]
                costs[f"box_extra_{step}"] = boxed[step][1]["trace"][-1]["cost_after"]
                receipts[f"tanh_extra_{step}"] = legacy[step][1]
                receipts[f"box_extra_{step}"] = boxed[step][1]
            with torch.no_grad():
                costs["same_prefix_cem"] = float((predicted_params(cem_params) - goal_coordinate).square().mean())
            best = min(costs, key=lambda name: (costs[name], name != "incumbent", name))
            record = {"seed": seed, "source_episode": old["source_episode"], "source_start_row": old["source_start_row"],
                "prefix_actions_raw": first.tolist(), "suffix_actions_raw": {name: value.tolist() for name, value in raw.items()},
                "model_costs": costs, "model_best": best, "receipts": receipts, "gates": gates,
                "saturation_fraction_at_same_initial_action": float((initial.abs() >= .99).float().mean()),
                "chain_rule_relative_error": chain_error, "fd_relative_errors": fd_errors}
            selections.append(record)
            pending.append((old, advanced, raw, record))
            print(json.dumps({"seed": seed, "gates": gates, "model_costs": costs, "model_best": best}), flush=True)
        write_new(paths["selection"], {"protocol_sha256": digest(paths["protocol"]), "rows": selections})
        selection_sha = digest(paths["selection"])
        print("All8 roots and ALL numerical suffix variants sealed. Begin physical terminal measurement.", flush=True)
        for old, advanced, raw, record in pending:
            seed, methods, gates = old["seed"], {}, dict(record["gates"])
            for name, actions in raw.items():
                execution.restore_execution_snapshot(env, seed=seed, snapshot=advanced["execution_snapshot"], root_identity="sealed_observation_exact_state")
                execution._execute_raw_steps(env, actions)
                physics = env.unwrapped.env.physics
                distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                image = np.asarray(env.render(), np.uint8)
                prepared = processor._prepare_info(terminal_rgb_batch(image, advanced["goal"], mean.size))
                actual_rgb_cost = float((p0._goal_latent(model, prepared["pixels"].to(device)) - p0._goal_latent(model, prepared["goal"].to(device))).square().mean())
                complete = np.concatenate((np.asarray(record["prefix_actions_raw"], np.float32), actions))
                if complete.shape != (30, 2) or not np.isfinite(complete).all() or np.abs(complete).max() > 1 + 1e-6:
                    raise RuntimeError("bounded complete action witness required")
                methods[name] = {"physical": {"final_qpos_distance": distance, "final_rgb_cost": actual_rgb_cost}, "action_witness_raw": complete.tolist()}
            methods["model_best"] = methods[record["model_best"]]
            gates["incumbent_physical_exact"] = methods["incumbent"]["physical"]["final_qpos_distance"] == old["methods"]["dispatch_incumbent"]["physical"]["final_qpos_distance"]
            row = {"seed": seed, "source_episode": old["source_episode"], "source_start_row": old["source_start_row"],
                "methods": methods, "model_costs": record["model_costs"], "model_best": record["model_best"], "gates": gates}
            write_new(paths["roots"] / f"root_{seed}.json", row)
            rows.append(row)
            print(json.dumps({"seed": seed, "distances": {name: item["physical"]["final_qpos_distance"] for name, item in methods.items()}, "gates": gates}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("exact baseline physical replay failed")
    finally:
        model.predict = predict
        env.close()
    names = sorted(rows[0]["methods"])
    aggregate = {"roots": len(rows), "physical_branches": 6 * len(rows),
        "mean_distance": {name: float(np.mean([row["methods"][name]["physical"]["final_qpos_distance"] for row in rows])) for name in names}}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_roots_retained": [row["seed"] for row in rows] == sorted(ref["root_seeds"]),
        "all_execution_gates": all(all(row["gates"].values()) for row in rows),
        "selection_unchanged": digest(paths["selection"]) == selection_sha,
        "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    write_new(output, {"schema": "reacher_d2p_suffix_action_chart_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "selection_sha256": selection_sha, "gates": gates,
        "aggregate": aggregate, "rows": rows, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
