"""Causal fixed-plan waypoint tracking on all eight audited roots.

Seal root-imagined waypoints and complete plans BEFORE any execution. At
each packet use only that branch's already observed RGB/action history.
This changes execution feedback, not the checkpoint or P1 admission rule.
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
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import reacher_d2p_box_action_suffix as box
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.analyze_reacher_d2p_state_native_selector import summarize as validate_reference
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


FAMILIES = ("native_dispatch", "greedy", "zero", "cem")


def initial_plan(choice, family, *, device):
    """Only the pre-prefix selection is accepted; observed suffixes are absent."""
    if family not in FAMILIES:
        raise ValueError("declared initial plan family required")
    if family == "cem":
        value = torch.tensor(choice["cem_params"], dtype=torch.float32, device=device).tanh()
    else:
        index = {"native_dispatch": choice["native_decision"]["selected_program_index"],
            "greedy": choice["native_decision"]["same_bank_greedy_index"], "zero": 0}[family]
        bank = torch.tensor(choice["word_params"], dtype=torch.float32, device=device)
        record = choice["word_records"][index]
        if record["index"] != index:
            raise ValueError("initial word index/receipt identity mismatch")
        tail = torch.tensor(record["planned_commands"], dtype=torch.float32, device=device)
        value = torch.cat((bank[index:index + 1].tanh(), tail), dim=1)
    if value.shape != (1, 6, 10) or not torch.isfinite(value).all() or bool((value.abs() > 1).any()):
        raise ValueError("six finite bounded pre-execution command packets required")
    return value


def validate_stage(stage, *, seed, family, mode, packet, initial_sha, parent_sha):
    if (stage["seed"] != seed or stage["family"] != family or stage["mode"] != mode
            or stage["packet"] != packet or stage["initial_selection_sha256"] != initial_sha
            or stage["parent_execution_sha256"] != parent_sha
            or stage["raw_steps_observed_before_selection"] != (packet - 1) * 5
            or stage["unexecuted_branch_observations_used"] is not False):
        raise ValueError("causal stage identity and observation frontier required")
    raw, target = np.asarray(stage["raw_actions"], float), np.asarray(stage["target_vector"], float)
    if raw.shape != (5, 2) or target.shape != (192,) or not np.isfinite(raw).all() or not np.isfinite(target).all() or np.max(np.abs(raw)) > 1 + 1e-6:
        raise ValueError("finite complete raw actions and frozen target required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    side = {k: args.reference.with_suffix(ext) for k, ext in (("prefix", ".selection.json"), ("suffix", ".suffix.selection.json"), ("protocol", ".protocol.json"))}
    prefix, suffix = (json.loads(side[k].read_text()) for k in ("prefix", "suffix"))
    if (digest(side["prefix"]) != reference["selection_sha256"] or digest(side["suffix"]) != reference["suffix_selection_sha256"]
            or json.loads(side["protocol"].read_text()) != reference["sealed_protocol"]
            or prefix["protocol_sha256"] != digest(side["protocol"]) or suffix["protocol_sha256"] != digest(side["protocol"])
            or suffix["prefix_selection_sha256"] != digest(side["prefix"])):
        raise ValueError("complete immutable reference seals required")
    prior = validate_reference(reference, prefix, suffix)
    if prior["goal_contract"] != "joint-feasible-v1":
        raise ValueError("explicit feasible-goal reference required")
    ref = reference["sealed_protocol"]
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"] or source_sha != ref["source_sha256"]):
        raise ValueError("original checkpoint/source/normalization required")
    paths = {k: args.output.with_suffix(ext) for k, ext in (("protocol", ".protocol.json"), ("initial", ".selection.json"), ("implementation", ".implementation"), ("roots", ".roots"), ("stages", ".stages"))}
    if any(p.exists() for p in [args.output, *paths.values()]):
        raise FileExistsError("new immutable experiment outputs required")
    modules = {"auditor": Path(__file__), "p0": Path(p0.__file__), "context": Path(execution.__file__),
        "goal_contract": Path(goals.__file__), "box": Path(box.__file__), "qp": Path(box.solve_box_quadratic.__code__.co_filename),
        "rescue": Path(box.rescue.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename),
        "loader": Path(shared.load_compatible_pretrained.__code__.co_filename)}
    for key in ("p0", "context", "goal_contract", "box", "qp", "rescue", "source_context"):
        if digest(modules[key]) != ref["implementation_sha256"][key]:
            raise ValueError("reference implementation changed: " + key)
    protocol = {"root_seeds": prior["root_seeds"], "reference_path": str(args.reference.resolve()), "reference_sha256": digest(args.reference),
        "prefix_sha256": digest(side["prefix"]), "suffix_sha256": digest(side["suffix"]),
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": source_sha,
        "goal_contract": prior["goal_contract"], "target_delta": ref["target_delta"], "families": list(FAMILIES), "modes": ["replay", "tracking"],
        "packets_per_branch": 6, "raw_steps_per_packet": 5, "damping": .01, "iterations": 4,
        "first_packet_unchanged": True, "all_initial_plans_and_waypoints_sealed_before_any_execution": True,
        "plan_source": "pre-prefix selection only; no observed-prefix suffix decisions", "physical_labels_used_for_planning": False,
        "feedback": "this branch's observed RGB/action history only; frozen root-imagined waypoint at each packet",
        "source_verification": "Reuse existing SHA-bound official source receipt; no new complete-file SHA claimed.",
        "implementation_sha256": {k: digest(p) for k, p in modules.items()},
        "scope": "Known-eight-root causal feedback intervention; unequal compute, not fresh efficacy, P1 admission or theory rejection."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    for name in ("implementation", "roots", "stages"):
        paths[name].mkdir()
    for k, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (k + ".py"))
    print("HOST CUDA: seal all8 roots x4 initial plans and all6 imagined waypoints before execution. Replay versus four-iteration causal RGB waypoint tracking; no future other-branch images, training, or P1 admission.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before, buffers = shared._parameter_hash(model), {k: v.clone() for k, v in model.named_buffers()}
    predict, samples, expected_samples = model.predict, 0, 0

    def counted(emb, actions):
        nonlocal samples
        if emb.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return predict(emb, actions)

    model.predict = counted
    env = goals.goal_wrapper(prior["goal_contract"])(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))
    lo, hi = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = (torch.as_tensor(v, device=device) for v in (lo, hi))

    def prepare(ctx):
        return p0._prepare(processor, ctx["frames"], ctx["root_frame"], ctx["goal"], np.asarray(ctx["history_action_blocks"], np.float32), device)

    def raw_commands(commands):
        normalized = ((lower + upper) / 2 + (upper - lower) / 2 * commands)[0].detach().cpu().numpy()
        return p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)

    choices = {r["seed"]: r for r in prefix["rows"]}
    pending, initial_rows, rows = [], [], []
    try:
        for old in sorted(reference["rows"], key=lambda r: r["seed"]):
            seed, plans = old["seed"], {}
            ctx = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(ctx[k] != choices[seed][k] for k in ("source_episode", "source_start_row", "root_rgb_sha256", "goal_rgb_sha256")):
                raise RuntimeError("source root/goal identity changed")
            admission = goals.capture_goal_admission(env, prior["goal_contract"])
            ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
            pixels, goal, history = prepare(ctx)
            emb, target = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            for family in FAMILIES:
                commands = initial_plan(choices[seed], family, device=device)
                raw = raw_commands(commands)
                if not np.array_equal(raw, np.asarray(old["methods"][family + "_carry"]["action_witness_raw"], np.float32)):
                    raise RuntimeError("initial complete carry action witness changed")
                with torch.no_grad():
                    predicted = [box.terminal_commands(model, emb, history, commands[:, :i], action_lower=lower, action_upper=upper)[0] for i in range(1, 7)]
                expected_samples += 21
                plans[family] = {"commands": commands.tolist(), "raw_actions": raw.tolist(), "waypoints": [z.tolist() for z in predicted],
                    "predicted_goal_costs": [float((z - target).square().mean()) for z in predicted], "baseline_actions_bit_exact": True}
            record = {"seed": seed, "source_episode": ctx["source_episode"], "source_start_row": ctx["source_start_row"],
                "root_rgb_sha256": ctx["root_rgb_sha256"], "goal_rgb_sha256": ctx["goal_rgb_sha256"], "goal_vector": target.tolist(),
                "goal_admission": admission, "plans": plans}
            initial_rows.append(record)
            pending.append((old, ctx, record))
        write_new(paths["initial"], {"protocol_sha256": digest(paths["protocol"]), "rows": initial_rows})
        initial_sha = digest(paths["initial"])
        print(json.dumps({"initial_selection_sha256": initial_sha, "all32_initial_plans_sealed": True}), flush=True)
        for old, ctx, initial in pending:
            seed, methods = initial["seed"], {}
            for family in FAMILIES:
                plan = initial["plans"][family]
                targets = torch.tensor(plan["waypoints"], device=device)
                commands = torch.tensor(plan["commands"], device=device)
                goal_vector = torch.tensor(initial["goal_vector"], device=device)
                for mode in protocol["modes"]:
                    frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=ctx["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    p0._verify_execution_identity(frame, goal_rgb, ctx["root_frame"], ctx["goal"])
                    current, packets, all_raw, parent_sha = ctx, [], [], initial_sha
                    for i in range(6):
                        pixels, _, history = prepare(current)
                        emb = p0._latent_context(model, pixels, history)
                        action = commands[:, i:i + 1]
                        receipt = None
                        if mode == "tracking" and i > 0:
                            start = samples
                            action, receipt = box.compile_box_snapshots(model, targets[i], emb, history, initial_commands=action,
                                action_lower=lower, action_upper=upper, damping=.01, checkpoints=(4,))[4]
                            if samples - start != receipt["model_predict_sample_evaluations"]:
                                raise RuntimeError("tracking optimizer compute receipt mismatch")
                            expected_samples += receipt["model_predict_sample_evaluations"]
                        with torch.no_grad():
                            predicted = box.terminal_commands(model, emb, history, action, action_lower=lower, action_upper=upper)[0]
                        expected_samples += 1
                        raw = raw_commands(action)
                        stage = {"seed": seed, "family": family, "mode": mode, "packet": i + 1,
                            "initial_selection_sha256": initial_sha, "parent_execution_sha256": parent_sha,
                            "raw_steps_observed_before_selection": i * 5, "unexecuted_branch_observations_used": False,
                            "input_root_rgb_sha256": current["root_rgb_sha256"], "input_history_rgb_sha256": current["history_rgb_sha256"],
                            "raw_actions": raw.tolist(), "target_vector": plan["waypoints"][i], "predicted_vector": predicted.tolist(), "receipt": receipt}
                        validate_stage(stage, seed=seed, family=family, mode=mode, packet=i + 1, initial_sha=initial_sha, parent_sha=parent_sha)
                        stage_path = paths["stages"] / f"{seed}_{family}_{mode}_{i + 1}.selection.json"
                        write_new(stage_path, stage)
                        stage_sha = digest(stage_path)
                        current = execution.advance_executed_context(env, context=current, raw_action_block=raw, action_block=5, verified_root_frame=current["root_frame"])
                        current["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(current["root_frame"])
                        pixels, _, history = prepare(current)
                        actual = p0._latent_context(model, pixels, history)[0, -1]
                        packet = {"packet": i + 1, "stage_selection_sha256": stage_sha, "stage_selection_path": str(stage_path.resolve()),
                            "root_rgb_sha256": current["root_rgb_sha256"], "history_rgb_sha256": current["history_rgb_sha256"],
                            "action_sha256": current["advanced_action_sha256"], "observed_vector": actual.tolist(),
                            "observed_goal_cost": float((actual - goal_vector).square().mean()),
                            "observed_waypoint_error_l2": float((actual - targets[i]).norm()),
                            "predicted_waypoint_error_l2": float((predicted - targets[i]).norm())}
                        execution_path = stage_path.with_suffix(".execution.json")
                        write_new(execution_path, packet)
                        parent_sha = digest(execution_path)
                        packets.append(packet)
                        all_raw.extend(raw.tolist())
                    physics = env.unwrapped.env.physics
                    distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                    physical = {"final_qpos_distance": distance, "final_rgb_cost": packets[-1]["observed_goal_cost"]}
                    if mode == "replay" and (all_raw != plan["raw_actions"] or any(abs(physical[k] - old["methods"][family + "_carry"]["physical"][k]) > 2e-6 for k in physical)):
                        raise RuntimeError("all-six-packet carry replay changed")
                    methods[family + "_" + mode] = {"packets": packets, "action_witness_raw": all_raw, "physical": physical,
                        "baseline_carry_metrics_replayed": mode == "replay", "final_execution_sha256": parent_sha}
                print(json.dumps({"seed": seed, "family": family, "final_distance": {m: methods[family + "_" + m]["physical"]["final_qpos_distance"] for m in protocol["modes"]}}), flush=True)
            row = {"seed": seed, "source_episode": initial["source_episode"], "source_start_row": initial["source_start_row"], "methods": methods}
            write_new(paths["roots"] / f"root_{seed}.json", row)
            rows.append(row)
    finally:
        model.predict = predict
        env.close()
    gates = {"cuda_only": True, "all8_roots": [r["seed"] for r in rows] == protocol["root_seeds"],
        "all64_branches_and384_packets": len(rows) == 8 and all(len(r["methods"]) == 8 and all(len(m["packets"]) == 6 for m in r["methods"].values()) for r in rows),
        "initial_plans_and_waypoints_unchanged": digest(paths["initial"]) == initial_sha,
        "reference_and_decisions_unchanged": digest(args.reference) == protocol["reference_sha256"] and digest(side["prefix"]) == protocol["prefix_sha256"] and digest(side["suffix"]) == protocol["suffix_sha256"],
        "parameters_unchanged": shared._parameter_hash(model) == before, "buffers_unchanged": all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()),
        "all_modules_eval": not any(m.training for m in model.modules()), "predict_sample_accounting": samples == expected_samples,
        "weights_file_unchanged": digest(args.policy / "weights.pt") == protocol["weights_sha256"],
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][k] for k, path in modules.items())}
    write_new(args.output, {"schema": "reacher_d2p_waypoint_tracking.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "protocol_sha256": digest(paths["protocol"]), "initial_selection_sha256": initial_sha,
        "gates": gates, "rows": rows, "actual_model_predict_samples": samples, "expected_model_predict_samples": expected_samples})
    print(json.dumps({"gates": gates, "actual_model_predict_samples": samples}), flush=True)


if __name__ == "__main__":
    main()
