"""Causal observed-RGB execution repair of the sealed masked suffix programs.

Keep prefixes/indices/tasks/goals unchanged; compare exact replay, one-packet
goal replanning and tracking root-imagined waypoints on every known root.
No future physical vector is passed to the compiler. This is not P1 admission.
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
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability import reacher_d2p_masked_programs as programs
from experiments.spectral_reachability import reacher_d2p_joint_distance as metric
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.analyze_reacher_d2p_masked_packet_fidelity import FAMILIES, TASKS, verify_files
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

MODES = ("replay", "goal_feedback", "waypoint_feedback")
RESIDUAL_MODES = (*MODES, "causal_residual_goal")


def causal_residual(observed, raw_prediction):
    """One-step lag, unit-gain disturbance estimate; no fitted parameters.

    Inputs must be from the same *already executed* packet. Never subtract a
    previously corrected prediction (which would double-compensate the bias).
    The caller seals its parent execution hash before using this next packet.
    """
    if (observed.shape != (192,) or raw_prediction.shape != (192,)
            or observed.device != raw_prediction.device or observed.dtype != raw_prediction.dtype
            or not torch.isfinite(observed).all() or not torch.isfinite(raw_prediction).all()):
        raise ValueError("finite same-device full192 observed/raw prediction required")
    residual = (observed.detach()-raw_prediction.detach()).clone()
    if not torch.isfinite(residual).all():
        raise ValueError("nonfinite causal residual")
    return residual


def initial_plans(report):
    """Whitelist only root-imagined predictions/actions, never actual futures."""
    output = []
    for row in sorted(report["rows"], key=lambda r: r["seed"]):
        tasks = {}
        for task in TASKS:
            branches = {}
            for family in FAMILIES:
                b = row["tasks"][task][family]
                branches[family] = {"bank_index": b["bank_index"], "raw_actions": b["raw_actions"],
                    "waypoints": [p["vectors"]["free"] for p in b["packets"]],
                    "goal": b["packets"][0]["vectors"]["goal"]}
            tasks[task] = branches
        output.append({"seed": row["seed"], "source_episode": row["source_episode"],
            "source_start_row": row["source_start_row"], "tasks": tasks})
    return output


def validate_stage(stage):
    if (stage["mode"] not in RESIDUAL_MODES or stage["task"] not in TASKS or stage["family"] not in FAMILIES
            or stage["packet"] not in (1, 2, 3) or isinstance(stage["packet"], bool)
            or stage["raw_steps_observed_before_selection"] != 5*(stage["packet"]-1)
            or stage["unexecuted_observations_used"] is not False):
        raise ValueError("causal per-branch observation frontier required")
    action, target, prediction = (np.asarray(stage[k], float) for k in ("raw_actions", "target_vector", "predicted_vector"))
    if (action.shape != (5, 2) or any(v.shape != (192,) for v in (target, prediction))
            or any(not np.isfinite(v).all() for v in (action, target, prediction))
            or (np.abs(action) > 1).any()
            or (stage["task"] == "shoulder_only" and np.any(action[:, 1] != 0))
            or (stage["task"] == "wrist_only" and np.any(action[:, 0] != 0))):
        raise ValueError("finite full target/prediction and exact masked raw packet required")
    optimize = stage["packet"] > 1 and stage["mode"] != "replay"
    receipt = stage["optimizer_receipt"]
    if optimize:
        if receipt is None or receipt["iterations"] != 4 or receipt["physical_labels_used"] is not False:
            raise ValueError("fixed model-only four-step feedback solve required")
    elif receipt is not None:
        raise ValueError("first prefix and replay actions cannot be reoptimized")
    if stage["mode"] == "causal_residual_goal":
        raw, bias = (np.asarray(stage[k], float) for k in ("raw_predicted_vector", "causal_residual_vector"))
        if (any(v.shape != (192,) or not np.isfinite(v).all() for v in (raw, bias))
                or stage["residual_source_packet"] != stage["packet"]-1
                or not np.allclose(raw+bias, prediction, rtol=2e-6, atol=2e-7)
                or (stage["packet"] == 1 and np.any(bias != 0))):
            raise ValueError("causal previous-packet residual and corrected forecast required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--causal-residual", action="store_true",
        help="Explicit exploratory unit-gain past-error correction; retains all original controls.")
    args = parser.parse_args()
    run_modes = RESIDUAL_MODES if args.causal_residual else MODES
    reference, _ = verify_files(args.reference)
    old = reference["sealed_protocol"]
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (digest(args.policy/"weights.pt") != old["weights_sha256"]
            or digest(args.adapter_artifact) != old["adapter_sha256"]
            or source_sha != old["source_sha256_receipt"]):
        raise ValueError("unchanged official checkpoint/source/stats required")
    paths = {k: args.output.with_suffix(s) for k, s in (("protocol", ".protocol.json"),
        ("plans", ".plans.json"), ("stages", ".stages"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in (args.output, *paths.values())):
        raise FileExistsError("new immutable outputs required")
    contexts = {}
    for b in old["references"]:
        prior = json.loads(Path(b["path"]).read_text())["sealed_protocol"]
        forecasts = json.loads(Path(b["predictions_path"]).read_text())
        contexts.update({r["seed"]: (prior.get("goal_contract", "joint-feasible-v1"), r) for r in forecasts["rows"]})
    plans = initial_plans(reference)
    modules = {"auditor": Path(__file__), "programs": Path(programs.__file__), "chart": Path(chart.__file__),
        "context": Path(execution.__file__), "p0": Path(p0.__file__), "goal_contract": Path(goals.__file__),
        "feasible_goal": Path(goals.feasible_goal_delta.__code__.co_filename),
        "qp": Path(programs.solve_box_quadratic.__code__.co_filename),
        "source_context": Path(shared._source_state_context.__code__.co_filename),
        "loader": Path(shared.load_compatible_pretrained.__code__.co_filename), "joint_metric": Path(metric.__file__)}
    protocol = {"root_seeds": [r["seed"] for r in plans], "reference_path": str(args.reference.resolve()),
        "reference_sha256": digest(args.reference), "weights_sha256": old["weights_sha256"],
        "adapter_sha256": old["adapter_sha256"], "source_sha256_receipt": source_sha,
        "families": list(FAMILIES), "tasks": list(TASKS), "modes": list(MODES),
        "physical_branches": 144*len(run_modes), "physical_packets": 432*len(run_modes),
        "expected_predict_samples": 144*(3+67*(len(run_modes)-1)),
        "iterations": 4, "damping": .01, "prefixes_and_initial_selection_unchanged": True,
        "joint_metric": metric.CONTRACT,
        "targets": "final goal or immutable root-imagined packet waypoint, never physical future",
        "scope": "Known16-root causal execution diagnosis. No independent efficacy, equal-compute advantage, changed prefix capability or P1 admission.",
        "implementation_sha256": {k: digest(p) for k, p in modules.items()}}
    protocol["modes"] = list(run_modes)
    if args.causal_residual:
        protocol["causal_residual_contract"] = "unit-gain-last-observed-minus-raw-prediction-v1"
        protocol["residual_parameters_fit"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    write_new(paths["plans"], {"protocol_sha256": digest(paths["protocol"]), "rows": plans})
    plans_sha = digest(paths["plans"])
    for k in ("roots", "stages", "implementation"):
        paths[k].mkdir()
    for k, p in modules.items():
        shutil.copyfile(p, paths["implementation"]/(k+".py"))
    print("HOST CUDA: all16 roots/3 masks/3 fixed prefixes, exact replay versus fixed feedback controls. Only the executed branch's past RGB enters each fixed4-step solve; optional unit-gain lag residual, no fitting, checkpoint training or P1 admission. Modes="+str(run_modes), flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean, std)
    processor = shared._Preprocessor(normalizer)
    mean_t, std_t = (torch.tensor(v, device="cuda") for v in (normalizer.block_mean, normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    params, buffers = shared._parameter_hash(model), {k: v.clone() for k, v in model.named_buffers()}
    original, samples = model.predict, 0

    def counted(emb, actions):
        nonlocal samples
        if emb.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += len(emb)
        return original(emb, actions)

    model.predict = counted

    def prepare(ctx):
        return p0._prepare(processor, ctx["frames"], ctx["root_frame"], ctx["goal"],
            np.asarray(ctx["history_action_blocks"], np.float32), torch.device("cuda"))

    known = {r["seed"]: r for r in reference["rows"]}
    rows = []
    try:
        for initial in plans:
            seed = initial["seed"]
            contract, forecast = contexts[seed]
            env = goals.goal_wrapper(contract)(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
            try:
                ctx = shared._source_state_context(env, dataset=dataset, seed=seed)
                ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
                if any(ctx[k] != forecast[k] for k in ("source_episode", "source_start_row", "root_rgb_sha256", "goal_rgb_sha256")) or ctx["execution_snapshot"] != forecast["initial_snapshot"]:
                    raise RuntimeError("original complete source/root/goal snapshot changed")
                tasks = {}
                for task in TASKS:
                    width = len(chart.active_columns(task))
                    branches = {}
                    for family in FAMILIES:
                        plan = initial["tasks"][task][family]
                        waypoints = torch.tensor(plan["waypoints"], device="cuda")
                        target = torch.tensor(plan["goal"], device="cuda")
                        commands = torch.tensor(np.asarray(plan["raw_actions"], np.float32).reshape(3, 10)[:, chart.active_columns(task)], device="cuda")
                        modes = {}
                        for mode in run_modes:
                            frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed,
                                snapshot=ctx["execution_snapshot"], root_identity="sealed_observation_exact_state")
                            p0._verify_execution_identity(frame, goal_rgb, ctx["root_frame"], ctx["goal"])
                            current, packets, all_raw, parent_sha = ctx, [], [], plans_sha
                            bias = torch.zeros_like(target)
                            for j in range(3):
                                pixels, goal_pixels, history = prepare(current)
                                emb = p0._latent_context(model, pixels, history)
                                if p0._goal_latent(model, goal_pixels).tolist() != plan["goal"]:
                                    raise RuntimeError("immutable goal coordinate changed")
                                objective = waypoints[j] if mode == "waypoint_feedback" else target
                                def terminal(flat):
                                    value = chart.terminal(model, emb, history, flat.reshape(1, 1, width),
                                        task=task, mean=mean_t, std=std_t)[0]
                                    # Keep the exact old arithmetic for all old modes.
                                    return value+bias if mode == "causal_residual_goal" else value
                                command, receipt = commands[j].clone(), None
                                if mode != "replay" and j > 0:
                                    before = samples
                                    command, receipt = programs.fit(terminal, command, objective, iterations=4)
                                    if samples-before != 32:
                                        raise RuntimeError("fixed4-step GN predict accounting drift")
                                with torch.no_grad():
                                    raw_prediction = chart.terminal(model, emb, history, command.reshape(1, 1, width),
                                        task=task, mean=mean_t, std=std_t)[0]
                                    prediction = raw_prediction+bias if mode == "causal_residual_goal" else raw_prediction
                                raw = chart.expand_commands(command.reshape(1, 1, width), task)[0].reshape(5, 2).detach().cpu().numpy()
                                stage = {"seed": seed, "task": task, "family": family, "mode": mode, "packet": j+1,
                                    "initial_plans_sha256": plans_sha, "parent_execution_sha256": parent_sha,
                                    "raw_steps_observed_before_selection": 5*j, "unexecuted_observations_used": False,
                                    "input_root_rgb_sha256": current["root_rgb_sha256"],
                                    "input_history_rgb_sha256": current["history_rgb_sha256"],
                                    "raw_actions": raw.tolist(), "target_vector": objective.tolist(),
                                    "predicted_vector": prediction.tolist(), "optimizer_receipt": receipt}
                                if mode == "causal_residual_goal":
                                    stage.update(raw_predicted_vector=raw_prediction.tolist(),
                                        causal_residual_vector=bias.tolist(), residual_source_packet=j)
                                validate_stage(stage)
                                stage_path = paths["stages"]/f"{seed}_{task}_{family}_{mode}_{j+1}.selection.json"
                                write_new(stage_path, stage)
                                current = execution.advance_executed_context(env, context=current, raw_action_block=raw,
                                    action_block=5, verified_root_frame=current["root_frame"])
                                current["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(current["root_frame"])
                                new_pixels, _, new_history = prepare(current)
                                actual = p0._latent_context(model, new_pixels, new_history)[0, -1]
                                if mode == "causal_residual_goal":
                                    bias = causal_residual(actual, raw_prediction)
                                physical = {"packet": j+1, "observed_vector": actual.tolist(),
                                    "qpos": np.asarray(env.unwrapped.env.physics.data.qpos, np.float64)[:2].tolist(),
                                    "root_rgb_sha256": current["root_rgb_sha256"], "action_sha256": current["advanced_action_sha256"]}
                                baseline = known[seed]["tasks"][task][family]["packets"][j]
                                if (mode == "replay" or j == 0) and physical != baseline["physical"]:
                                    raise RuntimeError("original complete physical packet changed")
                                if (mode == "replay" or j == 0) and prediction.tolist() != baseline["vectors"]["actual_history"]:
                                    raise RuntimeError("original actual-history prediction changed")
                                packet = {"physical": physical, "stage_path": str(stage_path.resolve()),
                                    "stage_sha256": digest(stage_path), "observed_goal_cost": float((actual-target).square().mean()),
                                    "observed_waypoint_l2": float((actual-waypoints[j]).norm())}
                                execution_path = stage_path.with_suffix(".execution.json")
                                write_new(execution_path, packet)
                                parent_sha = digest(execution_path)
                                packet["execution_path"] = str(execution_path.resolve())
                                packet["execution_sha256"] = parent_sha
                                packets.append(packet)
                                all_raw.extend(raw.tolist())
                            physics = env.unwrapped.env.physics
                            modes[mode] = {"packets": packets, "raw_actions": all_raw,
                                "final_rgb_cost": packets[-1]["observed_goal_cost"],
                                "final_qpos_distance": metric.distance(np.asarray(physics.data.qpos, np.float64)[:2],
                                    np.asarray(env.unwrapped.env.task.target_qpos, np.float64)[:2], forecast["goal_admission"]["limited_joints"])}
                        branches[family] = {"bank_index": plan["bank_index"], "modes": modes}
                    tasks[task] = branches
                row = {"seed": seed, "source_episode": initial["source_episode"], "source_start_row": initial["source_start_row"], "tasks": tasks}
                write_new(paths["roots"]/f"root_{seed}.json", row)
                rows.append(row)
                print(json.dumps({"seed": seed, "completed_branches": 9*len(run_modes),
                    "native_final_distance": {t: {m: tasks[t]["common_native"]["modes"][m]["final_qpos_distance"] for m in run_modes} for t in TASKS}}), flush=True)
            finally:
                env.close()
    finally:
        model.predict = original
    panel_gate = "all_declared_branches_packets" if args.causal_residual else "all432_branches1296_packets"
    gates = {"cuda_only": True, "all16_roots": [r["seed"] for r in rows] == protocol["root_seeds"],
        panel_gate: len(rows) == 16 and all(set(r["tasks"]) == set(TASKS)
            and all(set(b) == set(FAMILIES) and all(set(f["modes"]) == set(run_modes)
                and all(len(m["packets"]) == 3 for m in f["modes"].values()) for f in b.values()) for b in r["tasks"].values()) for r in rows),
        "all_replay_and_first_packets_exact": True, "predict_accounting": samples == protocol["expected_predict_samples"],
        "parameters_unchanged": shared._parameter_hash(model) == params,
        "buffers_unchanged": all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()),
        "all_modules_eval": not any(m.training for m in model.modules()),
        "plans_unchanged": digest(paths["plans"]) == plans_sha,
        "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "weights_file_unchanged": digest(args.policy/"weights.pt") == protocol["weights_sha256"],
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    write_new(args.output, {"schema": "reacher_d2p_masked_feedback.v2" if args.causal_residual else "reacher_d2p_masked_feedback.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "protocol_sha256": digest(paths["protocol"]),
        "plans_sha256": plans_sha, "actual_predict_samples": samples, "physical_branches": protocol["physical_branches"], "p1_admitted": False, "rows": rows})
    print(json.dumps({"gates": gates, "actual_predict_samples": samples, "p1_admitted": False}), flush=True)


if __name__ == "__main__":
    main()
