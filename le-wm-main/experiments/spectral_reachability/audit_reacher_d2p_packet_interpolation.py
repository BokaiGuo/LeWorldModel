"""Host-CUDA fixed source-to-counterfactual packet intervention.

Keep ALL reference roots/branches, not only known inversions. Interpolate raw
torques from the exact official next packet to each previously executed first
packet. This is an evaluator, not an action selector or a P1 gate override.
Use the immutable historical P0 implementation, not a newer stage-gate file.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_causal_innovation as replay
from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import audit_reacher_d2p_source_render_crossover as native
from experiments.spectral_reachability import reacher_d2p_context as execution

ALPHAS = (0., .25, .5, .75, 1.)
METHODS = replay.METHODS


def interpolate_packet(source, target, alpha):
    source, target = np.asarray(source, np.float32), np.asarray(target, np.float32)
    if source.shape != (5, 2) or target.shape != source.shape or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("finite five-step two-torque packets required")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("convex raw-action interpolation required")
    if alpha == 0:
        return source.copy()
    if alpha == 1:
        return target.copy()
    return np.asarray((1 - alpha) * source + alpha * target, np.float32)


def summarize(rows):
    if not rows or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("nonempty duplicate-free root rows required")
    result = {}
    for index, alpha in enumerate(ALPHAS):
        cells = [row["methods"][name][index] for row in rows for name in METHODS]
        eligible = sorted((row for row in rows if row["model_eligible"]), key=lambda row: row["seed"])
        margins = [{"seed": row["seed"],
            "predicted": row["methods"]["dispatch"][index]["predicted_cost"] - row["methods"]["zero"][index]["predicted_cost"],
            "observed": row["methods"]["dispatch"][index]["observed_cost"] - row["methods"]["zero"][index]["observed_cost"]} for row in eligible]
        result[str(alpha)] = {"branch_count": len(cells),
            "mean_latent_l2_error": float(np.mean([cell["latent_l2_error"] for cell in cells])),
            "mean_abs_goal_cost_error": float(np.mean([abs(cell["predicted_cost"] - cell["observed_cost"]) for cell in cells])),
            "eligible_margins_vs_interpolated_zero": margins,
            "strict_order_reversals": [cell["seed"] for cell in margins if cell["predicted"] * cell["observed"] < 0],
            "strict_order_agreements": sum(cell["predicted"] * cell["observed"] > 0 for cell in margins),
            "tied_model_or_observed": sum(cell["predicted"] * cell["observed"] == 0 for cell in margins)}
    return result


def load_historical_p0(reference_path, reference):
    path = reference_path.with_suffix(".implementation") / "p0.py"
    if replay.digest(path) != reference["implementation_sha256"]["p0"]:
        raise ValueError("historical P0 source snapshot does not match reference SHA")
    spec = importlib.util.spec_from_file_location("historical_p0_packet_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--adapter-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.reference.read_text())
    replay.validate_reference(report)
    ref, digest = report["sealed_protocol"], replay.digest
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("checkpoint or action statistics changed")
    p0, historical_path = load_historical_p0(args.reference, ref)
    modules = {"auditor": Path(__file__), "historical_p0": historical_path,
        "context": Path(execution.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename),
        "native_clip": Path(native.__file__)}
    if any(digest(modules[key]) != ref["implementation_sha256"][key] for key in ("context", "source_context")):
        raise ValueError("reference source context or execution drift")
    output = args.output.resolve()
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation"))):
        raise FileExistsError("new immutable output and sidecars required")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != ref["source_sha256"]:
        raise ValueError("source receipt drift")
    with h5py.File(dataset, "r", swmr=True) as handle:
        clips = {row["seed"]: native.load_native_clip(handle, row) for row in report["rows"]}
    protocol = {"root_seeds": sorted(ref["root_seeds"]), "reference_sha256": digest(args.reference),
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": source_sha,
        "raw_torque_alphas": ALPHAS, "methods": METHODS, "physical_labels_used_for_selection": False,
        "prediction_paths": "stored normalized endpoint vs direct raw normalization vs atanh parameter chart",
        "model_cost_atol": 2e-6, "prediction_path_max_abs_atol": 2e-6, "raw_action_atol": 1e-6,
        "native_physical_state_atol": 2e-7, "root_identity": "sealed_observation_exact_state",
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "data_verification": "Prior signed source SHA receipt reused; no new full archive SHA claimed.",
        "scope": "Known-root action intervention only, all branches retained, no selection, efficacy or P1 admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / f"{key}.py")
    print("All32 original roots, all4 fixed branches, all5 fixed raw interpolation levels. Checking action chart, source transition and exact original successors; host CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))
    low, high = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    rows = []
    try:
        for original in sorted(report["rows"], key=lambda row: row["seed"]):
            seed, clip = original["seed"], clips[original["seed"]]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            initial = original["methods"]["zero"]["execution_identity"]["initial"]
            if any(context[key] != original[key] for key in ("source_episode", "source_start_row")) or context["root_rgb_sha256"] != initial["reset_rgb_sha256"] or context["goal_rgb_sha256"] != initial["goal_rgb_sha256"]:
                raise RuntimeError("original root/goal identity differs")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], clip["history"], device)
            goal_latent = p0._goal_latent(model, goal)
            emb = p0._latent_context(model, pixels, history)
            compilation = original["methods"]["dispatch"]["first_receipt"]
            indices = {"dispatch": compilation["dispatch"]["selected_program_index"],
                "same_bank_greedy": compilation["dispatch"]["same_bank_greedy_index"], "zero": 0}

            def predict(normalized):
                token = torch.as_tensor(normalized.reshape(1, 1, 10), device=device, dtype=history.dtype)
                with torch.no_grad():
                    return model.predict(emb, model.action_encoder(torch.cat((history[:, 1:], token), dim=1)))[0, -1]

            methods, gates = {}, {}
            for name in METHODS:
                recorded = original["methods"][name]
                target = np.asarray(recorded["first_action_witness_raw"], np.float32)
                cells = []
                for alpha in ALPHAS:
                    raw = interpolate_packet(clip["next_raw_packet"], target, alpha)
                    normalized_raw = (raw.reshape(1, 10) - mean) / std
                    normalized = np.asarray(recorded["first_action_witness_normalized"], np.float32) if alpha == 1 else normalized_raw
                    predicted = predict(normalized)
                    token = torch.as_tensor(normalized.reshape(1, 1, 10), device=device, dtype=history.dtype)
                    chart = (2 * token - (upper + lower)) / (upper - lower)
                    params = torch.atanh(chart.clamp(-1 + torch.finfo(token.dtype).eps, 1 - torch.finfo(token.dtype).eps))
                    with torch.no_grad():
                        chart_prediction = p0._terminal_batch(model, pixels, goal, history, params, action_lower=lower, action_upper=upper)[0]
                        direct_prediction = predict(normalized_raw)
                    root, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    p0._verify_execution_identity(root, goal_rgb, context["root_frame"], context["goal"])
                    advanced = execution.advance_executed_context(env, context=context, raw_action_block=raw, action_block=5, verified_root_frame=root)
                    physical = env.unwrapped.env.physics
                    native_error = max(float(np.max(np.abs(np.asarray(physical.data.qpos) - clip["next_qpos"]))),
                        float(np.max(np.abs(np.asarray(physical.data.qvel) - clip["next_qvel"])))) if alpha == 0 else None
                    next_pixels, _, next_history = p0._prepare(processor, advanced["frames"], advanced["root_frame"], advanced["goal"], np.asarray(advanced["history_action_blocks"], np.float32), device)
                    observed = p0._latent_context(model, next_pixels, next_history)[0, -1]
                    cost = lambda value: float((value - goal_latent).square().mean())
                    observed_cost = cost(observed)
                    bridge = p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)
                    cell = {"alpha": alpha, "raw_packet": raw.tolist(), "normalized_packet": normalized.tolist(),
                        "predicted_cost": cost(predicted), "observed_cost": observed_cost,
                        "latent_l2_error": float((predicted - observed).norm()),
                        "predicted": predicted.cpu().tolist(), "observed": observed.cpu().tolist(),
                        "successor_rgb_sha256": advanced["root_rgb_sha256"],
                        "parameter_path_max_abs_error": float((chart_prediction - predicted).abs().max()),
                        "renormalization_path_max_abs_error": float((direct_prediction - predicted).abs().max()),
                        "raw_bridge_max_error": float(np.max(np.abs(bridge - raw))), "native_state_max_error": native_error}
                    label = f"{name}_{alpha}"
                    gates[label + "_action_bridge"] = cell["raw_bridge_max_error"] <= protocol["raw_action_atol"]
                    gates[label + "_chart_equivalence"] = max(cell["parameter_path_max_abs_error"], cell["renormalization_path_max_abs_error"]) <= protocol["prediction_path_max_abs_atol"]
                    gates[label + "_actions_within_box"] = bool(np.all(raw >= env.action_space.low - 1e-6) and np.all(raw <= env.action_space.high + 1e-6))
                    gates[label + "_finite"] = bool(torch.isfinite(predicted).all() and torch.isfinite(observed).all())
                    if alpha == 0:
                        gates[label + "_native_physics"] = native_error <= protocol["native_physical_state_atol"]
                    if alpha == 1:
                        gates[label + "_original_successor_exact"] = advanced["root_rgb_sha256"] == recorded["observed_successor_context"]["root_rgb_sha256"] and advanced["advanced_action_sha256"] == recorded["observed_successor_context"]["advanced_action_sha256"] and observed_cost == recorded["observed_first_chunk"]["candidate_goal_cost"]
                        if name in indices:
                            old_cost = compilation["reconfiguration_compilation"]["proposal_rows"][indices[name]]["immediate_goal_cost"]
                            cell["recorded_predicted_cost_abs_error"] = abs(cell["predicted_cost"] - old_cost)
                            gates[label + "_original_prediction_replayed"] = cell["recorded_predicted_cost_abs_error"] <= protocol["model_cost_atol"]
                    cells.append(cell)
                methods[name] = cells
            # Every branch starts at the exact same official action, not four different anchors.
            gates["all_native_anchor_observations_exact"] = len({methods[name][0]["successor_rgb_sha256"] for name in METHODS}) == 1
            row = {"seed": seed, "source_episode": original["source_episode"], "source_start_row": original["source_start_row"],
                "model_eligible": original["methods"]["dispatch"]["first_receipt"]["dispatch"]["reconfiguration_witness_found"],
                "methods": methods, "gates": gates}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "gates_pass": all(gates.values()), "failed_gates": [key for key, value in gates.items() if not value]}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("interpolation integrity failed; retained partial evidence, no root dropped")
    finally:
        env.close()
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_reference_roots_retained": [row["seed"] for row in rows] == sorted(ref["root_seeds"]),
        "all_replay_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_packet_interpolation_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "gates": gates, "aggregate": summarize(rows), "rows": rows,
        "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": result["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
