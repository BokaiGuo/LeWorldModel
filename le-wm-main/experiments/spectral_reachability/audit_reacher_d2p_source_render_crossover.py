"""Host-CUDA paired source-pixel/runtime-render prediction crossover.

Use the exact next official-source action, not a selected planner candidate.
The four source/runtime input and observation combinations separate appearance
shift from physical transition mismatch. This is diagnosis, not P1 admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_causal_innovation as replay
from experiments.spectral_reachability import audit_reacher_d2p_p1_first_chunk_fidelity as fidelity
from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def load_native_clip(handle, source):
    episode, start = int(source["source_episode"]), int(source["source_start_row"])
    lengths, offsets = np.asarray(handle["ep_len"]), np.asarray(handle["ep_offset"])
    if episode < 0 or episode >= len(lengths) or lengths.shape != offsets.shape:
        raise ValueError("invalid source episode metadata")
    offset, end = int(offsets[episode]), int(offsets[episode] + lengths[episode])
    # Native successor at start+20 must be IN this episode, not the next one.
    if start < offset or start + 20 >= end:
        raise ValueError("complete within-episode native successor required; no root removal")
    rows = start + np.arange(5) * 5
    frames = np.asarray(handle["pixels"][rows], dtype=np.uint8)
    actions = np.asarray(handle["action"][start:start + 20], dtype=np.float32)
    if frames.ndim != 4 or frames.shape[0] != 5 or frames.shape[-1] != 3 or actions.shape != (20, 2) or not np.isfinite(actions).all():
        raise ValueError("native RGB/action clip layout differs")
    return {"frames": frames, "history": actions[:15].reshape(3, 10),
        "next_raw_packet": actions[15:], "frame_rows": rows.tolist(),
        "next_qpos": np.asarray(handle["qpos"][start + 20], dtype=np.float64),
        "next_qvel": np.asarray(handle["qvel"][start + 20], dtype=np.float64)}


def pixel_metrics(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("paired same-shape uint8 images required")
    error = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {"mean_absolute_error": float(error.mean()), "max_absolute_error": int(error.max())}


def crossover_metrics(predicted_source, predicted_runtime, observed_source, observed_runtime, goal):
    values = (predicted_source, predicted_runtime, observed_source, observed_runtime, goal)
    if any(value.ndim != 1 or value.shape != goal.shape or value.device != goal.device or not torch.isfinite(value).all() for value in values):
        raise ValueError("finite same-device latent vectors required")
    ps, pr, os, or_, g = [value.detach().to(torch.float64) for value in values]
    input_shift, output_shift = pr - ps, or_ - os
    residual_shift = (pr - or_) - (ps - os)
    cells = {"source_input_source_observation": (ps, os), "source_input_runtime_observation": (ps, or_),
        "runtime_input_source_observation": (pr, os), "runtime_input_runtime_observation": (pr, or_)}
    return {"cells": {name: {"prediction_error_l2": float((p - o).norm()),
        "prediction_goal_cost": float((p - g).square().mean()), "observed_goal_cost": float((o - g).square().mean()),
        "goal_cost_error": float((p - g).square().mean() - (o - g).square().mean())} for name, (p, o) in cells.items()},
        "input_induced_prediction_shift_l2": float(input_shift.norm()),
        "source_vs_runtime_observation_shift_l2": float(output_shift.norm()),
        "prediction_residual_shift_l2": float(residual_shift.norm()),
        "residual_shift_identity_max_error": float((residual_shift - input_shift + output_shift).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.policy, args.output = args.policy.resolve(), args.output.resolve()
    report = json.loads(args.reference.read_text())
    replay.validate_reference(report)
    ref, digest = report["sealed_protocol"], replay.digest
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("checkpoint/action-normalizer drift")
    implementations = {"auditor": Path(__file__), "context": Path(execution.__file__), "p0": Path(p0.__file__),
        "fidelity": Path(fidelity.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename)}
    if any(digest(implementations[key]) != ref["implementation_sha256"][key] for key in ("context", "p0", "source_context")):
        raise ValueError("reference execution changed")
    output = args.output
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"))):
        raise FileExistsError("new immutable outputs required")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != ref["source_sha256"]:
        raise ValueError("source receipt changed")
    # Check all availability before inference, never silently drop edge roots.
    with h5py.File(dataset, "r", swmr=True) as handle:
        clips = {row["seed"]: load_native_clip(handle, row) for row in report["rows"]}
    protocol = {"root_seeds": ref["root_seeds"], "reference_sha256": digest(args.reference),
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": source_sha,
        "raw_action_law": "exact official next five actions from start+15 through start+19, cast to deployed float32",
        "image_crossover": "source/runtime histories crossed with source/runtime actual successor encodings, one shared runtime goal",
        "physical_qpos_qvel_atol": 2e-7, "root_identity": "sealed_observation_exact_state",
        "implementation_sha256": {key: digest(path) for key, path in implementations.items()},
        "scope": "Known-root native-action diagnosis only; not selected-prefix fidelity, efficacy, or P1 admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    print("Crossing source and runtime RGB on the exact same native next action; checking physical transitions separately, host CUDA only.", flush=True)
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
    rows = []
    try:
        for original in sorted(report["rows"], key=lambda row: row["seed"]):
            seed, clip = original["seed"], clips[original["seed"]]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            initial = original["methods"]["zero"]["execution_identity"]["initial"]
            if any(context[key] != original[key] for key in ("source_episode", "source_start_row")) or context["root_rgb_sha256"] != initial["reset_rgb_sha256"] or context["goal_rgb_sha256"] != initial["goal_rgb_sha256"]:
                raise RuntimeError("source root/goal identity changed")
            if not np.array_equal(np.asarray(context["history_action_blocks"]), clip["history"]):
                raise RuntimeError("source/runtime action history differs")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            runtime_pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], clip["history"], device)
            source_pixels, _, source_history = p0._prepare(processor, list(clip["frames"][:3]), clip["frames"][3], context["goal"], clip["history"], device)
            normalized = (clip["next_raw_packet"].reshape(1, 10) - mean) / std
            first = torch.as_tensor(normalized, device=device, dtype=history.dtype)
            ps = fidelity._first_successor_prediction(model, source_pixels, source_history, first)
            pr = fidelity._first_successor_prediction(model, runtime_pixels, history, first)
            root, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
            p0._verify_execution_identity(root, goal_rgb, context["root_frame"], context["goal"])
            advanced = execution.advance_executed_context(env, context=context, raw_action_block=clip["next_raw_packet"], action_block=5, verified_root_frame=root)
            physics = env.unwrapped.env.physics
            physical = {"max_qpos_error": float(np.max(np.abs(np.asarray(physics.data.qpos) - clip["next_qpos"]))),
                "max_qvel_error": float(np.max(np.abs(np.asarray(physics.data.qvel) - clip["next_qvel"])))}
            next_pixels, _, next_history = p0._prepare(processor, advanced["frames"], advanced["root_frame"], advanced["goal"], np.asarray(advanced["history_action_blocks"], np.float32), device)
            observed_runtime = p0._latent_context(model, next_pixels, next_history)[0, -1]
            # Use the same three-frame encoder shape on each side. Only the
            # last encoded frame is used; no learned temporal encoder here.
            native_next = processor._prepare_info({"pixels": clip["frames"][2:5][None], "action": np.asarray(advanced["history_action_blocks"], np.float32)[None]})
            observed_source = p0._latent_context(model, native_next["pixels"].to(device), native_next["action"].to(device))[0, -1]
            goal_latent = p0._goal_latent(model, goal)
            metrics = crossover_metrics(ps, pr, observed_source, observed_runtime, goal_latent)
            bridge = p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)
            row = {"seed": seed, "source_episode": original["source_episode"], "source_start_row": original["source_start_row"],
                "source_frame_rows": clip["frame_rows"], "source_frame_rgb_sha256": [execution.rgb_digest(frame) for frame in clip["frames"]],
                "runtime_root_sha256": context["root_rgb_sha256"], "runtime_successor_sha256": advanced["root_rgb_sha256"],
                "source_runtime_history_pixel_error": pixel_metrics(clip["frames"][1:4], np.stack(context["frames"][1:] + [context["root_frame"]])),
                "source_runtime_successor_pixel_error": pixel_metrics(clip["frames"][4], advanced["root_frame"]),
                "physical": physical, "metrics": metrics,
                "raw_action_bridge_max_error": float(np.max(np.abs(bridge - clip["next_raw_packet"]))),
                "latent_vectors": {"predicted_source": ps.cpu().tolist(), "predicted_runtime": pr.cpu().tolist(),
                    "observed_source": observed_source.cpu().tolist(), "observed_runtime": observed_runtime.cpu().tolist(), "goal": goal_latent.cpu().tolist()}}
            row["gates"] = {"physical_transition_matches_source": max(physical.values()) <= protocol["physical_qpos_qvel_atol"],
                "raw_action_bridge_valid": row["raw_action_bridge_max_error"] <= 1e-6,
                "residual_shift_identity": metrics["residual_shift_identity_max_error"] <= 1e-12}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "gates": row["gates"], "physical": physical,
                "native_error_l2": metrics["cells"]["source_input_source_observation"]["prediction_error_l2"],
                "runtime_error_l2": metrics["cells"]["runtime_input_runtime_observation"]["prediction_error_l2"],
                "residual_shift_l2": metrics["prediction_residual_shift_l2"]}), flush=True)
    finally:
        env.close()
    cell_names = tuple(rows[0]["metrics"]["cells"])
    aggregate = {"root_count": len(rows), "cells": {name: {"mean_prediction_error_l2": float(np.mean([row["metrics"]["cells"][name]["prediction_error_l2"] for row in rows])),
        "mean_absolute_goal_cost_error": float(np.mean([abs(row["metrics"]["cells"][name]["goal_cost_error"]) for row in rows]))} for name in cell_names},
        "mean_input_prediction_shift_l2": float(np.mean([row["metrics"]["input_induced_prediction_shift_l2"] for row in rows])),
        "mean_output_observation_shift_l2": float(np.mean([row["metrics"]["source_vs_runtime_observation_shift_l2"] for row in rows])),
        "mean_residual_shift_l2": float(np.mean([row["metrics"]["prediction_residual_shift_l2"] for row in rows])),
        "max_physical_qpos_error": max(row["physical"]["max_qpos_error"] for row in rows),
        "max_physical_qvel_error": max(row["physical"]["max_qvel_error"] for row in rows)}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_reference_roots_retained": [row["seed"] for row in rows] == sorted(ref["root_seeds"]),
        "all_integrity_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in implementations.items())}
    result = {"schema": "reacher_d2p_source_render_crossover.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "aggregate": aggregate, "rows": rows,
        "data_verification": "Prior source SHA receipt reused; exact reference root/goal/history identities checked and native frame hashes retained. No new full-source SHA claimed.",
        "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
