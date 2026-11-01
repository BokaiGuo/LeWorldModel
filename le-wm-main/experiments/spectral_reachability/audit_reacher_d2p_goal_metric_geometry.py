"""Host-CUDA audit of exact goal rendering and frozen RGB/qpos metric geometry.

Fixed diagnostic states only; no action selection, readout, or model update.
Evaluator qpos labels never become inputs to a deployed planner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0

RADII = (.05, .1, .2, .4, .8)
DIRECTIONS = tuple((float(np.cos(a)), float(np.sin(a))) for a in np.arange(8) * np.pi / 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = json.loads(args.reference.read_text())
    if report["status"] != "completed" or not all(report["gates"].values()):
        raise ValueError("complete physical reference required")
    if digest(args.policy / "weights.pt") != report["inputs"]["weights_sha256"] or digest(args.adapter_artifact) != report["inputs"]["adapter_sha256"]:
        raise ValueError("reference frozen artifacts differ")
    protocol = {"root_seeds": [row["seed"] for row in report["rows"]], "radii": RADII, "directions": DIRECTIONS,
        "reference_sha256": digest(args.reference), "weights_sha256": digest(args.policy / "weights.pt"),
        "scope": "Known diagnostic roots, no efficacy selection, no training or metric fitting."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    records = args.output.with_suffix(".roots")
    records.mkdir(exist_ok=False)
    print("Auditing exact qpos-goal RGB identity and fixed angular-ring geometry; this is not efficacy.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    rows = []
    try:
        for source in report["rows"]:
            context = shared._source_state_context(env, dataset=dataset, seed=source["seed"])
            identity = source["zero_prefix"]["execution_identity"]["initial"]
            if context["root_rgb_sha256"] != identity["reset_rgb_sha256"] or context["goal_rgb_sha256"] != identity["goal_rgb_sha256"]:
                raise RuntimeError("reference observed root/goal identity differs")
            raw = env.unwrapped
            physics = raw.env.physics
            target = np.asarray(raw.env.task.target_qpos).copy()

            def measured_cost(image):
                prepared = processor._prepare_info({"pixels": image[None, None], "goal": context["goal"][None, None],
                    "action": np.zeros((1, 1, mean.size), np.float32)})
                latent = p0._goal_latent(model, prepared["pixels"].to(device))
                goal = p0._goal_latent(model, prepared["goal"].to(device))
                return float((latent - goal).square().mean())

            physics.data.qpos[:] = target
            physics.data.qvel[:] = 0
            physics.forward()
            goal_frame = np.asarray(raw.render(width=224, height=224), np.uint8).copy()
            self_pixel_error = int(np.abs(goal_frame.astype(np.int16) - context["goal"].astype(np.int16)).max())
            self_cost = measured_cost(goal_frame)
            grid = []
            for radius in RADII:
                for direction_index, direction in enumerate(DIRECTIONS):
                    qpos = (target + radius * np.asarray(direction) + np.pi) % (2 * np.pi) - np.pi
                    physics.data.qpos[:] = qpos
                    physics.data.qvel[:] = 0
                    physics.forward()
                    image = np.asarray(raw.render(width=224, height=224), np.uint8).copy()
                    grid.append({"radius": radius, "direction_index": direction_index,
                        "qpos_distance": p0._qpos_distance(qpos, target), "observed_encoder_goal_cost": measured_cost(image)})
            nonmonotonic = sum(
                outer["observed_encoder_goal_cost"] < inner["observed_encoder_goal_cost"]
                for inner in grid for outer in grid
                if inner["direction_index"] == outer["direction_index"] and outer["radius"] == inner["radius"] * 2)
            execution.restore_execution_snapshot(env, seed=source["seed"], snapshot=context["execution_snapshot"])
            execution.assert_exact_execution_state(context["execution_snapshot"], execution.capture_execution_snapshot(env))
            row = {"seed": source["seed"], "goal_rgb_matches_evaluator_target_exactly": self_pixel_error == 0,
                "self_goal_rgb_max_error": self_pixel_error, "self_goal_encoder_cost": self_cost,
                "radial_nonmonotonic_adjacent_pairs": int(nonmonotonic),
                "distance_encoder_spearman": float(spearmanr([g["radius"] for g in grid], [g["observed_encoder_goal_cost"] for g in grid]).statistic),
                "grid": grid, "exact_execution_state_restored": True}
            with (records / f"root_{source['seed']}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "grid"}), flush=True)
    finally:
        env.close()
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_reference_roots_audited": len(rows) == len(report["rows"]),
        "goal_render_matches_physical_target": all(row["goal_rgb_matches_evaluator_target_exactly"] for row in rows),
        "exact_goal_zero_encoder_distance": all(row["self_goal_encoder_cost"] == 0 for row in rows)}
    result = {"schema": "reacher_d2p_goal_metric_geometry.v1", "status": "completed", "protocol": protocol,
        "cuda_host": host, "gates": gates, "rows": rows,
        "source_dataset_receipt_sha256": source_sha,
        "data_verification": "Source receipt reused, every root/goal RGB checked against prior SHA-bound physical reference; full 93GB dataset SHA not recomputed.",
        "aggregate": {"roots_with_radial_order_reversal": sum(row["radial_nonmonotonic_adjacent_pairs"] > 0 for row in rows),
            "radial_order_reversal_count": sum(row["radial_nonmonotonic_adjacent_pairs"] for row in rows)},
        "inputs": {"auditor_sha256": digest(Path(__file__)), "context_module_sha256": digest(Path(execution.__file__))},
        "claim_boundary": "Goal-interface and frozen-observation geometry diagnostic. Metric reversals are not prediction errors and not D2P efficacy/theory verdicts."}
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": result["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
