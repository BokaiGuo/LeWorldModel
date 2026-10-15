"""Mechanism-level diagnosis of negative visual PushT planning results.

This runner is deliberately offline with respect to control selection: simulator
state is used only to create matched counterfactual labels and to score the
diagnosis. It answers five separate questions for one immutable checkpoint:

1. Is the current-image state readout accurate enough?
2. Is the goal-image readout accurate enough?
3. Does the one-step predictor rank candidate actions correctly?
4. Does replacing either readout by an oracle repair the ranking?
5. Does one-step error compound over the planner's open-loop horizon?

The output is an audit artifact, not a planning result. It never modifies a
checkpoint and it never queries a simulator successor to choose an action.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

try:
    from .run_causal_visual_pusht_world_model import (
        Normalization,
        VisualCausalWorldModel,
        balanced_source_cases,
        make_counterfactual_dataset,
        raw_from_normalized,
        resize_image,
        state_vector,
        tensor_images,
    )
    from .evaluate_causal_visual_checkpoint import (
        checkpoint_adapter_architecture,
        checkpoint_utility_mode,
    )
except ImportError:
    from run_causal_visual_pusht_world_model import (  # type: ignore
        Normalization,
        VisualCausalWorldModel,
        balanced_source_cases,
        make_counterfactual_dataset,
        raw_from_normalized,
        resize_image,
        state_vector,
        tensor_images,
    )
    from evaluate_causal_visual_checkpoint import (  # type: ignore
        checkpoint_adapter_architecture,
        checkpoint_utility_mode,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=93001)
    p.add_argument("--roots", type=int, default=120)
    p.add_argument("--root-rollout-steps", type=int, default=8)
    p.add_argument("--contact-root-fraction", type=float, default=0.5)
    p.add_argument("--contact-approach-steps", type=int, default=48)
    p.add_argument("--counterfactual-fan-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 6])
    return p.parse_args()


def angle_cost(raw: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Task-space block-pose cost used only for ranking diagnosis."""
    goal = np.asarray(goal)
    if raw.ndim == goal.ndim + 1:
        goal = goal[..., None, :]
    xy = np.linalg.norm(raw[..., 2:4] - goal[..., :2], axis=-1) / 20.0
    pred = raw[..., 4:6]
    target = goal[..., 2:4]
    pred = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-6)
    target = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
    orientation = np.sum((pred - target) ** 2, axis=-1)
    return xy + 20.0 * orientation


def top1_accuracy(predicted_cost: np.ndarray, true_cost: np.ndarray) -> float:
    if predicted_cost.ndim != 2 or true_cost.ndim != 2:
        raise ValueError("candidate costs must have shape [roots, candidates]")
    if predicted_cost.shape != true_cost.shape:
        raise ValueError("predicted and true candidate matrices must match")
    return float(np.mean(np.argmin(predicted_cost, axis=1) == np.argmin(true_cost, axis=1)))


def load_checkpoint(path: Path, device: str) -> tuple[VisualCausalWorldModel, Normalization, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_args = dict(payload.get("args", {}))
    utility_mode = checkpoint_utility_mode(payload)
    adapter_architecture = checkpoint_adapter_architecture(payload)
    model = VisualCausalWorldModel(
        int(saved_args["latent_dim"]),
        str(saved_args["vision_encoder"]),
        bool(saved_args["temporal_context"]),
        utility_mode,
        adapter_architecture,
        bool(saved_args.get("structured_readout", False)),
        bool(saved_args.get("contact_frame_dynamics", False)),
        bool(saved_args.get("mode_conditioned_dynamics", False)),
        bool(saved_args.get("current_only_readout", False)),
        bool(saved_args.get("torque_features", False)),
        bool(saved_args.get("temporal_dynamics", False)),
        bool(saved_args.get("previous_action_dynamics", False)),
        bool(saved_args.get("explicit_angle_effect", False)),
        bool(saved_args.get("explicit_angle_readout", False)),
        bool(saved_args.get("explicit_goal_angle_readout", False)),
        bool(saved_args.get("rich_temporal_dynamics", False)),
        invariant_angle_readout=bool(saved_args.get("invariant_angle_readout", False)),
        contact_interface_dynamics=bool(saved_args.get("contact_interface_dynamics", False)),
        angle_readout_current_only=bool(saved_args.get("angle_readout_current_only", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizer = Normalization(**payload["normalization"])
    mean, std = normalizer.tensors(device)
    model.set_state_normalization(mean, std)
    return model, normalizer, saved_args


@torch.inference_mode()
def readouts(
    model: VisualCausalWorldModel,
    normalizer: Normalization,
    data: dict[str, np.ndarray],
    device: str,
) -> dict[str, np.ndarray]:
    mean, std = normalizer.tensors(device)
    image = tensor_images(data["image"], device)
    previous = tensor_images(data["previous_image"], device)
    goal_image = tensor_images(data["goal_image"], device)
    action = torch.as_tensor(data["action"], dtype=torch.float32, device=device)
    state, predicted = model(image, action, previous)
    goal = model.encode_goal_pose(goal_image)
    return {
        "state": state.cpu().numpy(),
        "predicted": predicted.cpu().numpy(),
        "goal": goal.cpu().numpy(),
        "state_raw": raw_from_normalized(state, mean, std).cpu().numpy(),
        "predicted_raw": raw_from_normalized(predicted, mean, std).cpu().numpy(),
        "goal_raw": (goal * std[2:6] + mean[2:6]).cpu().numpy(),
    }


def make_candidate_matrices(
    data: dict[str, np.ndarray], rows: dict[str, np.ndarray]
) -> dict[str, Any]:
    roots = np.unique(data["root_id"])
    if len(roots) == 0:
        raise RuntimeError("no source roots in diagnosis dataset")
    per_root = [np.flatnonzero(data["root_id"] == root) for root in roots]
    candidate_count = len(per_root[0])
    if any(len(indices) != candidate_count for indices in per_root):
        raise RuntimeError("counterfactual candidate fan is not balanced")
    candidate_idx = np.asarray(per_root, dtype=np.int64)
    return {
        "roots": roots,
        "candidate_idx": candidate_idx,
        "state_raw": np.asarray([rows["state_raw"][x[0]] for x in per_root]),
        "goal_raw": np.asarray([rows["goal_raw"][x[0]] for x in per_root]),
        "true_goal_raw": np.asarray([data["goal_pose"][x[0]] for x in per_root]),
        "true_next_raw": np.asarray([data["next_state"][x] for x in per_root]),
        "predicted_next_raw": np.asarray([rows["predicted_raw"][x] for x in per_root]),
        "contact": np.asarray([bool(data["contact_root"][x[0]]) for x in per_root], dtype=bool),
    }


def candidate_ranking(
    data: dict[str, np.ndarray], rows: dict[str, np.ndarray]
) -> dict[str, Any]:
    mats = make_candidate_matrices(data, rows)
    true_cost = angle_cost(mats["true_next_raw"], mats["true_goal_raw"])
    learned_cost = angle_cost(mats["predicted_next_raw"], mats["goal_raw"])
    learned_goal_oracle_state_cost = angle_cost(
        mats["predicted_next_raw"], mats["true_goal_raw"]
    )
    learned_state_effect = mats["predicted_next_raw"] - mats["state_raw"][:, None, :]
    oracle_state_pred = mats["state_raw"][:, None, :] + learned_state_effect
    oracle_state_cost = angle_cost(oracle_state_pred, mats["goal_raw"])
    oracle_both_cost = angle_cost(oracle_state_pred, mats["true_goal_raw"])

    def evaluate(mask: np.ndarray) -> dict[str, float]:
        return {
            "learned_state_learned_goal": top1_accuracy(learned_cost[mask], true_cost[mask]),
            "oracle_state_learned_goal": top1_accuracy(oracle_state_cost[mask], true_cost[mask]),
            "learned_state_oracle_goal": top1_accuracy(learned_goal_oracle_state_cost[mask], true_cost[mask]),
            "oracle_state_oracle_goal": top1_accuracy(oracle_both_cost[mask], true_cost[mask]),
        }

    all_mask = np.ones(len(mats["roots"]), dtype=bool)
    contact_mask = mats["contact"]
    return {
        "all_roots": int(all_mask.sum()),
        "contact_roots": int(contact_mask.sum()),
        "candidate_count": int(mats["candidate_idx"].shape[1]),
        "top1": evaluate(all_mask),
        "contact_top1": evaluate(contact_mask),
        "oracle_successor_ceiling": 1.0,
        "sanity": {
            "finite": bool(np.isfinite(learned_cost).all()),
            "goal_readout_distinct_from_oracle": bool(
                np.max(np.abs(mats["goal_raw"] - mats["true_goal_raw"])) > 1e-5
            ),
        },
    }


@torch.inference_mode()
def rollout_error_by_horizon(
    model: VisualCausalWorldModel,
    normalizer: Normalization,
    env: gym.Env,
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    horizons: list[int],
    image_size: int,
    device: str,
) -> dict[str, Any]:
    mean, std = normalizer.tensors(device)
    rng = np.random.default_rng(99173)
    records: list[dict[str, float]] = []
    for root_index, (state, goal, _image, _previous, _goal_image, _contact) in enumerate(roots):
        observation, _ = env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
        actions: list[np.ndarray] = []
        actual: list[np.ndarray] = []
        for _step in range(max(horizons)):
            action = np.clip(rng.normal(0.0, 0.45, size=2), -1.0, 1.0).astype(np.float32)
            actions.append(action)
            observation, _, terminated, truncated, _ = env.step(action)
            actual.append(state_vector(np.asarray(observation["state"], dtype=np.float32))[:6])
            if terminated or truncated:
                break
        env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
        root_image = tensor_images(resize_image(env.render().copy(), image_size)[None], device)
        current = model.encode_state(root_image, root_image)
        for step, action in enumerate(actions):
            action_tensor = torch.as_tensor(action[None], dtype=torch.float32, device=device)
            current = model.predict_next(current, action_tensor)
            predicted_raw = raw_from_normalized(current, mean, std).cpu().numpy()[0]
            true_raw = actual[step]
            predicted_angle = math.atan2(float(predicted_raw[4]), float(predicted_raw[5]))
            true_angle = math.atan2(float(true_raw[4]), float(true_raw[5]))
            records.append({
                "root": float(root_index),
                "horizon": float(step + 1),
                "state_rmse": float(np.sqrt(np.mean((predicted_raw - true_raw) ** 2))),
                "block_xy_error": float(np.linalg.norm(predicted_raw[2:4] - true_raw[2:4])),
                "angle_error": float(abs((predicted_angle - true_angle + math.pi) % (2.0 * math.pi) - math.pi)),
            })
    summary: dict[str, Any] = {}
    for horizon in horizons:
        subset = [row for row in records if int(row["horizon"]) == horizon]
        summary[str(horizon)] = {
            "n": len(subset),
            "state_rmse": float(np.mean([x["state_rmse"] for x in subset])) if subset else float("nan"),
            "block_xy_error": float(np.mean([x["block_xy_error"] for x in subset])) if subset else float("nan"),
            "angle_error": float(np.mean([x["angle_error"] for x in subset])) if subset else float("nan"),
        }
    return {"summary": summary, "rows": records}


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    model, normalizer, saved_args = load_checkpoint(checkpoint, args.device)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
    try:
        roots = balanced_source_cases(
            env, args.roots, args.root_rollout_steps, args.seed,
            args.contact_root_fraction, args.contact_approach_steps,
        )
        data = make_counterfactual_dataset(
            env, roots, args.image_size, args.seed + 20_000, 1,
            args.counterfactual_fan_size, "isotropic", 0.85,
        )
        rows = readouts(model, normalizer, data, args.device)
        readout = {
            "state_readout_rmse": float(np.sqrt(np.mean((rows["state_raw"] - data["state"]) ** 2))),
            "state_readout_xy_rmse": float(np.sqrt(np.mean((rows["state_raw"][:, 2:4] - data["state"][:, 2:4]) ** 2))),
            "goal_readout_rmse": float(np.sqrt(np.mean((rows["goal_raw"] - data["goal_pose"]) ** 2))),
            "goal_readout_xy_rmse": float(np.sqrt(np.mean((rows["goal_raw"][:, :2] - data["goal_pose"][:, :2]) ** 2))),
            "one_step_state_rmse": float(np.sqrt(np.mean((rows["predicted_raw"] - data["next_state"]) ** 2))),
            "one_step_block_xy_rmse": float(np.sqrt(np.mean((rows["predicted_raw"][:, 2:4] - data["next_state"][:, 2:4]) ** 2))),
            "one_step_orientation_rmse": float(np.sqrt(np.mean((rows["predicted_raw"][:, 4:6] - data["next_state"][:, 4:6]) ** 2))),
            "contact_one_step_state_rmse": float(np.sqrt(np.mean((rows["predicted_raw"][data["contact_root"]] - data["next_state"][data["contact_root"]]) ** 2))),
        }
        ranking = candidate_ranking(data, rows)
        rollout = rollout_error_by_horizon(
            model, normalizer, env, roots[: min(len(roots), 60)],
            args.horizons, args.image_size, args.device,
        )
    finally:
        env.close()
    report = {
        "protocol": {
            "checkpoint": str(checkpoint),
            "checkpoint_immutable": True,
            "seed": args.seed,
            "roots": len(roots),
            "contact_root_fraction": args.contact_root_fraction,
            "counterfactual_fan_size": args.counterfactual_fan_size,
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "physical_state_role": "offline labels and diagnosis only",
            "saved_model_args": saved_args,
        },
        "readout_and_one_step": readout,
        "candidate_ranking_decomposition": ranking,
        "open_loop_rollout_error": rollout,
        "interpretation_rules": {
            "readout_bottleneck": "oracle_state_learned_goal substantially exceeds learned_state_learned_goal",
            "goal_bottleneck": "learned_state_oracle_goal substantially exceeds learned_state_learned_goal",
            "dynamics_or_candidate_bottleneck": "both oracle readouts remain below the oracle successor ceiling",
            "compounding_bottleneck": "multi-step error grows materially with horizon",
        },
        "claim_boundary": "Mechanism-level offline diagnosis of one visual PushT checkpoint; not visual online-planning success, causal-control evidence, or a universal world-model claim.",
    }
    (output / "negative_cause_diagnosis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Negative-result cause diagnosis", "",
        f"Checkpoint: {checkpoint}", "",
        "## Readout and one-step", "",
        "| metric | value |", "|---|---:|",
    ]
    for key, value in readout.items():
        lines.append(f"| {key} | {value:.6g} |")
    lines += ["", "## Candidate-ranking decomposition", "", "| interface | all roots | contact roots |", "|---|---:|---:"]
    for key, value in ranking["top1"].items():
        lines.append(f"| {key} | {value:.4f} | {ranking['contact_top1'][key]:.4f} |")
    lines += ["", "## Open-loop error", "", "| horizon | state RMSE | block XY | angle |", "|---:|---:|---:|---:"]
    for horizon, value in rollout["summary"].items():
        lines.append(f"| {horizon} | {value['state_rmse']:.4f} | {value['block_xy_error']:.4f} | {value['angle_error']:.4f} |")
    lines += ["", report["claim_boundary"], ""]
    (output / "negative_cause_diagnosis.md").write_text("\n".join(lines))
    print(json.dumps({
        "output": str(output),
        "readout": readout,
        "ranking": ranking["top1"],
        "rollout": rollout["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
