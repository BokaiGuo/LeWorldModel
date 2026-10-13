"""RGB-only PushT geometry servo used to isolate contact-interface failure.

The controller uses only colour-mask moments from the current and goal RGB
frames for agent/block/goal geometry.  A checkpoint is used only for the
orientation readout.  Simulator state is never read to select an action.
This is a diagnostic controller and a strong visual-interface baseline, not
evidence that the learned dynamics model is correct.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch
import torch.nn.functional as F

try:
    from .evaluate_causal_visual_checkpoint import (
        checkpoint_adapter_architecture,
        checkpoint_utility_mode,
    )
    from .run_causal_visual_pusht_world_model import (
        Normalization,
        PUSHT_RENDER_SIZE,
        task_space_action_cost,
        VisualCausalWorldModel,
        raw_from_normalized,
        resize_image,
        state_vector,
        tensor_images,
    )
except ImportError:
    from evaluate_causal_visual_checkpoint import (  # type: ignore
        checkpoint_adapter_architecture,
        checkpoint_utility_mode,
    )
    from run_causal_visual_pusht_world_model import (  # type: ignore
        Normalization,
        PUSHT_RENDER_SIZE,
        task_space_action_cost,
        VisualCausalWorldModel,
        raw_from_normalized,
        resize_image,
        state_vector,
        tensor_images,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--action-scale", type=float, default=29.5 / PUSHT_RENDER_SIZE)
    p.add_argument("--contact-distance", type=float, default=46.0)
    p.add_argument("--contact-tolerance", type=float, default=20.0)
    p.add_argument("--orientation-threshold", type=float, default=0.28)
    p.add_argument("--torque-gain", type=float, default=0.75)
    p.add_argument("--push-gain", type=float, default=0.75)
    p.add_argument("--terminal-xy", type=float, default=35.0)
    p.add_argument("--learned-torque", action="store_true",
                   help="rank a fixed RGB-derived contact-frame torque fan with the learned successor")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    args = payload["args"]
    model = VisualCausalWorldModel(
        int(args["latent_dim"]),
        str(args["vision_encoder"]),
        bool(args["temporal_context"]),
        checkpoint_utility_mode(payload),
        checkpoint_adapter_architecture(payload),
        bool(args.get("structured_readout", False)),
        bool(args.get("contact_frame_dynamics", False)),
        bool(args.get("mode_conditioned_dynamics", False)),
        bool(args.get("current_only_readout", False)),
        bool(args.get("torque_features", False)),
        bool(args.get("temporal_dynamics", False)),
        bool(args.get("previous_action_dynamics", False)),
        bool(args.get("explicit_angle_effect", False)),
        bool(args.get("explicit_angle_readout", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizer = Normalization(**payload["normalization"])
    mean, std = normalizer.tensors(device)
    model.set_state_normalization(mean, std)
    return model, normalizer


def wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@torch.inference_mode()
def geometry_and_orientation(
    model: VisualCausalWorldModel,
    normalizer: Normalization,
    image: np.ndarray,
    previous: np.ndarray,
    goal_image: np.ndarray,
    image_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = tensor_images(resize_image(image, image_size)[None], device)
    previous_t = tensor_images(resize_image(previous, image_size)[None], device)
    goal = tensor_images(resize_image(goal_image, image_size)[None], device)
    current_features = model.encoder.features(current)
    goal_features = model.encoder.features(goal)
    mean, std = normalizer.tensors(device)
    current_state = model.encode_state(current, previous_t)
    goal_pose = model.encode_goal_pose(goal)
    current_raw = raw_from_normalized(current_state, mean, std)[0].cpu().numpy()
    goal_raw = (goal_pose * std[2:6] + mean[2:6])[0].cpu().numpy()
    # Feature chart: each RGB object contributes mass, mean-x, mean-y, ...
    # The chart coordinates are in [-1, 1]; use the actual RGB render scale.
    agent = current_features[0, 1:3].cpu().numpy()
    block = current_features[0, 11:13].cpu().numpy()
    goal_block = goal_features[0, 11:13].cpu().numpy()
    return agent, block, goal_block, np.asarray([current_raw[4:6], goal_raw[2:4]], dtype=np.float32)


def choose_action(
    agent: np.ndarray,
    block: np.ndarray,
    goal_block: np.ndarray,
    angles: np.ndarray,
    phase: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str, float, float]:
    direction = goal_block - block
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    desired_agent = block - (float(args.contact_distance) / PUSHT_RENDER_SIZE) * direction
    approach = desired_agent - agent
    approach_distance = float(np.linalg.norm(approach) * PUSHT_RENDER_SIZE)
    if phase == "approach" and approach_distance > float(args.contact_tolerance):
        action = np.clip(approach / float(args.action_scale), -1.0, 1.0)
        return action.astype(np.float32), "approach", approach_distance, float(np.linalg.norm(goal_block - block) * PUSHT_RENDER_SIZE)
    if phase == "approach":
        phase = "translate"
    current_angle = math.atan2(float(angles[0, 0]), float(angles[0, 1]))
    goal_angle = math.atan2(float(angles[1, 0]), float(angles[1, 1]))
    angle_error = wrap(goal_angle - current_angle)
    tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    correction = desired_agent - agent
    correction_norm = float(np.linalg.norm(correction))
    correction_direction = correction / max(correction_norm, 1e-6)
    if phase == "torque" or abs(angle_error) > float(args.orientation_threshold):
        push = direction + math.copysign(float(args.torque_gain), angle_error) * tangent
        phase = "torque"
    else:
        push = direction
        phase = "translate"
    if correction_norm > float(args.contact_tolerance) / PUSHT_RENDER_SIZE:
        push = push + 0.35 * correction_direction
    push /= max(float(np.linalg.norm(push)), 1e-6)
    action = np.clip(float(args.push_gain) * push, -1.0, 1.0)
    return action.astype(np.float32), phase, abs(angle_error), float(np.linalg.norm(goal_block - block) * PUSHT_RENDER_SIZE)


@torch.inference_mode()
def learned_torque_action(
    model: VisualCausalWorldModel,
    normalizer: Normalization,
    image: np.ndarray,
    previous: np.ndarray,
    goal_image: np.ndarray,
    direction: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Select a torque candidate using the learned one-step successor only."""
    current = tensor_images(resize_image(image, args.image_size)[None], args.device)
    previous_t = tensor_images(resize_image(previous, args.image_size)[None], args.device)
    goal = tensor_images(resize_image(goal_image, args.image_size)[None], args.device)
    mean, std = normalizer.tensors(args.device)
    state = model.encode_state(current, previous_t)
    goal_pose = model.encode_goal_pose(goal)
    raw_state = raw_from_normalized(state, mean, std)
    raw_goal = goal_pose * std[2:6] + mean[2:6]
    direction_t = torch.as_tensor(direction, dtype=torch.float32, device=args.device)
    direction_t = direction_t / direction_t.norm().clamp_min(1e-6)
    tangent_t = torch.stack([-direction_t[1], direction_t[0]])
    directions = torch.stack([
        direction_t, tangent_t, -tangent_t,
        F.normalize(direction_t + tangent_t, dim=0),
        F.normalize(direction_t - tangent_t, dim=0),
        F.normalize(direction_t + 0.5 * tangent_t, dim=0),
        F.normalize(direction_t - 0.5 * tangent_t, dim=0),
    ])
    magnitudes = torch.tensor([0.22, 0.36, 0.52, 0.72], device=args.device)
    candidates = (directions[:, None, :] * magnitudes[None, :, None]).reshape(-1, 2)
    predicted = model.predict_next(state.expand(len(candidates), -1), candidates)
    predicted_raw = raw_from_normalized(predicted, mean, std)
    cost = task_space_action_cost(
        predicted_raw,
        raw_goal.expand(len(candidates), -1),
        raw_state.expand(len(candidates), -1),
    )
    if model.adapter_architecture == "mode_uncertain":
        _, log_variance = model.predict_distribution(state.expand(len(candidates), -1), candidates)
        cost = cost + float(getattr(args, "uncertainty_risk_weight", 0.25)) * torch.exp(log_variance).mean(dim=-1)
    return candidates[int(torch.argmin(cost))].detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, normalizer = load(Path(args.checkpoint).resolve(), args.device)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=args.steps)
    rows = []
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            goal = np.asarray(info["goal_state"], dtype=np.float32).copy()
            goal_image = np.asarray(info["goal"], dtype=np.uint8).copy()
            previous = env.render().copy()
            phase = "approach"
            trace = []
            for step in range(args.steps):
                image = env.render().copy()
                agent, block, goal_block, angles = geometry_and_orientation(
                    model, normalizer, image, previous, goal_image, args.image_size, args.device,
                )
                action, phase, phase_value, xy_error = choose_action(
                    agent, block, goal_block, angles, phase, args,
                )
                if args.learned_torque and phase == "torque":
                    direction = goal_block - block
                    action = learned_torque_action(
                        model, normalizer, image, previous, goal_image, direction, args,
                    )
                observation, _, terminated, truncated, _ = env.step(action)
                previous = image
                state = np.asarray(observation["state"], dtype=np.float32)
                target = state_vector(goal)
                current = state_vector(state)
                angle_error = abs((math.atan2(float(current[4]), float(current[5])) - math.atan2(float(target[4]), float(target[5])) + math.pi) % (2.0 * math.pi) - math.pi)
                block_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
                trace.append({"step": step, "phase": phase, "action": action.tolist(), "block_xy": block_xy, "angle": angle_error, "phase_value": phase_value, "visual_xy": xy_error})
                if phase == "translate" and xy_error <= float(args.terminal_xy):
                    phase = "torque"
                if block_xy <= 20.0 and angle_error <= math.pi / 9.0:
                    break
                if terminated or truncated:
                    break
            rows.append({
                "episode": episode,
                "strict_success": bool(trace[-1]["block_xy"] <= 20.0 and trace[-1]["angle"] <= math.pi / 9.0),
                "steps": len(trace),
                "block_xy_distance": trace[-1]["block_xy"],
                "angle_error": trace[-1]["angle"],
                "trace": trace,
            })
    finally:
        env.close()
    report = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "runtime_input": "current/previous RGB and goal RGB only",
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "controller": "RGB geometry servo with checkpoint orientation readout",
            "args": vars(args),
        },
        "planning": {
            "episodes": rows,
            "strict_success_count": int(sum(row["strict_success"] for row in rows)),
            "strict_success_rate": float(np.mean([row["strict_success"] for row in rows])),
            "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in rows])),
            "mean_angle_error": float(np.mean([row["angle_error"] for row in rows])),
        },
        "claim_boundary": "RGB-only visual-interface diagnostic; it is not evidence that the learned successor is causal or that a universal world model has been established.",
    }
    (output / "visual_geometry_servo.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **report["planning"]}, indent=2))


if __name__ == "__main__":
    main()
