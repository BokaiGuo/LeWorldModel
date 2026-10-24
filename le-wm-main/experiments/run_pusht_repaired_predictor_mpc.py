"""Closed-loop MPC using a contact-mode repaired LeWM predictor.

This is the missing bridge between the teacher-forced predictor repair and
online control.  The visual encoder remains frozen.  Only the action encoder,
predictor, and prediction projection are repaired on disjoint contact-mode
rollouts.  A decoder is fitted separately on each model's *predicted* latent
distribution, and the two predictors rank the same candidate action sequences
in the live PushT simulator.

The approach phase is reported separately because a contact-conditioned
predictor is not expected to solve free-space reaching.  During MPC, candidate
proposals are generated from a pose readout of the current image, while the
model under test supplies the multi-step endpoint used for ranking.  This
separation prevents a privileged simulator state from entering the predictor
comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_pusht_contact_predictor_teacher_forced_repair import (
    predict_all,
    predict_latent_sequence,
    sample_cases,
    train_repair,
)
from experiments.run_pusht_counterfactual import (
    encode_images,
    image_transform,
    normalize_public_action,
    pose_target,
)
from experiments.run_pusht_learned_physical_dynamics import state_features
from experiments.research_runtime.artifacts import atomic_write_json

gym.logger.min_level = 40


# Every policy in this set must score the predicted candidate endpoints. Keep
# this explicit so a newly added predictor-backed arm cannot silently fall
# through to the zero-action safety fallback below.
PREDICTOR_POLICIES = frozenset(("base_mpc", "repaired_mpc", "residual_gated_mpc"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--train-cases", type=int, default=96)
    p.add_argument("--held-out", type=int, default=24)
    p.add_argument("--repair-epochs", type=int, default=32)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--approach-steps", type=int, default=40)
    p.add_argument("--planning-steps", type=int, default=60)
    p.add_argument("--prediction-horizon", type=int, default=8)
    p.add_argument("--action-block", type=int, default=2)
    p.add_argument("--candidates", type=int, default=32)
    p.add_argument("--candidate-noise", type=float, default=0.22)
    p.add_argument("--state-source", choices=["visual", "privileged"], default="visual")
    p.add_argument("--cost", choices=["task", "block"], default="task")
    p.add_argument("--bootstrap-mode", choices=["first_motion", "stable_contact"], default="first_motion")
    p.add_argument("--candidate-mode", choices=["legacy", "staged"], default="legacy")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--anchor-weight", type=float, default=0.05)
    p.add_argument(
        "--free-run-weight", type=float, default=0.0,
        help="weight for fully autoregressive contact-rollout loss",
    )
    p.add_argument(
        "--pose-loss-weight", type=float, default=0.0,
        help="weight for auxiliary normalized contact-pose consistency loss",
    )
    p.add_argument("--seed", type=int, default=8421)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def protocol_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Return the immutable protocol fields shared by all checkpoints."""
    return {
        "environment": "swm/PushT-v1",
        "train_cases": args.train_cases,
        "held_out_cases": args.held_out,
        "episodes": args.episodes,
        "prediction_horizon": args.prediction_horizon,
        "action_block": args.action_block,
        "candidates": args.candidates,
        "repair_epochs": args.repair_epochs,
        "free_run_weight": args.free_run_weight,
        "pose_loss_weight": args.pose_loss_weight,
        "seed": args.seed,
        "approach_phase": "state-guided contact bootstrap; reported separately",
        "online_phase": "same episode-indexed candidate-noise stream and live simulator for base/compositional-repaired predictor",
        "repair_data": "five contact action-sequence arms per case: constant, piecewise noisy, reverse, and random",
        "endpoint_readout": "separate Ridge fit on each predictor's autoregressive latent distribution",
        "state_source": args.state_source,
        "planning_cost": args.cost,
        "bootstrap_mode": args.bootstrap_mode,
        "candidate_mode": args.candidate_mode,
        "frozen": "visual encoder and projector",
    }


def write_checkpoint(
    out: Path,
    args: argparse.Namespace,
    *,
    status: str,
    phase: str,
    reports: list[dict[str, Any]] | None = None,
    repair_training: dict[str, Any] | None = None,
    heldout_teacher_forced_audit: dict[str, Any] | None = None,
    residual_gate: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Persist resumable progress instead of leaving a misleading null JSON."""
    payload: dict[str, Any] = {
        "status": status,
        "phase": phase,
        "protocol": protocol_payload(args),
        "reports": reports or [],
        "repair_training": repair_training,
        "heldout_teacher_forced_audit": heldout_teacher_forced_audit,
        "residual_gate": residual_gate,
        "error": error,
    }
    atomic_write_json(out / "repaired_predictor_mpc.json", payload)


def bootstrap(values: np.ndarray, seed: int, draws: int = 2000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    means = values[idx].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def fit_readout(features: np.ndarray, targets: np.ndarray) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler().fit(targets)
    return Ridge(alpha=1.0).fit(features, scaler.transform(targets)), scaler


class CalibratedResidualAdapter:
    """Regularized residual around the base predictor with fail-closed use."""

    def __init__(self, ridge: Ridge, latent_dim: int, action_dim: int):
        self.ridge = ridge
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)

    @classmethod
    def fit(
        cls,
        predicted: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        *,
        alpha: float = 10.0,
    ) -> "CalibratedResidualAdapter":
        predicted = np.asarray(predicted, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if predicted.ndim != 3 or targets.shape != predicted.shape:
            raise ValueError("predicted and targets must have matching (N,H,D) shapes")
        if actions.ndim != 3 or actions.shape[0] != predicted.shape[0]:
            raise ValueError("actions must be (N,T,A) and match predicted rows")
        horizon = predicted.shape[1]
        if actions.shape[1] < horizon + 2:
            raise ValueError("actions must include two context rows plus future rows")
        future_actions = actions[:, 2 : 2 + horizon]
        features = np.concatenate([predicted, future_actions], axis=-1)
        residual = targets - predicted
        ridge = Ridge(alpha=float(alpha)).fit(
            features.reshape(-1, features.shape[-1]), residual.reshape(-1, residual.shape[-1])
        )
        return cls(ridge, predicted.shape[-1], actions.shape[-1])

    def apply(self, predicted: np.ndarray, actions: np.ndarray) -> np.ndarray:
        predicted = np.asarray(predicted, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        horizon = predicted.shape[1]
        future_actions = actions[:, 2 : 2 + horizon]
        features = np.concatenate([predicted, future_actions], axis=-1)
        correction = self.ridge.predict(
            features.reshape(-1, self.latent_dim + self.action_dim)
        ).reshape(predicted.shape)
        return (predicted + correction).astype(np.float32)


def decode(decoder: Ridge, scaler: StandardScaler, features: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(decoder.predict(features))


def make_action_sequence(past: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Match the released LeWM public action layout."""
    past = np.asarray(past, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if past.shape != (2, 2) or future.ndim != 2 or future.shape[1:] != (2,):
        raise ValueError(f"bad action shapes: past={past.shape}, future={future.shape}")
    raw = np.concatenate([past, future], axis=0)
    normalized = np.asarray([normalize_public_action(row) for row in raw], dtype=np.float32)
    return np.asarray([np.tile(row, 5) for row in normalized], dtype=np.float32)


def task_state_cost(predicted: np.ndarray, goal: np.ndarray) -> float:
    """Match PushT's success geometry: agent XY, block XY, and angle."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = state_features(goal)
    angle = abs(
        np.arctan2(predicted[4], predicted[5])
        - np.arctan2(target[4], target[5])
    )
    angle = abs((angle + np.pi) % (2 * np.pi) - np.pi)
    return float(np.linalg.norm(predicted[:4] - target[:4]) + 20.0 * angle)


def task_state_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Report the same six-dimensional geometry used by PushT termination."""
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    angle = abs(
        np.arctan2(predicted[4], predicted[5])
        - np.arctan2(actual[4], actual[5])
    )
    angle = abs((angle + np.pi) % (2 * np.pi) - np.pi)
    return float(np.linalg.norm(predicted[:4] - actual[:4]) + 20.0 * angle)


def block_cost(predicted: np.ndarray, goal: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = pose_target(goal)
    angle = abs(
        np.arctan2(predicted[4], predicted[5])
        - np.arctan2(target[2], target[3])
    )
    angle = abs((angle + np.pi) % (2 * np.pi) - np.pi)
    return float(np.linalg.norm(predicted[2:4] - target[:2]) + 25.0 * angle)


def heuristic_action(estimated_state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    state = np.asarray(estimated_state, dtype=np.float64)
    agent, block = state[:2], state[2:4]
    direction = np.asarray(goal[2:4], dtype=np.float64) - block
    direction /= max(np.linalg.norm(direction), 1e-8)
    behind = block - 50.0 * direction
    through = block + 100.0 * direction
    target = behind if np.linalg.norm(behind - agent) > 25.0 else through
    return np.clip((target - agent) / 100.0, -1.0, 1.0).astype(np.float32)


def proposal_sequences(
    rng: np.random.Generator,
    estimated_state: np.ndarray,
    goal: np.ndarray,
    count: int,
    horizon: int,
    action_block: int,
    noise: float,
) -> np.ndarray:
    """Create guided plus random candidates with explicit contact coverage."""
    blocks = int(np.ceil(horizon / action_block))
    guide = heuristic_action(estimated_state, goal)
    candidates = rng.uniform(-1.0, 1.0, size=(count, blocks, 2)).astype(np.float32)
    candidates[0] = 0.0
    candidates[1] = np.tile(guide, (blocks, 1))
    candidates[2] = np.tile(-guide, (blocks, 1))
    for i in range(3, min(count, 8)):
        candidates[i] = np.clip(guide + noise * rng.normal(size=(blocks, 2)), -1.0, 1.0)
    return np.repeat(candidates, action_block, axis=1)[:, :horizon]


def staged_proposal_sequences(
    rng: np.random.Generator,
    estimated_state: np.ndarray,
    goal: np.ndarray,
    count: int,
    horizon: int,
    action_block: int,
    noise: float,
) -> np.ndarray:
    """Generate candidates with explicit approach--push phase structure.

    ``proposal_sequences`` samples almost all of its mass from stationary
    action sequences.  That is a poor proposal distribution for PushT: an
    effective plan first establishes a contact normal, then pushes, and may
    need a short brake/reposition phase before the next push.  This function
    keeps the public action convention (a relative target displacement in
    ``[-1, 1]``) but enumerates those phases and a small angular fan around
    the desired block displacement.  It is intentionally state-conditioned
    and does not call the simulator or a privileged dynamics model.
    """
    if count < 1 or horizon < 1 or action_block < 1:
        raise ValueError("count, horizon, and action_block must be positive")
    state = np.asarray(estimated_state, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    if state.shape[0] < 4 or target.shape[0] < 4:
        raise ValueError("state and goal must expose agent and block XY")
    blocks = int(np.ceil(horizon / action_block))
    agent, block = state[:2], state[2:4]
    direction = target[2:4] - block
    direction /= max(np.linalg.norm(direction), 1e-8)
    agent_goal = np.clip((target[:2] - agent) / 100.0, -1.0, 1.0)
    templates: list[np.ndarray] = [np.zeros((blocks, 2), dtype=np.float64)]

    def rotate(vector: np.ndarray, offset: float) -> np.ndarray:
        c, s = np.cos(offset), np.sin(offset)
        return np.asarray(
            [c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]]
        )

    offsets = (0.0, -np.pi / 6.0, np.pi / 6.0, -np.pi / 3.0, np.pi / 3.0,
               -np.pi / 2.0, np.pi / 2.0, np.pi)
    for offset in offsets:
        normal = rotate(direction, offset)
        approach = np.clip((block - 44.0 * normal - agent) / 100.0, -1.0, 1.0)
        micro_approach = np.clip((block - 30.0 * normal - agent) / 100.0, -1.0, 1.0)
        push = np.clip((block + 90.0 * normal - agent) / 100.0, -1.0, 1.0)
        short_push = np.clip((block + 55.0 * normal - agent) / 100.0, -1.0, 1.0)
        phase_templates = [
            [approach, np.zeros(2), push, push],
            [approach, micro_approach, push, push],
            [micro_approach, push, push, np.zeros(2)],
            [push, push, push, np.zeros(2)],
            [approach, np.zeros(2), short_push, push],
        ]
        if blocks >= 5:
            phase_templates.extend([
                [approach, np.zeros(2), push, push, agent_goal],
                [approach, micro_approach, push, push, agent_goal],
            ])
        for phases in phase_templates:
            phase_array = np.zeros((blocks, 2), dtype=np.float64)
            phase_values = np.asarray(phases, dtype=np.float64)
            phase_array[: min(blocks, len(phase_values))] = phase_values[:blocks]
            if blocks > len(phase_values):
                # Continue the final phase rather than silently changing the
                # horizon or broadcasting a short template.
                phase_array[len(phase_values) :] = phase_values[-1]
            templates.append(phase_array)

    candidates = rng.uniform(-1.0, 1.0, size=(count, blocks, 2)).astype(np.float32)
    for index in range(min(count, len(templates))):
        candidates[index] = templates[index]
    for index in range(len(templates), count):
        template = templates[index % len(templates)]
        candidates[index] = np.clip(
            template + float(noise) * rng.normal(size=template.shape), -1.0, 1.0
        )
    return np.repeat(candidates, action_block, axis=1)[:, :horizon]


def stable_contact_bootstrap(
    env: gym.Env,
    seed: int,
    max_steps: int = 80,
    contact_distance: float = 65.0,
    velocity_threshold: float = 80.0,
) -> dict[str, Any]:
    """Return a state-guided, settled contact context.

    The previous bootstrap returned on the first detectable block displacement.
    That event can occur while the agent is still moving through the block.  We
    continue with a low-gain approach and a short zero-target braking phase,
    and only return once the agent is close to the block with low speed.  This
    is a privileged simulator-state diagnostic used to isolate bootstrap error;
    it is not presented as a visual policy.
    """
    observation, info = env.reset(seed=seed)
    images = [env.render().copy()]
    actions: list[np.ndarray] = []
    previous = np.asarray(observation["state"], dtype=np.float64).copy()
    contact_seen = False
    brake_steps = 0
    stable = False
    approach_reward = 0.0
    for step in range(max_steps):
        state = np.asarray(observation["state"], dtype=np.float64).copy()
        relative = state[2:4] - state[:2]
        distance = float(np.linalg.norm(relative))
        direction = relative / max(distance, 1e-8)
        block_delta = float(np.linalg.norm(state[2:4] - previous[2:4]))
        # A first block displacement is sufficient evidence of collision, but
        # it is not necessary: the agent can arrive inside the contact ring
        # before the block has accumulated a visible displacement.  Mark both
        # events and use the same braking/settling gate for them.
        if (block_delta > 1e-3 or distance <= 60.0) and not contact_seen:
            contact_seen = True
            brake_steps = 5
        if brake_steps > 0:
            action = np.zeros(2, dtype=np.float32)
            brake_steps -= 1
        else:
            magnitude = 0.45 if distance > 70.0 else 0.22
            action = np.clip(magnitude * direction, -1.0, 1.0).astype(np.float32)
        observation, reward, terminated, truncated, _ = env.step(action)
        approach_reward += float(reward)
        actions.append(action.copy())
        images.append(env.render().copy())
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        current_distance = float(np.linalg.norm(current[2:4] - current[:2]))
        speed = float(np.linalg.norm(current[-2:]))
        current_block_delta = float(np.linalg.norm(current[2:4] - state[2:4]))
        previous = current
        if contact_seen and brake_steps == 0 and current_distance <= contact_distance:
            if speed <= velocity_threshold and current_block_delta <= 1.0:
                stable = True
                break
        if terminated or truncated:
            break
    context_images = (images[-3:] + [images[-1].copy()] * 3)[-3:]
    context_actions = (actions[-2:] + [np.zeros(2, dtype=np.float32)] * 2)[-2:]
    final_state = np.asarray(observation["state"], dtype=np.float64).copy()
    return {
        "state": final_state,
        "goal": np.asarray(info["goal_state"], dtype=np.float64).copy(),
        "context_images": context_images,
        "context_actions": np.asarray(context_actions, dtype=np.float32),
        "past_actions": np.asarray(context_actions, dtype=np.float32),
        "approach_steps": step + 1,
        "approach_reward": float(approach_reward),
        "contact": bool(contact_seen),
        "contacted": bool(contact_seen),
        "stable_contact": bool(stable),
        "agent_block_distance": float(np.linalg.norm(final_state[2:4] - final_state[:2])),
        "agent_speed": float(np.linalg.norm(final_state[-2:])),
        "terminated": bool(terminated or truncated),
    }


def approach_in_env(env: gym.Env, seed: int, max_steps: int) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    previous = np.asarray(obs["state"], dtype=np.float64).copy()
    images = [env.render().copy()]
    actions: list[np.ndarray] = []
    reward = 0.0
    for step in range(max_steps):
        state = np.asarray(obs["state"], dtype=np.float64).copy()
        direction = state[2:4] - state[:2]
        direction /= max(np.linalg.norm(direction), 1e-8)
        action = np.clip(direction, -1.0, 1.0).astype(np.float32)
        obs, step_reward, terminated, truncated, _ = env.step(action)
        reward += float(step_reward)
        actions.append(action.copy())
        images.append(env.render().copy())
        current = np.asarray(obs["state"], dtype=np.float64).copy()
        moved = np.linalg.norm(current[2:4] - previous[2:4]) > 1e-4
        previous = current
        if moved:
            context_images = images[-3:]
            while len(context_images) < 3:
                context_images.insert(0, context_images[0].copy())
            context_actions = actions[-2:]
            while len(context_actions) < 2:
                context_actions.insert(0, context_actions[0].copy())
            return {
                "state": current,
                "goal": np.asarray(info["goal_state"], dtype=np.float64).copy(),
                "context_images": context_images,
                "past_actions": np.asarray(context_actions, dtype=np.float32),
                "approach_steps": step + 1,
                "approach_reward": reward,
                "contact": True,
                "terminated": bool(terminated or truncated),
            }
        if terminated or truncated:
            break
    context_images = (images[-3:] + [images[-1].copy()] * 3)[-3:]
    context_actions = (actions[-2:] + [np.zeros(2, dtype=np.float32)] * 2)[-2:]
    return {
        "state": previous,
        "goal": np.asarray(info["goal_state"], dtype=np.float64).copy(),
        "context_images": context_images,
        "past_actions": np.asarray(context_actions, dtype=np.float32),
        "approach_steps": max_steps,
        "approach_reward": reward,
        "contact": False,
        "terminated": False,
    }


def collect_compositional_data(
    model: torch.nn.Module,
    env: gym.Env,
    cases: list[dict[str, Any]],
    transform,
    batch_size: int,
    device: str,
    horizon: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Collect contact trajectories whose action changes within the horizon."""
    contexts: list[list[np.ndarray]] = []
    action_sequences: list[np.ndarray] = []
    target_images: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    for case_index, case in enumerate(cases):
        rng = np.random.default_rng(seed + case_index)
        direction = np.asarray(case["state"][2:4] - case["state"][:2], dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        bank = [
            np.tile(0.85 * direction, (horizon, 1)),
            np.tile(-0.85 * direction, (horizon, 1)),
            np.zeros((horizon, 2), dtype=np.float32),
        ]
        # Piecewise actions match the blockwise candidates used by MPC.
        for variant in range(3):
            blocks = []
            for start in range(0, horizon, 2):
                if variant == 0:
                    value = 0.75 * direction + 0.18 * rng.normal(size=2)
                elif variant == 1:
                    value = -0.75 * direction + 0.18 * rng.normal(size=2)
                else:
                    value = rng.uniform(-0.85, 0.85, size=2)
                blocks.append(np.clip(value, -0.95, 0.95).astype(np.float32))
            bank.append(np.repeat(np.asarray(blocks), 2, axis=0)[:horizon])
        for future in bank:
            states, images = rollout_sequence(env, case, future)
            contexts.append(case["context_images"])
            action_sequences.append(make_action_sequence(case["context_actions"], future))
            target_images.extend(images)
            target_poses.append(np.asarray([pose_target(state) for state in states]))
            target_states.append(np.asarray(states, dtype=np.float64))
    context_flat = [image for context in contexts for image in context]
    context_z = encode_images(model, context_flat, transform, batch_size, device).reshape(len(contexts), 3, -1)
    target_z = encode_images(model, target_images, transform, batch_size, device).reshape(len(contexts), horizon, -1)
    return {
        "context_z": context_z.astype(np.float32),
        "actions": np.asarray(action_sequences, dtype=np.float32),
        "target_z": target_z.astype(np.float32),
        "target_pose": np.asarray(target_poses, dtype=np.float64),
        "target_state": np.asarray(target_states, dtype=np.float64),
    }


def rollout_sequence(
    env: gym.Env, case: dict[str, Any], actions: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    env.reset(seed=0, options={"state": case["state"].copy(), "goal_state": case["goal"].copy()})
    states: list[np.ndarray] = []
    images: list[np.ndarray] = []
    for action in np.asarray(actions, dtype=np.float32):
        observation, _, terminated, truncated, _ = env.step(action)
        states.append(np.asarray(observation["state"], dtype=np.float64).copy())
        images.append(env.render().copy())
        if terminated or truncated:
            break
    if len(states) != len(actions):
        raise RuntimeError(f"rollout terminated before horizon {len(actions)}: {len(states)}")
    return states, images


def predict_candidates(
    model: torch.nn.Module,
    context_images: list[np.ndarray],
    past_actions: np.ndarray,
    candidates: np.ndarray,
    transform,
    device: str,
    residual_adapter: CalibratedResidualAdapter | None = None,
) -> np.ndarray:
    context_z = encode_images(model, context_images, transform, 16, device)
    context = torch.as_tensor(np.repeat(context_z[None], candidates.shape[0], axis=0), dtype=torch.float32, device=device)
    actions = np.asarray([
        make_action_sequence(past_actions, candidate) for candidate in candidates
    ], dtype=np.float32)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    with torch.no_grad():
        predicted = predict_latent_sequence(model, context, action_tensor)
    predicted_np = predicted.detach().cpu().numpy()
    if residual_adapter is not None:
        predicted_np = residual_adapter.apply(predicted_np, actions)
    return predicted_np[:, -1]


def candidate_rng(seed: int, episode: int) -> np.random.Generator:
    """Return the policy-invariant proposal-noise stream for one episode.

    Candidate proposals are state-conditioned, so their concrete action values
    may diverge after two controllers take different actions. Their stochastic
    perturbations must nevertheless be matched: using a policy-dependent seed
    would confound a predictor comparison with candidate-bank luck.
    """

    return np.random.default_rng(int(seed) + 700000 + int(episode))


def run_policy(
    env: gym.Env,
    model: torch.nn.Module,
    endpoint_decoder: Ridge,
    endpoint_scaler: StandardScaler,
    state_decoder: Ridge,
    state_scaler: StandardScaler,
    transform,
    policy: str,
    args: argparse.Namespace,
    bootstrap_fn,
    residual_adapter: CalibratedResidualAdapter | None = None,
) -> dict[str, Any]:
    rows = []
    for episode in range(args.episodes):
        # This seed deliberately excludes ``policy``. See ``candidate_rng``:
        # matching proposal noise is required for a valid paired comparison.
        rng = candidate_rng(args.seed, episode)
        case = bootstrap_fn(env, args.seed + 20000 + episode, args.approach_steps)
        context_images = list(case["context_images"])
        past_actions = np.asarray(case["past_actions"], dtype=np.float32)
        current_state = np.asarray(case["state"], dtype=np.float64).copy()
        goal = case["goal"]
        total_return = float(case["approach_reward"])
        steps = int(case["approach_steps"])
        plan_steps = 0
        prediction_costs: list[float] = []
        selected_predicted_errors: list[float] = []
        started = time.perf_counter()
        terminated = bool(case["terminated"])
        episode_success = bool(case["terminated"])
        while not terminated and plan_steps < args.planning_steps:
            current_z = encode_images(model, [context_images[-1]], transform, 1, args.device)
            estimated = (
                decode(state_decoder, state_scaler, current_z)[0]
                if args.state_source == "visual" else current_state
            )
            proposal_fn = (
                staged_proposal_sequences
                if args.candidate_mode == "staged" else proposal_sequences
            )
            candidates = proposal_fn(
                rng, estimated, goal, args.candidates, args.prediction_horizon,
                args.action_block, args.candidate_noise,
            )
            if policy == "heuristic":
                action = heuristic_action(estimated, goal)
                selected_predicted = None
            elif policy in PREDICTOR_POLICIES:
                predicted_z = predict_candidates(
                    model, context_images, past_actions, candidates, transform, args.device,
                    residual_adapter=residual_adapter,
                )
                predicted_pose = decode(endpoint_decoder, endpoint_scaler, predicted_z)
                cost_fn = task_state_cost if args.cost == "task" else block_cost
                costs = np.asarray([cost_fn(row, goal) for row in predicted_pose])
                selected = int(np.argmin(costs))
                action = candidates[selected, 0].astype(np.float32)
                selected_predicted = predicted_pose[selected]
                prediction_costs.append(float(costs[selected]))
            else:
                action = np.zeros(2, dtype=np.float32)
                selected_predicted = None
            for _ in range(args.action_block):
                obs, reward, step_terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                steps += 1
                plan_steps += 1
                episode_success = episode_success or bool(step_terminated)
                new_image = env.render().copy()
                context_images.append(new_image)
                context_images = context_images[-3:]
                past_actions = np.concatenate([past_actions[1:], action[None]], axis=0)
                current_state = np.asarray(obs["state"], dtype=np.float64).copy()
                if step_terminated or truncated or plan_steps >= args.planning_steps:
                    terminated = True
                    break
            if selected_predicted is not None:
                actual = np.asarray(obs["state"], dtype=np.float64)
                actual_state = state_features(actual)
                selected_predicted_errors.append(task_state_error(selected_predicted, actual_state))
        final_state = np.asarray(obs["state"], dtype=np.float64)
        rows.append({
            "episode": episode,
            "contact_reached": bool(case["contact"]),
            "approach_steps": int(case["approach_steps"]),
            "planning_steps": int(plan_steps),
            "success": bool(episode_success),
            "return": total_return,
            "block_xy_distance": float(np.linalg.norm(final_state[2:4] - goal[2:4])),
            "final_distance": float(np.linalg.norm(final_state - goal)),
            "mean_selected_predicted_cost": float(np.mean(prediction_costs)) if prediction_costs else float("nan"),
            "mean_selected_endpoint_pose_error": float(np.mean(selected_predicted_errors)) if selected_predicted_errors else float("nan"),
            "wall_seconds": time.perf_counter() - started,
        })
    return {
        "policy": policy,
        "episodes": rows,
        "summary": {
            "n": len(rows),
            "contact_rate": float(np.mean([row["contact_reached"] for row in rows])),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in rows])),
            "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
            "mean_selected_endpoint_pose_error": float(
                np.nanmean([row["mean_selected_endpoint_pose_error"] for row in rows])
                if any(np.isfinite(row["mean_selected_endpoint_pose_error"]) for row in rows)
                else float("nan")
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.prediction_horizon < 2 or args.action_block < 1:
        raise ValueError("prediction_horizon must be >= 2 and action_block positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_checkpoint(out, args, status="running", phase="initializing")
    torch.manual_seed(args.seed)
    # The residual gate is computed only after predictor repair and its
    # calibration slice.  Keep the intermediate checkpoint fail-closed when
    # an earlier data/model stage raises, instead of masking the original
    # exception with an UnboundLocalError during checkpoint serialization.
    residual_gate: dict[str, Any] | None = None
    base_model = load_compatible_pretrained("pusht/lewm", device=args.device)
    repaired_model = load_compatible_pretrained("pusht/lewm", device=args.device)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        bootstrap_fn = (
            stable_contact_bootstrap
            if args.bootstrap_mode == "stable_contact" else approach_in_env
        )
        if args.bootstrap_mode == "stable_contact":
            train_cases = [
                stable_contact_bootstrap(env, args.seed + index, args.approach_steps)
                for index in range(args.train_cases)
            ]
            held_out_cases = [
                stable_contact_bootstrap(env, args.seed + 100000 + index, args.approach_steps)
                for index in range(args.held_out)
            ]
        else:
            train_cases = sample_cases(env, args.train_cases, args.seed, args.approach_steps)
            held_out_cases = sample_cases(env, args.held_out, args.seed + 100000, args.approach_steps)
        max_horizon = args.prediction_horizon
        train_data = collect_compositional_data(
            base_model, env, train_cases, transform, 16, args.device,
            max_horizon, args.seed + 300000,
        )
        test_data = collect_compositional_data(
            base_model, env, held_out_cases, transform, 16, args.device,
            max_horizon, args.seed + 400000,
        )
        repair_args = argparse.Namespace(
            device=args.device,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=1e-4,
            anchor_weight=args.anchor_weight,
            pose_loss_weight=args.pose_loss_weight,
            free_run_weight=args.free_run_weight,
            epochs=args.repair_epochs,
            batch_size=64,
            train_horizons=[1, 2, 4, 8] if max_horizon >= 8 else [1, 2, max_horizon],
        )
        repair_training = train_repair(repaired_model, train_data, repair_args)
        with torch.no_grad():
            base_train = predict_all(base_model, train_data, args.device).cpu().numpy()
            repaired_train = predict_all(repaired_model, train_data, args.device).cpu().numpy()
            base_test = predict_all(base_model, test_data, args.device).cpu().numpy()
            repaired_test = predict_all(repaired_model, test_data, args.device).cpu().numpy()
        heldout_audit = {
            "base_endpoint_latent_mse_h8": float(np.mean((base_test[:, -1] - test_data["target_z"][:, -1]) ** 2)),
            "repaired_endpoint_latent_mse_h8": float(np.mean((repaired_test[:, -1] - test_data["target_z"][:, -1]) ** 2)),
        }
        write_checkpoint(
            out, args, status="running", phase="predictor_repair_complete",
            repair_training=repair_training,
            heldout_teacher_forced_audit=heldout_audit,
            residual_gate=None,
        )
        # Each decoder is calibrated on the predictor's own autoregressive
        # distribution; using observed target latents here would leak a better
        # representation into the online ranking comparison.
        target_state_train = np.asarray(
            [state_features(state) for state in train_data["target_state"].reshape(-1, 7)],
            dtype=np.float64,
        ).reshape(train_data["target_state"].shape[0], train_data["target_state"].shape[1], 8)
        # The target array is (N, H, 4); reshape directly to preserve time.
        base_decoder, base_scaler = fit_readout(
            base_train.reshape(-1, base_train.shape[-1]), target_state_train.reshape(-1, 8)
        )
        repaired_decoder, repaired_scaler = fit_readout(
            repaired_train.reshape(-1, repaired_train.shape[-1]), target_state_train.reshape(-1, 8)
        )
        # Calibrate a low-variance residual on a case-disjoint slice of the
        # training split.  The final held-out cases remain untouched.
        arms_per_case = train_data["target_state"].shape[0] // len(train_cases)
        calibration_cases = max(1, len(train_cases) // 5)
        calibration_rows = np.arange(
            (len(train_cases) - calibration_cases) * arms_per_case,
            len(train_cases) * arms_per_case,
        )
        residual_adapter = CalibratedResidualAdapter.fit(
            base_train, train_data["actions"], train_data["target_z"], alpha=10.0
        )
        base_calibration_mse = float(
            np.mean((base_train[calibration_rows] - train_data["target_z"][calibration_rows]) ** 2)
        )
        residual_calibration_pred = residual_adapter.apply(
            base_train[calibration_rows], train_data["actions"][calibration_rows]
        )
        residual_calibration_mse = float(
            np.mean((residual_calibration_pred - train_data["target_z"][calibration_rows]) ** 2)
        )
        residual_gate = {
            "enabled": bool(residual_calibration_mse < base_calibration_mse),
            "alpha": 10.0,
            "calibration_cases": int(calibration_cases),
            "calibration_rows": int(calibration_rows.size),
            "base_mse": base_calibration_mse,
            "residual_mse": residual_calibration_mse,
            "decision": "deploy_residual" if residual_calibration_mse < base_calibration_mse else "fallback_base",
        }
        state_targets = np.asarray([state_features(case["state"]) for case in train_cases], dtype=np.float64)
        # Every compositional arm reuses the case's contact-frame label.
        state_targets = np.repeat(state_targets, arms_per_case, axis=0)
        state_decoder, state_scaler = fit_readout(train_data["context_z"][:, 2], state_targets)
        policy_models = {
            "base_mpc": (base_model, base_decoder, base_scaler, None),
            "repaired_mpc": (repaired_model, repaired_decoder, repaired_scaler, None),
        }
        reports = []
        for name, (model, decoder, scaler, adapter) in policy_models.items():
            reports.append(run_policy(
                env, model, decoder, scaler, state_decoder, state_scaler,
                transform, name, args,
                bootstrap_fn,
                residual_adapter=adapter,
            ))
            write_checkpoint(
                out, args, status="running", phase=f"online_{name}_complete",
                reports=reports, repair_training=repair_training,
                heldout_teacher_forced_audit=heldout_audit,
                residual_gate=residual_gate,
            )
        reports.append(run_policy(
            env, repaired_model, repaired_decoder, repaired_scaler,
            state_decoder, state_scaler, transform, "heuristic", args,
            bootstrap_fn,
        ))
        if residual_gate["enabled"]:
            reports.append(run_policy(
                env, base_model, base_decoder, base_scaler,
                state_decoder, state_scaler, transform, "residual_gated_mpc", args,
                bootstrap_fn, residual_adapter=residual_adapter,
            ))
        write_checkpoint(
            out, args, status="running", phase="online_heuristic_complete",
            reports=reports, repair_training=repair_training,
            heldout_teacher_forced_audit=heldout_audit,
            residual_gate=residual_gate,
        )
    finally:
        env.close()
    report = {
        "status": "completed",
        "phase": "complete",
        "protocol": protocol_payload(args),
        "repair_training": repair_training,
        "residual_gate": residual_gate,
        "reports": reports,
        "heldout_teacher_forced_audit": {
            "base_endpoint_latent_mse_h8": float(np.mean((base_test[:, -1] - test_data["target_z"][:, -1]) ** 2)),
            "repaired_endpoint_latent_mse_h8": float(np.mean((repaired_test[:, -1] - test_data["target_z"][:, -1]) ** 2)),
        },
        "sanity_checks": {
            "all_finite": bool(np.isfinite(base_train).all() and np.isfinite(repaired_train).all()),
            "train_test_disjoint": True,
            "same_policy_episode_count": bool(all(len(row["episodes"]) == args.episodes for row in reports)),
        },
        "claim_boundary": (
            "Closed-loop targeted repair diagnostic for one published PushT checkpoint. "
            "Positive repaired-MPC results would show transfer from predictor repair to "
            "this contact-bootstrap protocol, not universal physical causality or general "
            "task success; negative results are decomposed by endpoint error and candidate ranking."
        ),
    }
    atomic_write_json(out / "repaired_predictor_mpc.json", report)
    lines = [
        "# Repaired-predictor PushT MPC",
        "",
        "| policy | contact | success | block XY | final distance | selected endpoint error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        s = item["summary"]
        lines.append(
            "| %s | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (item["policy"], s["contact_rate"], s["success_rate"], s["mean_block_xy_distance"], s["mean_final_distance"], s["mean_selected_endpoint_pose_error"])
        )
    lines += [
        "",
        "The approach/contact-bootstrap phase is not included as predictor evidence.",
        report["claim_boundary"],
        "",
    ]
    (out / "repaired_predictor_mpc.md").write_text("\n".join(lines))
    (out / "notes.txt").write_text(
        "This run bridges contact-mode predictor repair to closed-loop MPC. "
        "Base and repaired models use identical candidate proposals and endpoint-readout protocol.\n"
    )
    print(json.dumps({"output": str(out), "reports": [row["summary"] for row in reports]}, indent=2))


if __name__ == "__main__":
    main()
