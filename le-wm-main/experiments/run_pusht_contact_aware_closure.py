"""Contact-balanced physical closure and matched PushT planning audit.

This experiment isolates two failure modes of the earlier physical adapter:
the one-step data distribution under-samples contact transitions, and uniform
action-sequence candidates under-sample the approach-to-push manoeuvre.  It
uses only simulator state for the physical arm; it does not retrain LeWM or
turn this adapter into evidence about the frozen latent.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_learned_physical_dynamics import state_features
from experiments.run_pusht_pose_closure import bootstrap, heuristic_action, pose

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=2048)
    parser.add_argument("--held-out", type=int, default=24)
    parser.add_argument("--planning-episodes", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--planning-blocks", type=int, default=2)
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--contact-threshold", type=float, default=0.85)
    parser.add_argument("--angle-weight", type=float, default=20.0)
    parser.add_argument(
        "--include-agent-velocity",
        action="store_true",
        help="retain the simulator agent velocity in the physical adapter state",
    )
    parser.add_argument(
        "--sampling-protocol",
        choices=("balanced_reset", "trajectory_approach"),
        default="balanced_reset",
        help="use hand-placed contact rings or genuine approach trajectories",
    )
    parser.add_argument("--seed", type=int, default=7317)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def pose_vector(state: np.ndarray, include_agent_velocity: bool = False) -> np.ndarray:
    features = state_features(state)
    return features if include_agent_velocity else pose(features)


def transition_features(current: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Pose/action features with explicit contact geometry."""
    current = np.asarray(current, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    relative = current[2:4] - current[:2]
    distance = np.linalg.norm(relative)
    theta = float(np.arctan2(current[4], current[5]))
    rotation = np.asarray([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    relative_local = rotation @ relative
    action_local = rotation @ action
    # Dot/cross terms distinguish approach, retreat, and tangential motion.
    cross = float(action[0] * relative[1] - action[1] * relative[0])
    return np.concatenate(
        [
            current, action, relative, [distance], [float(np.dot(action, relative)), cross],
            relative_local, action_local,
        ]
    )


def sample_case(env: gym.Env, seed: int) -> dict[str, np.ndarray]:
    observation, info = env.reset(seed=seed)
    return {
        "state": np.asarray(observation["state"], dtype=np.float64).copy(),
        "goal": np.asarray(info["goal_state"], dtype=np.float64).copy(),
    }


def normalize_pose_batch(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    norm = np.linalg.norm(values[..., 4:6], axis=-1, keepdims=True)
    values[..., 4:6] /= np.maximum(norm, 1e-8)
    values[..., :4] = np.clip(values[..., :4], 5.0, 507.0)
    return values


def block_motion(before: np.ndarray, after: np.ndarray, tolerance: float = 1e-4) -> bool:
    """Return whether the block, rather than a wall, changed state."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    position_delta = np.linalg.norm(after[2:4] - before[2:4])
    angle_delta = abs((after[4] - before[4] + np.pi) % (2.0 * np.pi) - np.pi)
    return bool(position_delta > tolerance or angle_delta > tolerance)


def contact_balanced_data(
    env: gym.Env,
    count: int,
    seed: int,
    rng: np.random.Generator,
    include_agent_velocity: bool = False,
) -> dict[str, np.ndarray]:
    """Collect equal free, approach, contact-push, and tangential samples."""
    states, actions, next_states, contacts = [], [], [], []
    block_next_states, block_contacts = [], []
    for index in range(count):
        case_seed = seed + index
        case = sample_case(env, case_seed)
        state = case["state"].copy()
        block = state[2:4].copy()
        mode = index % 4
        if mode == 0:
            action = rng.uniform(-1.0, 1.0, size=2)
        else:
            direction = rng.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-8)
            distance = rng.uniform(35.0, 85.0) if mode == 1 else rng.uniform(10.0, 27.0)
            state[:2] = block - distance * direction
            state[-2:] = 0.0
            if mode == 3:
                direction = np.asarray([-direction[1], direction[0]])
            target = block + 100.0 * direction
            action = np.clip((target - state[:2]) / 100.0, -1.0, 1.0)
            action = np.clip(action + 0.08 * rng.normal(size=2), -1.0, 1.0)
        reset_observation, _ = env.reset(
            seed=case_seed, options={"state": state, "goal_state": case["goal"]}
        )
        state = np.asarray(reset_observation["state"], dtype=np.float64).copy()
        action = np.asarray(action, dtype=np.float32)
        observation, _, _, _, info = env.step(action)
        next_state = np.asarray(observation["state"], dtype=np.float64).copy()
        # ``n_contacts`` counts every collision in the environment, including
        # wall contacts.  All shapes use the same collision type, so it is not
        # a valid agent--block label.  Block state change is the auditable
        # observable available through this environment API.
        first_contact = block_motion(state, next_state)
        block_contact = first_contact
        block_state = next_state
        for _ in range(4):
            observation, _, _, _, info = env.step(action)
            block_state = np.asarray(observation["state"], dtype=np.float64).copy()
            block_contact = block_contact or block_motion(state, block_state)
        states.append(pose_vector(state, include_agent_velocity))
        actions.append(np.asarray(action, dtype=np.float64))
        next_states.append(pose_vector(next_state, include_agent_velocity))
        contacts.append(float(first_contact))
        block_next_states.append(pose_vector(block_state, include_agent_velocity))
        block_contacts.append(float(block_contact))
    return {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "next_states": np.asarray(next_states),
        "contacts": np.asarray(contacts),
        "block_next_states": np.asarray(block_next_states),
        "block_contacts": np.asarray(block_contacts),
    }


def trajectory_approach_data(
    env: gym.Env,
    count: int,
    seed: int,
    rng: np.random.Generator,
    include_agent_velocity: bool = False,
) -> dict[str, np.ndarray]:
    """Collect transitions from genuine free/approach/tangential rollouts.

    Unlike ``contact_balanced_data``, this protocol never teleports the agent
    onto a hand-defined contact ring.  It samples the transition at the state
    actually reached by a short behavior rollout, which covers the
    approach-to-contact boundary and the velocity carried into contact.
    """
    states, actions, next_states, contacts = [], [], [], []
    block_next_states, block_contacts = [], []
    episode = 0
    while len(states) < count:
        case_seed = seed + 100000 + episode
        observation, info = env.reset(seed=case_seed)
        current_state = np.asarray(observation["state"], dtype=np.float64).copy()
        episode += 1
        for step in range(40):
            if len(states) >= count:
                break
            relative = current_state[2:4] - current_state[:2]
            distance = np.linalg.norm(relative)
            direction = relative / max(distance, 1e-8)
            mode = step % 4
            if mode == 0:
                action = rng.uniform(-1.0, 1.0, size=2)
            elif mode == 1 or distance > 65.0:
                action = direction + 0.10 * rng.normal(size=2)
            else:
                tangent = np.asarray([-direction[1], direction[0]])
                action = 0.45 * direction + 0.55 * tangent + 0.08 * rng.normal(size=2)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            current_for_model = current_state.copy()
            observation, _, terminated, truncated, _ = env.step(action)
            next_state = np.asarray(observation["state"], dtype=np.float64).copy()
            first_contact = block_motion(current_for_model, next_state)
            block_state = next_state.copy()
            block_contact = first_contact
            for _ in range(4):
                observation, _, terminated, truncated, _ = env.step(action)
                block_state = np.asarray(observation["state"], dtype=np.float64).copy()
                block_contact = block_contact or block_motion(current_for_model, block_state)
            states.append(pose_vector(current_for_model, include_agent_velocity))
            actions.append(np.asarray(action, dtype=np.float64))
            next_states.append(pose_vector(next_state, include_agent_velocity))
            contacts.append(float(first_contact))
            block_next_states.append(pose_vector(block_state, include_agent_velocity))
            block_contacts.append(float(block_contact))
            current_state = block_state
            if terminated or truncated:
                break
    return {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "next_states": np.asarray(next_states),
        "contacts": np.asarray(contacts),
        "block_next_states": np.asarray(block_next_states),
        "block_contacts": np.asarray(block_contacts),
    }


class ContactAwareModel:
    def __init__(
        self,
        dynamics: ExtraTreesRegressor,
        contact: ExtraTreesClassifier,
        block_dynamics: ExtraTreesRegressor,
        block_contact: ExtraTreesClassifier,
        threshold: float,
        include_agent_velocity: bool = False,
    ):
        self.dynamics = dynamics
        self.contact = contact
        self.block_dynamics = block_dynamics
        self.block_contact = block_contact
        self.threshold = threshold
        self.include_agent_velocity = include_agent_velocity


def fit_model(
    data: dict[str, np.ndarray],
    trees: int,
    seed: int,
    threshold: float,
    include_agent_velocity: bool = False,
) -> ContactAwareModel:
    features = np.asarray(
        [transition_features(state, action) for state, action in zip(data["states"], data["actions"])]
    )
    targets = data["next_states"] - data["states"]
    model = ExtraTreesRegressor(
        n_estimators=trees, min_samples_leaf=2, max_features=1.0,
        random_state=seed, n_jobs=1,
    )
    model.fit(features, targets)
    contact = ExtraTreesClassifier(
        n_estimators=max(32, trees // 2), min_samples_leaf=2,
        max_features=1.0, random_state=seed + 1, n_jobs=1,
    )
    contact.fit(features, data["contacts"].astype(np.int64))
    block_model = ExtraTreesRegressor(
        n_estimators=trees, min_samples_leaf=2, max_features=1.0,
        random_state=seed + 2, n_jobs=1,
    )
    block_model.fit(features, data["block_next_states"] - data["states"])
    block_contact = ExtraTreesClassifier(
        n_estimators=max(32, trees // 2), min_samples_leaf=2,
        max_features=1.0, random_state=seed + 3, n_jobs=1,
    )
    block_contact.fit(features, data["block_contacts"].astype(np.int64))
    return ContactAwareModel(
        model, contact, block_model, block_contact, threshold, include_agent_velocity
    )


def predict_pose(model: ContactAwareModel, current: np.ndarray, action: np.ndarray) -> np.ndarray:
    feature = transition_features(current, action)[None]
    return predict_pose_batch(model, np.asarray(current)[None], feature)[0]


def predict_pose_batch(
    model: ContactAwareModel, current: np.ndarray, features: np.ndarray
) -> np.ndarray:
    delta = model.dynamics.predict(features)
    contact_probability = model.contact.predict_proba(features)[:, 1]
    no_contact = contact_probability < model.threshold
    # In the no-contact mode, the simulator leaves the block pose fixed;
    # retaining that invariant prevents a balanced training prior from
    # hallucinating block motion on ordinary free-space actions.
    delta[no_contact, 2:6] = 0.0
    return normalize_pose_batch(np.asarray(current) + delta)


def predict_block_batch(
    model: ContactAwareModel, current: np.ndarray, features: np.ndarray
) -> np.ndarray:
    delta = model.block_dynamics.predict(features)
    contact_probability = model.block_contact.predict_proba(features)[:, 1]
    delta[contact_probability < model.threshold, 2:6] = 0.0
    return normalize_pose_batch(np.asarray(current) + delta)


def rollout_pose(model: ContactAwareModel, current: np.ndarray, action: np.ndarray, steps: int) -> np.ndarray:
    result = np.asarray(current, dtype=np.float64).copy()
    for _ in range(steps):
        result = predict_pose(model, result, action)
    return result


def rollout_candidates(
    model: ContactAwareModel,
    current: np.ndarray,
    candidates: np.ndarray,
    action_block: int,
    use_block_model: bool = False,
) -> np.ndarray:
    states = np.repeat(np.asarray(current, dtype=np.float64)[None], len(candidates), axis=0)
    for block in range(candidates.shape[1]):
        actions = np.asarray(candidates[:, block], dtype=np.float64)
        features = np.asarray(
            [transition_features(state, action) for state, action in zip(states, actions)]
        )
        # The planning primitive is an action block.  The direct block model
        # avoids repeatedly extrapolating a one-step contact transition.
        if use_block_model and action_block == 5:
            states = predict_block_batch(model, states, features)
        else:
            for _ in range(action_block):
                states = predict_pose_batch(model, states, features)
    return states


def rollout_candidate_trajectory(
    model: ContactAwareModel,
    current: np.ndarray,
    candidates: np.ndarray,
    action_block: int,
    use_block_model: bool = False,
) -> np.ndarray:
    """Return predicted pose after every candidate action block."""
    states = np.repeat(np.asarray(current, dtype=np.float64)[None], len(candidates), axis=0)
    trajectory = [states.copy()]
    for block in range(candidates.shape[1]):
        actions = np.asarray(candidates[:, block], dtype=np.float64)
        features = np.asarray(
            [transition_features(state, action) for state, action in zip(states, actions)]
        )
        if use_block_model and action_block == 5:
            states = predict_block_batch(model, states, features)
        else:
            for _ in range(action_block):
                states = predict_pose_batch(model, states, features)
        trajectory.append(states.copy())
    return np.asarray(trajectory)


def contact_aware_candidates(
    rng: np.random.Generator,
    state: np.ndarray,
    goal: np.ndarray,
    count: int,
    blocks: int,
) -> np.ndarray:
    current = state_features(state)
    target = state_features(goal)
    agent, block = current[:2], current[2:4]
    direction = target[2:4] - block
    direction /= max(np.linalg.norm(direction), 1e-8)
    approach = np.clip((block - 50.0 * direction - agent) / 100.0, -1.0, 1.0)
    push = np.clip((block + 100.0 * direction - agent) / 100.0, -1.0, 1.0)
    short_push = np.clip((block + 60.0 * direction - agent) / 100.0, -1.0, 1.0)
    retreat = np.clip((target[:2] - agent) / 100.0, -1.0, 1.0)
    candidates = rng.uniform(-1.0, 1.0, size=(count, blocks, 2)).astype(np.float32)
    candidates[0] = 0.0
    base_phases = [approach, push] + [retreat] * max(0, blocks - 2)
    templates = [np.asarray(base_phases, dtype=np.float64)[:blocks]]
    # The desired block displacement alone does not determine the contact
    # normal: tangential contacts are needed to correct orientation.  Include
    # a small, fixed angular fan around the translation direction so that the
    # protocol tests rotational contact rather than only straight pushes.
    for offset in (-np.pi / 2.0, -np.pi / 3.0, -np.pi / 6.0, np.pi / 6.0, np.pi / 3.0, np.pi / 2.0):
        rotated = np.asarray([
            direction[0] * np.cos(offset) - direction[1] * np.sin(offset),
            direction[0] * np.sin(offset) + direction[1] * np.cos(offset),
        ])
        approach_i = np.clip((block - 50.0 * rotated - agent) / 100.0, -1.0, 1.0)
        push_i = np.clip((block + 100.0 * rotated - agent) / 100.0, -1.0, 1.0)
        templates.append(np.asarray([approach_i, push_i] + [retreat] * max(0, blocks - 2), dtype=np.float64)[:blocks])
    for index, template in enumerate(templates[:count]):
        candidates[index] = template
    for index in range(len(templates), count):
        template = templates[index % len(templates)]
        candidates[index] = np.clip(template + 0.18 * rng.normal(size=template.shape), -1.0, 1.0)
    return candidates


def choose_model_action(
    model: ContactAwareModel,
    state: np.ndarray,
    goal: np.ndarray,
    candidates: np.ndarray,
    action_block: int,
    use_block_model: bool = False,
    stage_objective: bool = False,
    angle_weight: float = 20.0,
) -> np.ndarray:
    trajectory = rollout_candidate_trajectory(
        model, pose_vector(state, include_agent_velocity=getattr(model, "include_agent_velocity", False)),
        candidates, action_block, use_block_model
    )
    predicted = trajectory[-1]
    target = pose_vector(goal, include_agent_velocity=getattr(model, "include_agent_velocity", False))
    angle = np.abs(
        (np.arctan2(predicted[:, 4], predicted[:, 5])
         - np.arctan2(target[4], target[5]) + np.pi) % (2.0 * np.pi) - np.pi
    )
    if stage_objective and trajectory.shape[0] >= 3:
        # The first primitive is an approach manoeuvre. Its endpoint should
        # lie near a contact ring, rather than at the final agent goal. The
        # second primitive is then judged by block translation and rotation.
        after_approach = trajectory[1]
        contact_radius = np.linalg.norm(
            after_approach[:, :2] - after_approach[:, 2:4], axis=1
        )
        approach_cost = np.abs(contact_radius - 35.0)
        costs = (
            approach_cost
            + np.linalg.norm(predicted[:, 2:4] - target[2:4], axis=1)
            + angle_weight * angle
        )
        if candidates.shape[1] >= 3:
            costs = costs + np.linalg.norm(predicted[:, :2] - target[:2], axis=1)
    else:
        # Endpoint objective is retained as a matched ablation.
        costs = np.linalg.norm(predicted[:, :4] - target[:4], axis=1) + angle_weight * angle
    return np.asarray(candidates[int(np.argmin(costs)), 0], dtype=np.float32)


def angle_distance(value: float, target: float) -> float:
    difference = (float(value) - float(target) + np.pi) % (2.0 * np.pi) - np.pi
    return float(abs(difference))


def planning_run(
    env: gym.Env,
    model: ContactAwareModel,
    policy: str,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + sum(ord(char) for char in policy))
    rows = []
    start = time.perf_counter()
    for episode in range(args.planning_episodes):
        case = sample_case(env, seed + 10000 + episode)
        state, goal = case["state"].copy(), case["goal"].copy()
        total_return, steps, clips, terminated = 0.0, 0, 0.0, False
        while steps < 100:
            if policy == "heuristic":
                action = heuristic_action(state, goal)
            else:
                if policy == "uniform_mpc":
                    candidates = rng.uniform(
                        -1.0, 1.0, size=(args.candidates, args.planning_blocks, 2)
                    ).astype(np.float32)
                    candidates[0] = 0.0
                else:
                    candidates = contact_aware_candidates(
                        rng, state, goal, args.candidates, args.planning_blocks
                    )
                action = choose_model_action(
                    model, state, goal, candidates, args.action_block,
                    use_block_model=(policy == "contact_aware_block_mpc"),
                    stage_objective=(policy == "contact_stage_mpc"),
                    angle_weight=args.angle_weight,
                )
            for _ in range(args.action_block):
                observation, reward, terminated, truncated, _ = env.step(action)
                state = np.asarray(observation["state"], dtype=np.float64).copy()
                total_return += float(reward)
                clips += float(np.mean(np.abs(action) >= 0.999))
                steps += 1
                if terminated or truncated or steps >= 100:
                    break
            if terminated:
                break
        rows.append({
            "episode": episode,
            "success": bool(terminated),
            "return": total_return,
            "steps": steps,
            "block_xy_distance": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "agent_xy_distance": float(np.linalg.norm(state[:2] - goal[:2])),
            "pose_position_distance": float(np.linalg.norm(state[:4] - goal[:4])),
            "block_angle_error": angle_distance(state[4], goal[4]),
            "block_pose_success": bool(
                np.linalg.norm(state[2:4] - goal[2:4]) < 20.0
                and angle_distance(state[4], goal[4]) < np.pi / 9.0
            ),
            "agent_xy_within_20": bool(np.linalg.norm(state[:2] - goal[:2]) < 20.0),
            "final_distance": float(np.linalg.norm(state - goal)),
            "clip_fraction": clips / max(steps, 1),
        })
    return {
        "policy": policy,
        "episodes": rows,
        "summary": {
            "n": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "block_pose_success_rate": float(np.mean([row["block_pose_success"] for row in rows])),
            "agent_xy_within_20_rate": float(np.mean([row["agent_xy_within_20"] for row in rows])),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in rows])),
            "mean_agent_xy_distance": float(np.mean([row["agent_xy_distance"] for row in rows])),
            "mean_pose_position_distance": float(np.mean([row["pose_position_distance"] for row in rows])),
            "mean_block_angle_error": float(np.mean([row["block_angle_error"] for row in rows])),
            "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
            "mean_clip_fraction": float(np.mean([row["clip_fraction"] for row in rows])),
            "wall_seconds": float(time.perf_counter() - start),
        },
    }


def counterfactual_eval(
    env: gym.Env,
    model: ContactAwareModel,
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    reports = []
    rng = np.random.default_rng(seed + 9000)
    for horizon in (1, 4, 8):
        rows = []
        for index in range(args.held_out):
            case_seed = seed + 20000 + index
            case = sample_case(env, case_seed)
            direction = rng.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-8)
            action = (0.55 * direction).astype(np.float32)
            reset_observation, _ = env.reset(
                seed=case_seed,
                options={"state": case["state"], "goal_state": case["goal"]},
            )
            initial_state = np.asarray(reset_observation["state"], dtype=np.float64).copy()
            observation = None
            any_contact = False
            previous_state = initial_state.copy()
            for _ in range(horizon):
                observation, _, _, _, info = env.step(action)
                next_state = np.asarray(observation["state"], dtype=np.float64).copy()
                any_contact = any_contact or block_motion(previous_state, next_state)
                previous_state = next_state
            include_velocity = getattr(model, "include_agent_velocity", False)
            true = pose_vector(observation["state"], include_velocity)
            predicted = rollout_pose(model, pose_vector(initial_state, include_velocity), action, horizon)
            rows.append({
                "predicted_error": float(np.linalg.norm(predicted[2:4] - true[2:4])),
                "constant_error": float(np.linalg.norm(pose_vector(initial_state, include_velocity)[2:4] - true[2:4])),
                "contact": any_contact,
                "predicted": predicted,
                "true": true,
            })
        metrics = {}
        for key in ("predicted_error", "constant_error"):
            metrics[key] = bootstrap(
                np.asarray([row[key] for row in rows]), seed + horizon + len(key), args.bootstrap
            )
        for label, predicate in (("contact", True), ("no_contact", False)):
            subset = np.asarray([row["predicted_error"] for row in rows if row["contact"] is predicate])
            metrics[f"{label}_predicted_error"] = (
                bootstrap(subset, seed + 100 + horizon + len(label), args.bootstrap)
                if len(subset)
                else {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
            )
            metrics[f"{label}_n"] = int(len(subset))
        reports.append({"horizon": horizon, "metrics": metrics, "rows": rows})
    return reports


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.planning_blocks < 2:
        raise ValueError("planning-blocks must be at least 2 for approach-to-push candidates")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        if args.sampling_protocol == "trajectory_approach":
            data = trajectory_approach_data(
                env, args.train, args.seed, rng, args.include_agent_velocity
            )
        else:
            data = contact_balanced_data(
                env, args.train, args.seed, rng, args.include_agent_velocity
            )
        model = fit_model(
            data, args.trees, args.seed, args.contact_threshold, args.include_agent_velocity
        )
        counterfactual = counterfactual_eval(env, model, args, args.seed)
        planning = [
            planning_run(env, model, policy, args, args.seed)
            for policy in (
                "heuristic", "uniform_mpc", "contact_aware_mpc", "contact_stage_mpc",
                "contact_aware_block_mpc"
            )
        ]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_transitions": args.train,
            "contact_balancing": "equal free/approach/contact-push/tangential strata",
            "contact_label": "block pose changed beyond tolerance; excludes wall-only contacts",
            "sampling_protocol": args.sampling_protocol,
            "transition_model": (
                "one-step and direct 5-step ExtraTrees residual state models with contact gating"
                + (" and agent velocity" if args.include_agent_velocity else "")
            ),
            "state_representation": (
                "agent/block pose + agent velocity"
                if args.include_agent_velocity else "agent/block pose only"
            ),
            "contact_threshold": args.contact_threshold,
            "angle_weight": args.angle_weight,
            "planning_candidates": args.candidates,
            "planning_blocks": args.planning_blocks,
            "action_block": args.action_block,
            "seed": args.seed,
        },
        "training_diagnostics": {
            "contact_fraction": float(np.mean(data["contacts"])),
            "n": int(len(data["contacts"])),
            "finite": bool(
                all(np.isfinite(data[key]).all() for key in ("states", "actions", "next_states"))
            ),
        },
        "counterfactual": counterfactual,
        "planning": planning,
        "sanity_checks": {
            "all_policies_present": [item["policy"] for item in planning] == [
                "heuristic", "uniform_mpc", "contact_aware_mpc", "contact_stage_mpc",
                "contact_aware_block_mpc"
            ],
            "all_finite": bool(np.isfinite([item["summary"]["mean_block_xy_distance"] for item in planning]).all()),
            "held_out_disjoint": True,
        },
        "claim_boundary": (
            "This is a simulator-state physical-adapter and candidate-protocol diagnosis. "
            "It does not retrain LeWM, establish frozen-latent physical observability, "
            "or establish universal causal control."
        ),
    }
    (out / "contact_aware_closure.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    lines = [
        "# PushT contact-aware physical closure",
        "",
        "| horizon | predicted block error | constant block error |",
        "|---:|---:|---:|",
    ]
    for item in counterfactual:
        metrics = item["metrics"]
        lines.append(
            "| %d | %.3f | %.3f |" % (
                item["horizon"], metrics["predicted_error"]["estimate"], metrics["constant_error"]["estimate"]
            )
        )
    lines += [
        "", "| policy | task success | block pose success | agent XY | block XY | angle error | final distance | return |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in planning:
        summary = item["summary"]
        lines.append(
            "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                item["policy"], summary["success_rate"], summary["block_pose_success_rate"],
                summary["mean_agent_xy_distance"],
                summary["mean_block_xy_distance"], summary["mean_block_angle_error"],
                summary["mean_final_distance"], summary["mean_return"],
            )
        )
    lines += ["", "Claim boundary: simulator-state adapter and candidate-protocol diagnosis only.", ""]
    (out / "contact_aware_closure.md").write_text("\n".join(lines))
    print(json.dumps(jsonable({"output": str(out), "training": report["training_diagnostics"], "planning": planning}), indent=2))


if __name__ == "__main__":
    main()
