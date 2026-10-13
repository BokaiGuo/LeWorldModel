"""Audit and repair signed torque ranking on root-disjoint PushT contacts.

The RGB terminal ranker failures are not safely diagnosed by another endpoint
cost sweep.  This experiment isolates the remaining interface question:
given a contact root and a small inward/tangential pulse, can a learned model
predict the *signed* change in block orientation?  Train and evaluation roots
are generated with disjoint seeds.  The model never receives a simulator
successor at selection time; successors are used only for offline labels and
the next observation in the closed loop.

The report has three separate gates:

* signed-effect direction accuracy on held-out roots;
* held-out angular regret against an oracle pulse;
* terminal-start closed-loop angular/strict success.

This is a privileged physical-adapter experiment, not visual LeWM evidence.
It is intentionally fail-closed: a model is not promoted by the script, and
the report records whether the learned selector beats the fixed geometric
sign baseline on the same held-out roots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_receding_block_pose_planner import make_state_only_env
from experiments.run_causal_visual_pusht_world_model import terminal_pose_cases


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def angle_error(value: float, target: float) -> float:
    return float(abs(float(wrap_angle(float(value) - float(target)))))


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), 1.0e-8)


def support_inward_normal(env: gym.Env, state: np.ndarray) -> np.ndarray:
    """Return the inward normal of the visible/physical T support boundary."""
    state = np.asarray(state, dtype=np.float64)
    block = env.unwrapped.block
    vertices = np.asarray([
        [float(world_point.x), float(world_point.y)]
        for shape in block.shapes
        for point in shape.get_vertices()
        for world_point in [block.local_to_world(point)]
    ], dtype=np.float64)
    outward_to_agent = unit(state[:2] - state[2:4])
    support = vertices[int(np.argmax(vertices @ outward_to_agent))]
    # The action component that maintains contact points from the pusher into
    # the block, i.e. opposite the support-to-agent outward normal.
    return unit(support - state[:2])


def raw_signed_delta(before: np.ndarray, after: np.ndarray) -> float:
    return float(wrap_angle(float(after[4]) - float(before[4])))


def contact_features(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Construct a translation-invariant, action-conditioned contact chart."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    agent_to_block = state[2:4] - state[:2]
    distance = float(np.linalg.norm(agent_to_block))
    normal = unit(agent_to_block)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    theta = float(state[4])
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    block_axes = np.asarray([cos_theta, sin_theta], dtype=np.float64)
    # The PushT contact mode is determined by where the agent touches the
    # asymmetric T, not only by the world-frame lever arm.  Explicitly rotate
    # both relative position and action into the block frame so the learner
    # can distinguish stem/arm contacts under global orientation changes.
    body_rotation = np.asarray([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    body_relative = body_rotation @ agent_to_block
    body_action = body_rotation @ action
    action_normal = float(np.dot(action, normal))
    action_tangent = float(np.dot(action, tangent))
    lever = -agent_to_block
    signed_moment = float(lever[0] * action[1] - lever[1] * action[0])
    return np.asarray([
        distance,
        normal[0], normal[1],
        tangent[0], tangent[1],
        np.sin(theta), np.cos(theta),
        block_axes[0], block_axes[1],
        action[0], action[1],
        action_normal, action_tangent, signed_moment,
        abs(action_normal), abs(action_tangent),
        body_relative[0], body_relative[1], body_action[0], body_action[1],
    ], dtype=np.float64)


def pulse_actions(
    state: np.ndarray,
    tangent_magnitudes: tuple[float, ...],
    inward_magnitude: float,
    support_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired negative/positive tangent actions and their moment signs."""
    state = np.asarray(state, dtype=np.float64)
    normal = unit(state[2:4] - state[:2]) if support_normal is None else unit(support_normal)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    actions: list[np.ndarray] = []
    signs: list[float] = []
    for magnitude in tangent_magnitudes:
        for sign in (-1.0, 1.0):
            action = inward_magnitude * normal + sign * float(magnitude) * tangent
            actions.append(action.astype(np.float32))
            # The physical lever is agent - block = -normal.  With the
            # tangent convention above, its cross product with +tangent is
            # negative; record the actual signed moment, not the arbitrary
            # tangent enumeration sign.
            lever = -normal
            signs.append(float(np.sign(lever[0] * action[1] - lever[1] * action[0])))
    return np.asarray(actions, dtype=np.float32), np.asarray(signs, dtype=np.float64)


def make_terminal_roots(
    env: gym.Env,
    count: int,
    seed: int,
    *,
    jitter: float = 16.0,
    contact_radius: float = 36.0,
    root_protocol: str = "terminal_pose",
    support_gap_max: float = 0.0,
) -> list[dict[str, np.ndarray]]:
    """Generate explicit terminal/contact roots from independent environment seeds."""
    if root_protocol == "terminal_pose":
        cases = terminal_pose_cases(
            env, int(count), int(seed), position_jitter=float(jitter),
            support_gap_max=float(support_gap_max), support_angle_samples=32,
        )
        return [
            {"state": np.asarray(case[0], dtype=np.float64).copy(), "goal": np.asarray(case[1], dtype=np.float64).copy()}
            for case in cases
        ]
    if root_protocol != "radial_contact":
        raise ValueError(f"unknown root protocol: {root_protocol}")
    roots: list[dict[str, np.ndarray]] = []
    index = 0
    while len(roots) < int(count):
        _, info = env.reset(seed=int(seed + index))
        goal = np.asarray(info["goal_state"], dtype=np.float64).copy()
        rng = np.random.default_rng(int(seed + 10000 + index))
        block = goal.copy()
        block[2:4] += rng.uniform(-jitter, jitter, size=2)
        # Keep the root in the valid canvas while retaining the goal angle as
        # the only task target.  Errors are deliberately broad enough to make
        # both torque signs meaningful, but remain in the local terminal tube.
        block[2:4] = np.clip(block[2:4], 70.0, 442.0)
        block[4] = float(wrap_angle(goal[4] + rng.choice((-1.0, 1.0)) * rng.uniform(0.45, 1.55)))
        normal_angle = rng.uniform(-np.pi, np.pi)
        normal = np.asarray([np.cos(normal_angle), np.sin(normal_angle)])
        state = block.copy()
        state[:2] = block[2:4] + contact_radius * normal
        state[:2] = np.clip(state[:2], 30.0, 482.0)
        state[5:] = 0.0
        # Validate the explicit reset and store the simulator's accepted root.
        observation, reset_info = env.reset(
            seed=int(seed + 50000 + index),
            options={"state": state, "goal_state": goal},
        )
        accepted = np.asarray(observation["state"], dtype=np.float64).copy()
        accepted_goal = np.asarray(reset_info["goal_state"], dtype=np.float64).copy()
        distance = float(np.linalg.norm(accepted[:2] - accepted[2:4]))
        if 24.0 <= distance <= 52.0 and np.isfinite(accepted).all():
            roots.append({"state": accepted, "goal": accepted_goal})
        index += 1
    return roots


def execute_pulse(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    pulse_steps: int,
) -> dict[str, Any]:
    observation, _ = env.reset(
        seed=0,
        options={"state": np.asarray(state).copy(), "goal_state": np.asarray(goal).copy()},
    )
    before = np.asarray(observation["state"], dtype=np.float64).copy()
    for _ in range(int(pulse_steps)):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            break
    after = np.asarray(observation["state"], dtype=np.float64).copy()
    return {
        "before": before,
        "after": after,
        "signed_delta": raw_signed_delta(before, after),
        "angle_error_before": angle_error(before[4], goal[4]),
        "angle_error_after": angle_error(after[4], goal[4]),
        "xy_before": float(np.linalg.norm(before[2:4] - goal[2:4])),
        "xy_after": float(np.linalg.norm(after[2:4] - goal[2:4])),
        "contact_distance_after": float(np.linalg.norm(after[:2] - after[2:4])),
    }


def collect_effect_rows(
    env: gym.Env,
    roots: list[dict[str, np.ndarray]],
    tangent_magnitudes: tuple[float, ...],
    inward_magnitude: float,
    pulse_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_id, root in enumerate(roots):
        env.reset(seed=0, options={"state": root["state"].copy(), "goal_state": root["goal"].copy()})
        support_normal = support_inward_normal(env, root["state"])
        actions, signs = pulse_actions(root["state"], tangent_magnitudes, inward_magnitude, support_normal)
        for action_id, (action, moment_sign) in enumerate(zip(actions, signs)):
            result = execute_pulse(env, root["state"], root["goal"], action, pulse_steps)
            rows.append({
                "root_id": int(root_id),
                "action_id": int(action_id),
                "state": root["state"].copy(),
                "goal": root["goal"].copy(),
                "action": np.asarray(action, dtype=np.float64),
                "moment_sign": float(moment_sign),
                "features": np.concatenate([contact_features(root["state"], action), support_normal]),
                "support_normal": support_normal.copy(),
                **result,
            })
    return rows


def fit_models(rows: list[dict[str, Any]], seed: int) -> tuple[ExtraTreesRegressor, ExtraTreesRegressor, ExtraTreesClassifier]:
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    deltas = np.asarray([row["signed_delta"] for row in rows], dtype=np.float64)
    block_deltas = np.asarray([
        np.asarray(row["after"], dtype=np.float64)[2:4]
        - np.asarray(row["before"], dtype=np.float64)[2:4]
        for row in rows
    ], dtype=np.float64)
    labels = (deltas > 0.0).astype(np.int64)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("signed-effect training labels contain only one direction")
    regressor = ExtraTreesRegressor(
        n_estimators=128, min_samples_leaf=2, max_features=1.0,
        random_state=int(seed), n_jobs=1,
    ).fit(features, deltas)
    pose_regressor = ExtraTreesRegressor(
        n_estimators=128, min_samples_leaf=2, max_features=1.0,
        random_state=int(seed + 2), n_jobs=1,
    ).fit(features, np.column_stack([deltas, block_deltas]))
    classifier = ExtraTreesClassifier(
        n_estimators=128, min_samples_leaf=2, max_features=1.0,
        class_weight="balanced", random_state=int(seed + 1), n_jobs=1,
    ).fit(features, labels)
    return regressor, pose_regressor, classifier


def bootstrap_mean(values: list[float] | np.ndarray, seed: int, draws: int = 4000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    means = values[rng.integers(0, values.size, size=(int(draws), values.size))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def json_safe(value: Any) -> Any:
    """Serialize non-finite diagnostics as explicit JSON nulls."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer, str, bool)) or value is None:
        return value
    return value


def select_index(
    root_rows: list[dict[str, Any]],
    regressor: ExtraTreesRegressor,
    pose_regressor: ExtraTreesRegressor,
    classifier: ExtraTreesClassifier,
    mode: str,
) -> int:
    features = np.asarray([row["features"] for row in root_rows], dtype=np.float64)
    if mode == "signed_regressor":
        predicted_delta = np.asarray(regressor.predict(features), dtype=np.float64)
        current = float(root_rows[0]["before"][4])
        goal = float(root_rows[0]["goal"][4])
        costs = np.asarray([angle_error(current + delta, goal) for delta in predicted_delta])
    elif mode in {"direction_classifier", "pose_regressor"}:
        probabilities = classifier.predict_proba(features)
        positive_class = int(np.flatnonzero(classifier.classes_ == 1)[0])
        probability_positive = probabilities[:, positive_class]
        current = float(root_rows[0]["before"][4])
        goal = float(root_rows[0]["goal"][4])
        desired = 1.0 if float(wrap_angle(goal - current)) >= 0.0 else -1.0
        # Classifier is deliberately used only as a half-space filter; the
        # smallest predicted angular error breaks ties within that half-space.
        valid = np.where((probability_positive >= 0.5) == (desired > 0.0))[0]
        if valid.size == 0:
            valid = np.arange(len(root_rows))
        predicted_delta = np.asarray(regressor.predict(features), dtype=np.float64)
        if mode == "pose_regressor":
            predicted_pose = np.asarray(pose_regressor.predict(features), dtype=np.float64)
            predicted_xy = np.asarray(root_rows[0]["before"][2:4])[None, :] + predicted_pose[:, 1:3]
            predicted_angle = current + predicted_pose[:, 0]
            costs = np.asarray([
                float(np.linalg.norm(xy - np.asarray(root_rows[0]["goal"][2:4])) / 20.0)
                + angle_error(theta, goal) / (np.pi / 9.0)
                for xy, theta in zip(predicted_xy, predicted_angle)
            ])
        else:
            costs = np.asarray([angle_error(current + delta, goal) for delta in predicted_delta])
        masked = np.full_like(costs, np.inf)
        masked[valid] = costs[valid]
        costs = masked
    elif mode == "geometric_sign":
        current = float(root_rows[0]["before"][4])
        goal = float(root_rows[0]["goal"][4])
        desired = 1.0 if float(wrap_angle(goal - current)) >= 0.0 else -1.0
        moment = np.asarray([row["moment_sign"] for row in root_rows])
        costs = np.where(moment == desired, 0.0, 1.0)
    else:
        raise ValueError(f"unknown selection mode: {mode}")
    return int(np.argmin(costs))


def evaluate_ranking(
    rows: list[dict[str, Any]],
    regressor: ExtraTreesRegressor,
    pose_regressor: ExtraTreesRegressor,
    classifier: ExtraTreesClassifier,
) -> dict[str, Any]:
    by_root: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_root.setdefault(int(row["root_id"]), []).append(row)
    summary: dict[str, list[float]] = {
        "model_direction_accuracy": [], "geometric_direction_accuracy": [],
        "oracle_regret": [], "regressor_regret": [], "classifier_regret": [],
        "pose_regressor_regret": [], "geometric_regret": [],
    }
    root_records: list[dict[str, Any]] = []
    for root_id, root_rows in sorted(by_root.items()):
        actual_signed = np.asarray([row["signed_delta"] for row in root_rows])
        moment = np.asarray([row["moment_sign"] for row in root_rows])
        features = np.asarray([row["features"] for row in root_rows], dtype=np.float64)
        predicted_signed = np.asarray(regressor.predict(features), dtype=np.float64)
        actual_sign = np.sign(actual_signed)
        valid_sign = actual_sign != 0.0
        if np.any(valid_sign):
            summary["model_direction_accuracy"].append(
                float(np.mean(np.sign(predicted_signed[valid_sign]) == actual_sign[valid_sign]))
            )
            summary["geometric_direction_accuracy"].append(
                float(np.mean(moment[valid_sign] == actual_sign[valid_sign]))
            )
        actual_cost = np.asarray([row["angle_error_after"] for row in root_rows])
        oracle = float(np.min(actual_cost))
        summary["oracle_regret"].append(0.0)
        actual_pose_cost = np.asarray([
            float(np.linalg.norm(np.asarray(row["after"])[2:4] - np.asarray(root_rows[0]["goal"])[2:4]) / 20.0)
            + float(row["angle_error_after"]) / (np.pi / 9.0)
            for row in root_rows
        ])
        pose_oracle = float(np.min(actual_pose_cost))
        selections = {}
        regrets = {}
        for mode, key in (("signed_regressor", "regressor_regret"), ("direction_classifier", "classifier_regret"), ("pose_regressor", "pose_regressor_regret"), ("geometric_sign", "geometric_regret")):
            selected = select_index(root_rows, regressor, pose_regressor, classifier, mode)
            selections[mode] = selected
            regret = float((actual_pose_cost[selected] - pose_oracle) if mode == "pose_regressor" else (actual_cost[selected] - oracle))
            regrets[key] = regret
            summary[key].append(regret)
        root_records.append({
            "root_id": root_id,
            "goal_angle_error": float(root_rows[0]["angle_error_before"]),
            "actual_signed_delta": actual_signed.tolist(),
            "moment_sign": moment.tolist(),
            "actual_angle_cost": actual_cost.tolist(),
            "selections": selections,
            "regrets": regrets,
        })
    return {
        "metrics": {key: bootstrap_mean(value, 91000 + index) for index, (key, value) in enumerate(summary.items())},
        "rows": root_records,
    }


def closed_loop(
    env: gym.Env,
    roots: list[dict[str, np.ndarray]],
    regressor: ExtraTreesRegressor,
    pose_regressor: ExtraTreesRegressor,
    classifier: ExtraTreesClassifier,
    tangent_magnitudes: tuple[float, ...],
    inward_magnitude: float,
    steps: int,
    selection_mode: str,
    feedback_reversal: bool,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for root_id, root in enumerate(roots):
        observation, _ = env.reset(seed=0, options={"state": root["state"].copy(), "goal_state": root["goal"].copy()})
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        goal = root["goal"]
        initial_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
        initial_angle = angle_error(current[4], goal[4])
        trace: list[dict[str, Any]] = []
        previous_observed_delta = 0.0
        for step in range(int(steps)):
            support_normal = support_inward_normal(env, current)
            actions, signs = pulse_actions(current, tangent_magnitudes, inward_magnitude, support_normal)
            candidate_rows: list[dict[str, Any]] = []
            for action_id, (action, moment_sign) in enumerate(zip(actions, signs)):
                candidate_rows.append({
                    "before": current.copy(), "goal": goal.copy(), "action": action,
                    "moment_sign": float(moment_sign),
                    "features": np.concatenate([contact_features(current, action), support_normal]),
                    "angle_error_before": angle_error(current[4], goal[4]),
                })
            selected = select_index(candidate_rows, regressor, pose_regressor, classifier, selection_mode)
            if feedback_reversal and abs(previous_observed_delta) > 1.0e-5:
                # A physical observation is allowed only after the preceding
                # pulse has completed.  If it moved the block in the wrong
                # angular direction, constrain the next choice to the
                # opposite predicted signed effect.  This is an online
                # residual correction, not a simulator successor query.
                features = np.asarray([row["features"] for row in candidate_rows], dtype=np.float64)
                predicted = np.asarray(regressor.predict(features), dtype=np.float64)
                desired_sign = -float(np.sign(previous_observed_delta))
                valid = np.flatnonzero(np.sign(predicted) == desired_sign)
                if valid.size:
                    current_angle = float(current[4])
                    target = float(goal[4])
                    costs = np.asarray([
                        angle_error(current_angle + predicted[index], target) for index in valid
                    ])
                    selected = int(valid[int(np.argmin(costs))])
            action = np.asarray(candidate_rows[selected]["action"], dtype=np.float32)
            observation, _, terminated, truncated, _ = env.step(action)
            next_state = np.asarray(observation["state"], dtype=np.float64).copy()
            trace.append({
                "step": step, "selected": selected, "action": action.tolist(),
                "predicted_signed_delta": float(regressor.predict(np.asarray([candidate_rows[selected]["features"]]))[0]),
                "angle_before": float(angle_error(current[4], goal[4])),
                "angle_after": float(angle_error(next_state[4], goal[4])),
                "xy_after": float(np.linalg.norm(next_state[2:4] - goal[2:4])),
                "contact_distance_after": float(np.linalg.norm(next_state[:2] - next_state[2:4])),
                "feedback_reversal_used": int(feedback_reversal and abs(previous_observed_delta) > 1.0e-5),
            })
            previous_observed_delta = raw_signed_delta(current, next_state)
            current = next_state
            if terminated or truncated:
                break
        final_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
        final_angle = angle_error(current[4], goal[4])
        records.append({
            "root_id": root_id, "initial_xy": initial_xy, "initial_angle": initial_angle,
            "final_xy": final_xy, "final_angle": final_angle,
            "angle_improvement": initial_angle - final_angle,
            "strict_success": int(final_xy < 20.0 and final_angle < np.pi / 9.0),
            "trace": trace,
        })
    return {
        "metrics": {
            "strict_success": bootstrap_mean([row["strict_success"] for row in records], 92001),
            "final_xy": bootstrap_mean([row["final_xy"] for row in records], 92002),
            "final_angle": bootstrap_mean([row["final_angle"] for row in records], 92003),
            "angle_improvement": bootstrap_mean([row["angle_improvement"] for row in records], 92004),
        },
        "rows": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-roots", type=int, default=64)
    parser.add_argument("--held-out-roots", type=int, default=32)
    parser.add_argument(
        "--root-protocol", choices=("terminal_pose", "radial_contact"), default="terminal_pose",
        help="terminal_pose reuses the project terminal-pose root construction; radial_contact is retained as an explicit ablation",
    )
    parser.add_argument("--support-gap-max", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=9181)
    parser.add_argument("--pulse-steps", type=int, default=4)
    parser.add_argument("--closure-steps", type=int, default=24)
    parser.add_argument(
        "--selection-mode", choices=("direction_classifier", "signed_regressor", "pose_regressor"),
        default="direction_classifier",
        help="fixed closed-loop selector; pose_regressor uses predicted XY and angle, the others target angular correction",
    )
    parser.add_argument(
        "--feedback-reversal", action=argparse.BooleanOptionalAction, default=False,
        help="after an observed wrong-way pulse, constrain the next pulse to the opposite learned signed effect",
    )
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+", default=[0.06, 0.10, 0.16])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_roots < 16 or args.held_out_roots < 8:
        raise ValueError("root counts are too small for the declared held-out audit")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tangent_magnitudes = tuple(float(x) for x in args.tangent_magnitudes)
    # The support-aware root protocol and the physical support extractor both
    # require the real block geometry.  The state-only stub intentionally
    # replaces render() with a zero image and is therefore valid only for the
    # radial ablation.
    env = (
        gym.make("swm/PushT-v1", render_mode="rgb_array")
        if args.root_protocol == "terminal_pose"
        else make_state_only_env()
    )
    try:
        train_roots = make_terminal_roots(env, args.train_roots, args.seed + 1000, root_protocol=args.root_protocol, support_gap_max=args.support_gap_max)
        held_roots = make_terminal_roots(env, args.held_out_roots, args.seed + 2000, root_protocol=args.root_protocol, support_gap_max=args.support_gap_max)
        train_rows = collect_effect_rows(env, train_roots, tangent_magnitudes, args.inward_magnitude, args.pulse_steps)
        held_rows = collect_effect_rows(env, held_roots, tangent_magnitudes, args.inward_magnitude, args.pulse_steps)
        regressor, pose_regressor, classifier = fit_models(train_rows, args.seed)
        ranking = evaluate_ranking(held_rows, regressor, pose_regressor, classifier)
        closure = closed_loop(env, held_roots, regressor, pose_regressor, classifier, tangent_magnitudes, args.inward_magnitude, args.closure_steps, args.selection_mode, args.feedback_reversal)
    finally:
        env.close()
    direction = ranking["metrics"]["model_direction_accuracy"]["estimate"]
    selected_regret_key = {
        "direction_classifier": "classifier_regret",
        "signed_regressor": "regressor_regret",
        "pose_regressor": "pose_regressor_regret",
    }[args.selection_mode]
    learned = ranking["metrics"][selected_regret_key]["estimate"]
    geometric = ranking["metrics"]["geometric_regret"]["estimate"]
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_roots": int(args.train_roots), "held_out_roots": int(args.held_out_roots),
            "train_seed": int(args.seed + 1000), "held_out_seed": int(args.seed + 2000),
            "root_protocol": str(args.root_protocol),
            "support_gap_max": float(args.support_gap_max),
            "pulse_steps": int(args.pulse_steps), "closure_steps": int(args.closure_steps),
            "inward_magnitude": float(args.inward_magnitude),
            "tangent_magnitudes": list(tangent_magnitudes),
            "feature_interface": "translation-invariant contact geometry plus action-conditioned signed moment",
            "selection_interface": "direction classifier half-space filter followed by multi-output signed-angle/XY effect regressor",
            "closed_loop_selection_mode": str(args.selection_mode),
            "selected_regret_metric": selected_regret_key,
        },
        "training": {"rows": len(train_rows), "unique_roots": len(train_roots)},
        "held_out_ranking": ranking,
        "closed_loop": closure,
        "gates": {
            "direction_accuracy_target": 0.70,
            "direction_accuracy_pass": bool(direction >= 0.70),
            "learned_regret_below_geometric": bool(learned < geometric),
            "strict_success_is_reported_not_promoted": True,
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": "Privileged simulator-state signed-effect adapter audit; not visual LeWM planning or universal causal dynamics evidence.",
    }
    (output / "signed_torque_audit.json").write_text(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    metrics = ranking["metrics"]
    closure_metrics = closure["metrics"]
    lines = [
        "# PushT signed-torque audit", "",
        "| metric | estimate | 95% low | 95% high |", "|---|---:|---:|---:|",
    ]
    for key in ("model_direction_accuracy", "geometric_direction_accuracy", "regressor_regret", "classifier_regret", "pose_regressor_regret", "geometric_regret"):
        value = metrics[key]
        lines.append(f"| held-out {key} | {value['estimate']:.4f} | {value['ci95_low']:.4f} | {value['ci95_high']:.4f} |")
    for key in ("strict_success", "final_xy", "final_angle", "angle_improvement"):
        value = closure_metrics[key]
        lines.append(f"| closed-loop {key} | {value['estimate']:.4f} | {value['ci95_low']:.4f} | {value['ci95_high']:.4f} |")
    lines += [
        "", f"Gate: direction accuracy >= 0.70: {report['gates']['direction_accuracy_pass']}.",
        f"Gate: learned classifier regret below geometric regret: {report['gates']['learned_regret_below_geometric']}.",
        "Claim boundary: privileged signed-effect adapter audit; no visual-planning or universal-world-model claim.", "",
    ]
    (output / "signed_torque_audit.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "gates": report["gates"], "ranking": metrics, "closed_loop": closure_metrics}, indent=2))


if __name__ == "__main__":
    main()
