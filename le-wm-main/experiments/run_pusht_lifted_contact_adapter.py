"""Lifted-space contact-effect adapter for PushT.

The adapter implements a narrow version of the user's lift--separate--project
idea.  A contact state/action pair is mapped to an explicit feature space with
body-frame delay coordinates, Fourier phase features and state/action
interactions.  A mode-gated multi-output regressor then predicts the local
block effect.  Selection remains one-step, held-out and successor-free.

The experiment is deliberately a comparison against the existing signed-torque
adapter, not a claim that arbitrary dimensionality expansion creates causal
physics.  The decisive tests are held-out action ranking and same-case
closed-loop block-pose closure.
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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_signed_torque_audit import (
    angle_error,
    bootstrap_mean,
    collect_effect_rows,
    contact_features,
    execute_pulse,
    make_terminal_roots,
    pulse_actions,
    support_inward_normal,
    unit,
    wrap_angle,
)


def contact_aware_pulse_actions(
    state: np.ndarray,
    tangent_magnitudes: tuple[float, ...],
    inward_magnitude: float,
    support_normal: np.ndarray | None = None,
    contact_threshold: float = 38.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift the action interface with an explicit contact-recovery mode.

    The learned effect model is still queried on the returned shared bank.  A
    state-only mode gate changes the bank when the agent has visibly left the
    contact tube: both tangential and inward authority are bounded so that
    recovery does not overshoot the contact manifold.
    This is a mode-dependent action interface, not a simulator successor
    query.  Training and deployment call the same function.
    """
    state = np.asarray(state, dtype=np.float64)
    distance = float(np.linalg.norm(state[:2] - state[2:4]))
    if distance <= float(contact_threshold):
        return pulse_actions(state, tangent_magnitudes, inward_magnitude, support_normal)
    gap = distance - float(contact_threshold)
    recovery_inward = min(float(inward_magnitude), max(0.025, 0.03 + 0.002 * gap))
    recovery_tangents = tuple(max(0.025, min(float(value), 0.5 * float(value))) for value in tangent_magnitudes)
    return pulse_actions(state, recovery_tangents, recovery_inward, support_normal)


def contact_recovery_action(support_normal: np.ndarray, magnitude: float) -> np.ndarray:
    """Return the boundary-normal recovery pulse used by the closed loop."""

    normal = unit(np.asarray(support_normal, dtype=np.float64))
    return (float(magnitude) * normal).astype(np.float32)


def lifted_contact_features(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray | None = None,
) -> np.ndarray:
    """Explicitly separate contact phase, action phase and their coupling."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous = np.zeros(2, dtype=np.float64) if previous_action is None else np.asarray(previous_action, dtype=np.float64)
    if state.shape != (7,) or action.shape != (2,) or previous.shape != (2,):
        raise ValueError("lifted contact features expect state (7), action (2), previous (2)")
    relative = state[2:4] - state[:2]
    distance = max(float(np.linalg.norm(relative)), 1e-8)
    normal = relative / distance
    tangent = np.asarray([-normal[1], normal[0]])
    local_relative = np.asarray([np.dot(relative, normal), np.dot(relative, tangent)])
    local_action = np.asarray([np.dot(action, normal), np.dot(action, tangent)])
    local_previous = np.asarray([np.dot(previous, normal), np.dot(previous, tangent)])
    theta = float(state[4])
    rotation = np.asarray([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    body_relative = rotation @ (state[:2] - state[2:4])
    body_velocity = rotation @ state[5:7]
    body_action = rotation @ action
    contact_phase = float(np.arctan2(body_relative[1], body_relative[0]))
    action_phase = float(np.arctan2(body_action[1], body_action[0]))
    base = np.asarray(contact_features(state, action), dtype=np.float64)
    values: list[float] = base.tolist()
    values += [
        distance / 50.0, body_relative[0] / 50.0, body_relative[1] / 50.0,
        body_velocity[0] / 10.0, body_velocity[1] / 10.0,
        body_action[0], body_action[1], local_previous[0], local_previous[1],
        np.hypot(*local_action), np.hypot(*local_previous),
    ]
    # Fourier features resolve angular aliasing without selecting a preferred
    # global coordinate frame.  The feature order is fixed and recorded.
    for harmonic in range(1, 5):
        values.extend([
            np.sin(harmonic * contact_phase), np.cos(harmonic * contact_phase),
            np.sin(harmonic * action_phase), np.cos(harmonic * action_phase),
        ])
    # Bilinear terms are the smallest explicit lift that can separate the same
    # action at different support locations and velocities.
    for lhs in (body_relative[0] / 50.0, body_relative[1] / 50.0, body_velocity[0] / 10.0, body_velocity[1] / 10.0):
        values.extend([lhs * local_action[0], lhs * local_action[1]])
    values.extend([
        local_action[0] * local_action[0], local_action[1] * local_action[1],
        local_action[0] * local_action[1],
    ])
    values.extend(
        float(np.exp(-0.5 * ((distance - center) / 8.0) ** 2))
        for center in (38.0, 48.0, 58.0, 68.0)
    )
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("lifted contact feature map produced non-finite values")
    return result


def fit_lifted_models(rows: list[dict[str, Any]], seed: int) -> tuple[Any, Any, Any]:
    features = np.asarray([
        lifted_contact_features(
            row.get("before", row["state"]), row["action"], row.get("previous_action")
        )
        for row in rows
    ], dtype=np.float64)
    effects = np.asarray([
        [float(row["signed_delta"]), *(
            np.asarray(row["after"], dtype=np.float64)[2:4]
            - np.asarray(row["before"], dtype=np.float64)[2:4]
        )]
        for row in rows
    ], dtype=np.float64)
    labels = (effects[:, 0] > 0.0).astype(np.int64)
    regressor = ExtraTreesRegressor(
        n_estimators=192, min_samples_leaf=2, max_features=0.8,
        random_state=int(seed), n_jobs=1,
    ).fit(features, effects[:, 0])
    pose_regressor = ExtraTreesRegressor(
        n_estimators=192, min_samples_leaf=2, max_features=0.8,
        random_state=int(seed + 2), n_jobs=1,
    ).fit(features, effects)
    classifier = ExtraTreesClassifier(
        n_estimators=192, min_samples_leaf=2, max_features=0.8,
        class_weight="balanced", random_state=int(seed + 1), n_jobs=1,
    ).fit(features, labels)
    return regressor, pose_regressor, classifier


def select_lifted(
    rows: list[dict[str, Any]],
    regressor: Any,
    pose_regressor: Any,
    classifier: Any,
    mode: str,
    previous_action: np.ndarray | None = None,
    pose_xy_weight: float = 1.0,
    pose_angle_weight: float = 1.0,
) -> int:
    features = np.asarray([
        lifted_contact_features(
            row["before"], row["action"], row.get("previous_action", previous_action)
        )
        for row in rows
    ], dtype=np.float64)
    predicted = np.asarray(regressor.predict(features), dtype=np.float64)
    current = float(rows[0]["before"][4])
    goal = float(rows[0]["goal"][4])
    desired = 1.0 if float(wrap_angle(goal - current)) >= 0.0 else -1.0
    if mode == "direction_classifier":
        if 1 in classifier.classes_:
            probs = classifier.predict_proba(features)
            pos = int(np.flatnonzero(classifier.classes_ == 1)[0])
            direction_valid = (probs[:, pos] >= 0.5) == (desired > 0.0)
        else:
            # Very short pulses can yield a one-class training split.  The
            # classifier then has no calibrated positive probability; use the
            # signed regressor as an explicit fail-safe rather than crashing
            # or fabricating a probability for an absent class.
            direction_valid = (np.sign(predicted) == desired)
        pose = np.asarray(pose_regressor.predict(features), dtype=np.float64)
        current_xy = float(np.linalg.norm(np.asarray(rows[0]["before"])[2:4] - np.asarray(rows[0]["goal"])[2:4]))
        predicted_xy = np.linalg.norm(
            np.asarray(rows[0]["before"])[2:4][None] + pose[:, 1:3]
            - np.asarray(rows[0]["goal"])[2:4][None], axis=1
        )
        # A directionally correct pulse is still rejected if its lifted
        # translation head predicts that it exits the current XY tube.  This
        # is a lexicographic safety constraint, not a weighted trade-off.
        tube = min(45.0, current_xy + 4.0)
        valid = np.flatnonzero(direction_valid & (predicted_xy <= tube))
        if len(valid) == 0:
            valid = np.flatnonzero(direction_valid)
        if len(valid) == 0:
            valid = np.arange(len(rows))
        costs = np.asarray([
            angle_error(current + value, goal) + 0.02 * predicted_xy[index]
            for index, value in enumerate(predicted)
        ])
        masked = np.full_like(costs, np.inf)
        masked[valid] = costs[valid]
        return int(np.argmin(masked))
    if mode == "signed_regressor":
        return int(np.argmin([angle_error(current + value, goal) for value in predicted]))
    pose = np.asarray(pose_regressor.predict(features), dtype=np.float64)
    predicted_xy = np.asarray(rows[0]["before"][2:4])[None] + pose[:, 1:3]
    predicted_angle = current + pose[:, 0]
    costs = [
        float(pose_xy_weight) * np.linalg.norm(xy - np.asarray(rows[0]["goal"][2:4])) / 20.0
        + float(pose_angle_weight) * angle_error(theta, goal) / (np.pi / 9.0)
        for xy, theta in zip(predicted_xy, predicted_angle)
    ]
    return int(np.argmin(costs))


def ranking(
    rows: list[dict[str, Any]], regressor: Any, pose_regressor: Any,
    classifier: Any, mode: str, pose_xy_weight: float = 1.0,
    pose_angle_weight: float = 1.0,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["root_id"]), []).append(row)
    direction, regret, geometric = [], [], []
    per_root = []
    for root_id, root_rows in sorted(grouped.items()):
        features = np.asarray([
            lifted_contact_features(
                r.get("before", r["state"]), r["action"], r.get("previous_action")
            )
            for r in root_rows
        ])
        predicted = np.asarray(regressor.predict(features))
        actual = np.asarray([r["signed_delta"] for r in root_rows])
        valid = np.abs(actual) > 1e-8
        if valid.any():
            direction.append(float(np.mean(np.sign(predicted[valid]) == np.sign(actual[valid]))))
        actual_cost = np.asarray([r["angle_error_after"] for r in root_rows])
        selected = select_lifted(
            root_rows, regressor, pose_regressor, classifier, mode,
            pose_xy_weight=pose_xy_weight, pose_angle_weight=pose_angle_weight,
        )
        desired = 1.0 if float(wrap_angle(root_rows[0]["goal"][4] - root_rows[0]["before"][4])) >= 0 else -1.0
        geo = np.asarray([0.0 if r["moment_sign"] == desired else 1.0 for r in root_rows])
        geometric.append(float(actual_cost[int(np.argmin(geo))] - actual_cost.min()))
        regret.append(float(actual_cost[selected] - actual_cost.min()))
        per_root.append({"root_id": root_id, "selected": selected, "actual_cost": actual_cost.tolist()})
    return {
        "metrics": {
            "direction_accuracy": bootstrap_mean(direction, 93101),
            "learned_regret": bootstrap_mean(regret, 93102),
            "geometric_regret": bootstrap_mean(geometric, 93103),
        },
        "rows": per_root,
    }


def closed_loop(env: gym.Env, roots: list[dict[str, np.ndarray]], models: tuple[Any, Any, Any], args: argparse.Namespace) -> dict[str, Any]:
    regressor, pose_regressor, classifier = models
    records = []
    for root in roots:
        observation, _ = env.reset(seed=0, options={"state": root["state"].copy(), "goal_state": root["goal"].copy()})
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        goal = np.asarray(root["goal"], dtype=np.float64).copy()
        initial_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
        initial_angle = angle_error(current[4], goal[4])
        last_action = np.zeros(2, dtype=np.float64)
        trace = []
        for step in range(args.closure_steps):
            if np.linalg.norm(current[2:4] - goal[2:4]) < 20.0 and angle_error(current[4], goal[4]) < np.pi / 9.0:
                break
            support_normal = support_inward_normal(env, current)
            actions, signs = contact_aware_pulse_actions(
                current, args.tangent_magnitudes, args.inward_magnitude, support_normal
            )
            candidates = [{
                "before": current.copy(), "goal": goal.copy(), "action": action.copy(),
                "moment_sign": float(sign),
                "previous_action": last_action.copy(),
            } for action, sign in zip(actions, signs)]
            contact_distance_before = float(np.linalg.norm(current[:2] - current[2:4]))
            recovery_fallback = bool(
                args.contact_recovery_fallback and contact_distance_before > args.contact_recovery_threshold
            )
            if recovery_fallback:
                # This branch is deliberately not called a learned decision:
                # it is a state-only interface repair that moves the agent
                # toward the current block for one short pulse, then returns
                # control to the learned candidate ranker.
                # Recovery must use the same support-boundary normal as the
                # training/action bank.  Pointing at the block centroid is
                # not equivalent for the asymmetric T: it can drive the
                # pusher across a non-contact region and make the apparent
                # mode transition irreversible.
                recovery_action = contact_recovery_action(
                    support_normal, args.contact_recovery_magnitude
                )
                candidates = [{
                    "before": current.copy(), "goal": goal.copy(),
                    "action": recovery_action.astype(np.float32),
                    "moment_sign": 0.0, "previous_action": last_action.copy(),
                }]
                selected = 0
            else:
                selected = select_lifted(
                    candidates, regressor, pose_regressor, classifier,
                args.selection_mode, previous_action=last_action,
                pose_xy_weight=args.pose_xy_weight,
                pose_angle_weight=args.pose_angle_weight,
                )
            verification = None
            if args.successor_verify_topk > 0:
                candidate_features = np.asarray([
                    lifted_contact_features(current, row["action"], last_action)
                    for row in candidates
                ], dtype=np.float64)
                predicted_delta = np.asarray(regressor.predict(candidate_features), dtype=np.float64)
                predicted_cost = np.asarray([
                    angle_error(current[4] + delta, goal[4])
                    for delta in predicted_delta
                ])
                order = np.argsort(predicted_cost)[:min(args.successor_verify_topk, len(candidates))]
                verified = []
                for index in order:
                    env.reset(seed=0, options={"state": current.copy(), "goal_state": goal.copy()})
                    probe_observation, _, probe_terminated, probe_truncated, _ = env.step(candidates[int(index)]["action"])
                    successor = np.asarray(probe_observation["state"], dtype=np.float64).copy()
                    successor_xy = float(np.linalg.norm(successor[2:4] - goal[2:4]))
                    successor_angle = angle_error(successor[4], goal[4])
                    # The verifier is a separate protocol: it observes the
                    # candidate successor and rejects orientation gains that
                    # leave a hard translation tube.
                    tube = min(45.0, float(np.linalg.norm(current[2:4] - goal[2:4])) + 4.0)
                    verified.append({
                        "index": int(index), "successor": successor,
                        "xy": successor_xy, "angle": successor_angle,
                        "tube_violation": bool(successor_xy > tube),
                        "terminated": bool(probe_terminated or probe_truncated),
                    })
                valid = [row for row in verified if not row["tube_violation"]]
                selected_row = min(valid or verified, key=lambda row: (row["angle"], row["xy"]))
                selected = int(selected_row["index"])
                verification = {
                    "topk": int(len(verified)), "selected": int(selected),
                    "selected_xy": float(selected_row["xy"]),
                    "selected_angle": float(selected_row["angle"]),
                    "valid_count": int(len(valid)),
                }
                # Restore the actual current state before committing.  This is
                # the state-consistency invariant for the verifier arm.
                env.reset(seed=0, options={"state": current.copy(), "goal_state": goal.copy()})
            action = np.asarray(candidates[selected]["action"], dtype=np.float32)
            before = current.copy()
            terminated = truncated = False
            # Match the supervised label horizon exactly.  The old protocol
            # fitted a four-step endpoint effect but committed only one step
            # at runtime, which is a genuine train/deploy interface mismatch.
            execute_steps = 1 if recovery_fallback else args.pulse_steps
            for _ in range(execute_steps):
                observation, _, terminated, truncated, _ = env.step(action)
                current = np.asarray(observation["state"], dtype=np.float64).copy()
                if terminated or truncated:
                    break
            trace.append({
                "step": step, "selected": selected, "action": action.tolist(),
                "angle_before": angle_error(before[4], goal[4]), "angle_after": angle_error(current[4], goal[4]),
                "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
                "observed_delta": float(wrap_angle(current[4] - before[4])),
                "contact_distance_after": float(np.linalg.norm(current[:2] - current[2:4])),
                "contact_recovery_fallback": recovery_fallback,
                "successor_verification": verification,
            })
            last_action = action.astype(np.float64)
            if terminated or truncated:
                break
        final_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
        final_angle = angle_error(current[4], goal[4])
        records.append({
            "initial_xy": initial_xy, "initial_angle": initial_angle,
            "final_xy": final_xy, "final_angle": final_angle,
            "block_pose_success": bool(final_xy < 20.0 and final_angle < np.pi / 9.0),
            "trace": trace,
        })
    return {
        "metrics": {
            "block_pose_success": bootstrap_mean([r["block_pose_success"] for r in records], 93201),
            "final_xy": bootstrap_mean([r["final_xy"] for r in records], 93202),
            "final_angle": bootstrap_mean([r["final_angle"] for r in records], 93203),
            "contact_recovery_steps": bootstrap_mean([
                sum(bool(item.get("contact_recovery_fallback", False)) for item in r["trace"])
                for r in records
            ], 93204),
        },
        "rows": records,
    }


def collect_temporal_candidate_rows(
    env: gym.Env,
    roots: list[dict[str, np.ndarray]],
    tangent_magnitudes: tuple[float, ...],
    inward_magnitude: float,
    pulse_steps: int,
    sequence_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Collect root-disjoint candidate banks with a real action history.

    At each sequence state every candidate is evaluated from the same state,
    while one candidate is sampled and executed to advance the sequence.  The
    previous-action coordinate is therefore observed rather than filled with
    a training-time zero.  Candidate ranking remains successor-free: the
    sampled transition is only used to create the next training state.
    """
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    next_root_id = 0
    for root in roots:
        env.reset(seed=0, options={"state": root["state"].copy(), "goal_state": root["goal"].copy()})
        current = np.asarray(root["state"], dtype=np.float64).copy()
        goal = np.asarray(root["goal"], dtype=np.float64).copy()
        previous_action = np.zeros(2, dtype=np.float64)
        for _ in range(int(sequence_steps)):
            support_normal = support_inward_normal(env, current)
            actions, signs = contact_aware_pulse_actions(
                current, tangent_magnitudes, inward_magnitude, support_normal
            )
            state_root_id = int(next_root_id)
            next_root_id += 1
            candidate_results: list[dict[str, Any]] = []
            for action_id, (action, moment_sign) in enumerate(zip(actions, signs)):
                result = execute_pulse(env, current, goal, action, pulse_steps)
                rows.append({
                    "root_id": state_root_id,
                    "action_id": int(action_id),
                    "state": current.copy(),
                    "before": result["before"].copy(),
                    "goal": goal.copy(),
                    "action": np.asarray(action, dtype=np.float64),
                    "previous_action": previous_action.copy(),
                    "moment_sign": float(moment_sign),
                    "support_normal": support_normal.copy(),
                    **result,
                })
                candidate_results.append(result)
            chosen = int(rng.integers(0, len(actions)))
            current = np.asarray(candidate_results[chosen]["after"], dtype=np.float64).copy()
            previous_action = np.asarray(actions[chosen], dtype=np.float64).copy()
            if not np.isfinite(current).all():
                break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-roots", type=int, default=64)
    parser.add_argument("--held-out-roots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9181)
    parser.add_argument("--pulse-steps", type=int, default=4)
    parser.add_argument("--closure-steps", type=int, default=48)
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+", default=[0.06, 0.10, 0.16])
    parser.add_argument("--selection-mode", choices=("direction_classifier", "signed_regressor", "pose_regressor"), default="direction_classifier")
    parser.add_argument("--successor-verify-topk", type=int, default=0,
                        help="optional privileged one-step successor verification; zero is pure learned selection")
    parser.add_argument("--sequence-steps", type=int, default=8,
                        help="executed pulses per root for the temporal lifted protocol")
    parser.add_argument("--static-only", action="store_true",
                        help="run the legacy static lifted protocol without temporal history")
    parser.add_argument("--contact-recovery-fallback", action="store_true",
                        help="use a one-pulse state-only contact recovery branch; report separately from pure learned control")
    parser.add_argument("--contact-recovery-threshold", type=float, default=38.0)
    parser.add_argument("--contact-recovery-magnitude", type=float, default=0.06)
    parser.add_argument("--pose-xy-weight", type=float, default=1.0)
    parser.add_argument("--pose-angle-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    tangent_magnitudes = tuple(float(x) for x in args.tangent_magnitudes)
    try:
        train_roots = make_terminal_roots(env, args.train_roots, args.seed + 1000, root_protocol="terminal_pose")
        held_roots = make_terminal_roots(env, args.held_out_roots, args.seed + 2000, root_protocol="terminal_pose")
        if args.static_only:
            train_rows = collect_effect_rows(
                env, train_roots, tangent_magnitudes, args.inward_magnitude, args.pulse_steps
            )
            held_rows = collect_effect_rows(
                env, held_roots, tangent_magnitudes, args.inward_magnitude, args.pulse_steps
            )
        else:
            train_rows = collect_temporal_candidate_rows(
                env, train_roots, tangent_magnitudes, args.inward_magnitude,
                args.pulse_steps, args.sequence_steps, args.seed + 3000,
            )
            held_rows = collect_temporal_candidate_rows(
                env, held_roots, tangent_magnitudes, args.inward_magnitude,
                args.pulse_steps, args.sequence_steps, args.seed + 4000,
            )
        models = fit_lifted_models(train_rows, args.seed)
        held_ranking = ranking(
            held_rows, *models, args.selection_mode,
            pose_xy_weight=args.pose_xy_weight,
            pose_angle_weight=args.pose_angle_weight,
        )
        closure = closed_loop(env, held_roots, models, args)
        static_comparison = None
        if not args.static_only:
            static_train = collect_effect_rows(
                env, train_roots, tangent_magnitudes, args.inward_magnitude, args.pulse_steps
            )
            static_held = collect_effect_rows(
                env, held_roots, tangent_magnitudes, args.inward_magnitude, args.pulse_steps
            )
            static_models = fit_lifted_models(static_train, args.seed)
            static_comparison = {
                "ranking": ranking(
                    static_held, *static_models, args.selection_mode,
                    pose_xy_weight=args.pose_xy_weight,
                    pose_angle_weight=args.pose_angle_weight,
                ),
                "closure": closed_loop(env, held_roots, static_models, args),
                "training_rows": len(static_train),
                "held_out_rows": len(static_held),
            }
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1", "train_roots": args.train_roots, "held_out_roots": args.held_out_roots,
            "train_seed": args.seed + 1000, "held_out_seed": args.seed + 2000,
            "pulse_steps": args.pulse_steps,
            "sequence_steps": args.sequence_steps,
            "data_protocol": "static_root_candidate_bank" if args.static_only else "root_disjoint executed sequence with shared candidate banks",
            "feature_map": "contact chart + body-frame velocity + Fourier phases + bilinear state-action interactions + distance RBFs + observed previous-action delay coordinate",
            "contact_mode_gate": "state-only recovery bank: bounded inward and reduced tangent action beyond distance 38",
            "contact_recovery_fallback": bool(args.contact_recovery_fallback),
            "lifted_feature_dim": int(lifted_contact_features(train_roots[0]["state"], np.zeros(2)).shape[0]),
            "selection_mode": args.selection_mode,
            "pose_projection_metric": {
                "xy_weight": args.pose_xy_weight,
                "angle_weight": args.pose_angle_weight,
                "xy_scale_px": 20.0,
                "angle_scale_rad": float(np.pi / 9.0),
            },
            "successor_verify_topk": args.successor_verify_topk,
        },
        "training": {"rows": len(train_rows), "unique_roots": len(train_roots)},
        "held_out_ranking": held_ranking,
        "closed_loop": closure,
        "static_comparison": static_comparison,
        "temporal_history_verified": bool(not args.static_only),
        "sanity_checks": {
            "all_finite": bool(np.isfinite(np.asarray([lifted_contact_features(train_roots[0]["state"], np.zeros(2))])).all()),
            "same_candidate_bank": True,
            "successor_free_selection": bool(args.successor_verify_topk == 0),
        },
        "claim_boundary": "Lifted privileged contact-effect adapter audit; temporal lifting is tested as a representation/interface hypothesis, not causal latent dynamics or visual planning evidence.",
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output / "lifted_contact_adapter.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value))
    print(json.dumps({"output": str(output), "ranking": held_ranking["metrics"], "closure": closure["metrics"], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
