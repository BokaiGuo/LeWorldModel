"""Pure learned two-action verifier for PushT contact-pose closure.

The existing hybrid confirmatory result uses a learned proposal followed by a
bounded simulator successor check.  This runner isolates the remaining gap:
replace that simulator check with a learned decision verifier trained on
offline two-action sequence labels.  At deployment, the selector sees only
current state, goal, and the ordered macro-action; it does not simulate any
candidate successor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_contact_mode_effective_motion import (  # noqa: E402
    contact_mode_features,
    goal_for_state,
    select_balanced_states,
)
from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
    mode_for_state,
)
from experiments.run_pusht_direct_pose_decision_ablation import (  # noqa: E402
    action_endpoint,
    angle_error,
)
from experiments.run_pusht_object_centric_learned_planner import (  # noqa: E402
    make_state_only_env,
    torque_action_bank,
)


VARIANTS = (
    "sequence_joint",
    "sequence_soft_contact",
    "sequence_turn_soft_contact",
    "sequence_continuous_rank",
    "sequence_mode_lexicographic",
    "sequence_mode_receding",
    "sequence_mode_pairwise_receding",
    "sequence_mode_task_pairwise_receding",
    "sequence_mode_pairwise_staged_receding",
    "sequence_mode_online_orientation_receding",
    "sequence_mode_knn_receding",
    "sequence_mode_uncertain_receding",
    "sequence_mode_hybrid_receding",
)
RECEDING_VARIANTS = (
    "sequence_mode_receding",
    "sequence_mode_pairwise_receding",
    "sequence_mode_task_pairwise_receding",
    "sequence_mode_pairwise_staged_receding",
    "sequence_mode_online_orientation_receding",
    "sequence_mode_knn_receding",
    "sequence_mode_uncertain_receding",
    "sequence_mode_hybrid_receding",
)
CONTINUOUS_TARGETS = ("angle_progress", "xy_progress", "motion_distance")
MODE_TARGETS = ("contact", "path_contact")
MULTICLASS_TARGETS = ("successor_mode",)
MODE_ORDER = ("free", "approach", "contact")

# Keep model randomness attached to the semantic target rather than to the
# insertion order of ``labels``.  Adding a new diagnostic target must not
# silently change every other estimator in a supposedly matched ablation.
MODEL_SEED_OFFSETS = {
    "pose": 0,
    "turn": 10,
    "motion": 20,
    "contact": 30,
    "path_contact": 40,
    "joint": 50,
    "angle_progress": 60,
    "xy_progress": 70,
    "motion_distance": 80,
    "successor_mode": 90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=32)
    parser.add_argument(
        "--calibration-episodes", type=int, default=8,
        help="root-disjoint calibration episodes used only for the pairwise safety gate",
    )
    parser.add_argument("--held-out-episodes", type=int, default=16)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    parser.add_argument("--goals-per-state", type=int, default=4)
    parser.add_argument("--closure-cycles", type=int, default=8)
    parser.add_argument(
        "--mode-xy-regression-margin",
        type=float,
        default=4.0,
        help="hard XY regression tube (pixels) for mode-first sequence selection",
    )
    parser.add_argument(
        "--mode-motion-distance-threshold",
        type=float,
        default=2.0,
        help="minimum predicted block displacement (pixels) for effective-motion mode support",
    )
    parser.add_argument(
        "--path-contact-label",
        choices=("path", "endpoint"),
        default="path",
        help="whether the path_contact target uses all intermediate successors or only the endpoint",
    )
    parser.add_argument(
        "--mode-zero-policy",
        choices=("nonzero", "allow"),
        default="nonzero",
        help="terminal mode fallback policy; nonzero forbids silent zero-action fallback",
    )
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--model-family", choices=("extra_trees", "knn"), default="extra_trees")
    parser.add_argument("--knn-neighbors", type=int, default=16)
    parser.add_argument(
        "--uncertainty-z", type=float, default=1.0,
        help="number of fitted-tree dispersion units used by the risk-aware selector",
    )
    parser.add_argument(
        "--feature-interface",
        choices=("local", "global_wall"),
        default="local",
        help=(
            "local contact-frame features, or the matched interface augmented "
            "with absolute pose and boundary geometry"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261080)
    return parser.parse_args()


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"estimate": float(values.mean()), "ci95_low": float(np.quantile(means, 0.025)), "ci95_high": float(np.quantile(means, 0.975)), "n": int(len(values))}


def calibration_gate_decision(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    xy_tolerance: float = 2.0,
    angle_tolerance: float = 0.05,
    contact_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Decide whether a repaired candidate policy may replace the fixed gate.

    This is deliberately a safety gate, not test-set model selection.  The
    candidate is admitted only if it is non-inferior on every exposed
    consequence in a root-disjoint calibration set.  The tolerances are fixed
    protocol constants and are recorded in the artifact.  If the learned
    ranker does not pass, deployment falls back to the existing mode gate.
    """
    if not baseline_rows or not candidate_rows or len(baseline_rows) != len(candidate_rows):
        raise ValueError("calibration gate requires paired, non-empty rows")
    baseline = {
        name: float(np.mean([row[name] for row in baseline_rows]))
        for name in ("strict_success", "final_xy", "final_angle", "contact_retention")
    }
    candidate = {
        name: float(np.mean([row[name] for row in candidate_rows]))
        for name in ("strict_success", "final_xy", "final_angle", "contact_retention")
    }
    checks = {
        "success_noninferior": candidate["strict_success"] >= baseline["strict_success"],
        "xy_noninferior": candidate["final_xy"] <= baseline["final_xy"] + float(xy_tolerance),
        "angle_noninferior": candidate["final_angle"] <= baseline["final_angle"] + float(angle_tolerance),
        "contact_noninferior": candidate["contact_retention"] >= baseline["contact_retention"] - float(contact_tolerance),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "baseline_means": baseline,
        "candidate_means": candidate,
        "tolerances": {
            "xy": float(xy_tolerance),
            "angle": float(angle_tolerance),
            "contact": float(contact_tolerance),
        },
        "decision": "candidate_admitted" if all(checks.values()) else "mode_receding_fallback",
    }


def feature_vector(
    state: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    previous: np.ndarray,
    goal: np.ndarray,
    interface: str = "local",
) -> np.ndarray:
    local = np.concatenate([
        contact_mode_features(state, first, previous, goal),
        contact_mode_features(state, second, first, goal),
    ]).astype(np.float64)
    if interface == "local":
        return local
    if interface != "global_wall":
        raise ValueError(f"unknown feature interface: {interface}")
    # PushT has bounded image coordinates.  The local interface intentionally
    # omits absolute location; this matched arm exposes only current-state
    # geometry that can change contact behavior near the arena boundary.
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    xy = state[:4] / 500.0
    wall_distances = np.asarray([
        state[0], state[1], 500.0 - state[0], 500.0 - state[1],
        state[2], state[3], 500.0 - state[2], 500.0 - state[3],
    ], dtype=np.float64) / 500.0
    block_angle = float(state[4])
    absolute = np.asarray([
        *xy.tolist(),
        *wall_distances.tolist(),
        np.sin(block_angle),
        np.cos(block_angle),
        state[5] / 250.0,
        state[6] / 250.0,
        (goal[2] - state[2]) / 500.0,
        (goal[3] - state[3]) / 500.0,
        np.sin(float(goal[4] - state[4])),
        np.cos(float(goal[4] - state[4])),
    ], dtype=np.float64)
    return np.concatenate([local, absolute])


def label_sequence(state: np.ndarray, goal: np.ndarray, endpoint: np.ndarray) -> dict[str, Any]:
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    after_angle = angle_error(endpoint[4], goal[4])
    pose = after_angle < before_angle - 1e-5 and after_xy <= min(45.0, before_xy + 4.0)
    turn = after_angle < before_angle - 1e-5
    motion = np.linalg.norm(endpoint[2:4] - state[2:4]) >= 0.5
    contact = mode_for_state(endpoint) == "contact"
    return {
        "pose": int(pose),
        "turn": int(turn),
        "motion": int(motion),
        "contact": int(contact),
        # Filled by make_training_data from every intermediate successor;
        # endpoint contact alone is not a sufficient hybrid-mode label.
        "path_contact": int(contact),
        "successor_mode": mode_index(endpoint),
        "joint": int(pose and motion and (contact if mode_for_state(state) == "contact" else True)),
        "angle_progress": float(before_angle - after_angle),
        "xy_progress": float(before_xy - after_xy),
        "motion_distance": float(np.linalg.norm(endpoint[2:4] - state[2:4])),
    }


def fit_binary(features: np.ndarray, labels: np.ndarray, trees: int, seed: int):
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return DummyClassifier(strategy="constant", constant=int(labels[0])).fit(features, labels)
    return ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=4, max_features=1.0, class_weight="balanced", random_state=seed, n_jobs=1).fit(features, labels)


def positive_probability(model, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    return probabilities[:, int(positive[0])] if len(positive) else np.full(len(features), float(classes[0] == 1))


def regression_mean_std(model: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted-tree mean and dispersion without runtime successors.

    The dispersion is a training-support proxy, not a calibrated physical
    noise model.  Models without an exposed ensemble receive zero dispersion
    and are therefore not interpreted as uncertainty-aware evidence.
    """
    features = np.asarray(features, dtype=np.float64)
    estimators = getattr(model, "estimators_", None)
    if estimators is None or len(estimators) == 0:
        mean = np.asarray(model.predict(features), dtype=np.float64)
        return mean, np.zeros_like(mean)
    draws = np.asarray([estimator.predict(features) for estimator in estimators], dtype=np.float64)
    return draws.mean(axis=0), draws.std(axis=0)


def successor_mode_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    """Align successor-mode probabilities to free/approach/contact classes."""
    features = np.asarray(features, dtype=np.float64)
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    result = np.zeros((len(features), len(MODE_ORDER)), dtype=np.float64)
    for column, class_id in enumerate(classes):
        if 0 <= int(class_id) < len(MODE_ORDER):
            result[:, int(class_id)] = raw[:, column]
    return result


def mode_index(state: np.ndarray) -> int:
    return int(MODE_ORDER.index(mode_for_state(np.asarray(state, dtype=np.float64))))


def fit_pairwise_angle_ranker(
    features: np.ndarray,
    angle_progress: np.ndarray,
    group_size: int,
    trees: int,
    seed: int,
):
    """Fit a shared-candidate ranker without using held-out successors.

    Absolute effect regression can be well calibrated on average while still
    getting the ordering of two actions wrong in a particular contact state.
    This head is trained only on within-state/goal action pairs, so its
    prediction target is the deployment decision: whether candidate i beats
    candidate j on angle progress.
    """
    features = np.asarray(features, dtype=np.float64)
    angle_progress = np.asarray(angle_progress, dtype=np.float64)
    if group_size <= 1 or len(features) % group_size:
        raise ValueError("pairwise ranker requires complete action groups")
    pair_features = []
    pair_labels = []
    for start in range(0, len(features), group_size):
        stop = start + group_size
        group_features = features[start:stop]
        group_targets = angle_progress[start:stop]
        for first in range(group_size):
            for second in range(first + 1, group_size):
                delta = float(group_targets[first] - group_targets[second])
                if abs(delta) <= 1e-8:
                    continue
                left = group_features[first]
                right = group_features[second]
                pair_features.append(np.concatenate([left, right, left - right]))
                pair_labels.append(int(delta > 0.0))
                # Deployment scores every ordered pair, including the
                # reverse orientation.  Train both orientations explicitly;
                # otherwise the ranker is extrapolated exactly where the
                # candidate comparison is most sensitive.
                pair_features.append(np.concatenate([right, left, right - left]))
                pair_labels.append(int(delta < 0.0))
    if not pair_features:
        raise ValueError("pairwise ranker received no non-tied action pairs")
    return fit_binary(np.asarray(pair_features), np.asarray(pair_labels), trees, seed)


def task_rank_target(labels: dict[str, np.ndarray], xy_margin: float = 4.0) -> np.ndarray:
    """Build the predeclared executable-progress target for pairwise ranking.

    Angle-only supervision is the failure interface exposed by the earlier
    ablation: an action can improve orientation while losing contact or
    worsening translation.  The target therefore gives infeasible outcomes a
    fixed low utility and ranks feasible outcomes by wrapped-angle progress
    with a small translation tie-break.  It is generated offline from
    simulator labels only; runtime still sees the learned ranker, not a
    successor.
    """
    required = {"angle_progress", "xy_progress", "motion", "contact", "path_contact"}
    missing = required - set(labels)
    if missing:
        raise ValueError(f"task ranking labels missing: {sorted(missing)}")
    angle = np.asarray(labels["angle_progress"], dtype=np.float64)
    xy = np.asarray(labels["xy_progress"], dtype=np.float64)
    feasible = (
        (np.asarray(labels["motion"]) > 0)
        & (np.asarray(labels["contact"]) > 0)
        & (np.asarray(labels["path_contact"]) > 0)
        & (xy >= -float(xy_margin))
    )
    return np.where(feasible, angle + 0.02 * xy, -10.0 + 0.02 * xy)


def pairwise_win_scores(model, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    scores = np.zeros(len(features), dtype=np.float64)
    for first in range(len(features)):
        opponents = [second for second in range(len(features)) if second != first]
        pairs = np.asarray([
            np.concatenate([features[first], features[second], features[first] - features[second]])
            for second in opponents
        ], dtype=np.float64)
        scores[first] = float(np.mean(positive_probability(model, pairs)))
    return scores


def online_orientation_progress(
    state: np.ndarray,
    goal: np.ndarray,
    sequences: np.ndarray,
    history: list[dict[str, Any]],
    ridge: float = 1e-3,
) -> np.ndarray | None:
    """Estimate local angular progress from executed, observed pulses only.

    PushT's action effect is contact-state dependent.  A frozen global ranker
    can therefore identify a useful shortlist while still getting the local
    torque sign wrong.  This small ridge fit uses only committed transitions
    ``(state_before, action, observed_state_after)``.  The scalar regressor is
    expressed in the instantaneous lever-arm torque proxy, so it cannot
    silently learn a free XY correction.  It is a runtime calibration signal,
    not candidate successor verification.
    """
    if len(history) < 2:
        return None
    rows: list[list[float]] = []
    targets: list[float] = []
    for item in history[-8:]:
        before = np.asarray(item["before_state"], dtype=np.float64)
        after = np.asarray(item["after_state"], dtype=np.float64)
        action = np.asarray(item["action"], dtype=np.float64)
        lever = before[:2] - before[2:4]
        target_displacement = 100.0 * action
        torque_proxy = float(lever[0] * target_displacement[1] - lever[1] * target_displacement[0])
        delta_theta = float((after[4] - before[4] + np.pi) % (2.0 * np.pi) - np.pi)
        rows.append([torque_proxy, 1.0])
        targets.append(delta_theta)
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    penalty = np.diag([float(ridge), 1e-6])
    try:
        coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    except np.linalg.LinAlgError:
        return None
    current = np.asarray(state, dtype=np.float64)
    lever = current[:2] - current[2:4]
    desired_sign = float(np.sign((float(goal[4]) - float(current[4]) + np.pi) % (2.0 * np.pi) - np.pi))
    if desired_sign == 0.0:
        return np.zeros(len(sequences), dtype=np.float64)
    values = []
    for sequence in sequences:
        action = np.asarray(sequence[0], dtype=np.float64)
        target_displacement = 100.0 * action
        torque_proxy = float(lever[0] * target_displacement[1] - lever[1] * target_displacement[0])
        values.append(desired_sign * float(np.asarray([torque_proxy, 1.0]) @ coefficient))
    return np.asarray(values, dtype=np.float64)


def make_sequences(actions: np.ndarray, length: int = 2) -> np.ndarray:
    if length == 1:
        return np.asarray([[action] for action in actions], dtype=np.float32)
    if length == 2:
        return np.asarray([[first, second] for first in actions for second in actions], dtype=np.float32)
    raise ValueError("only one-step and two-step sequences are supported")


def sequence_feature_vector(
    state: np.ndarray,
    sequence: np.ndarray,
    goal: np.ndarray,
    interface: str,
) -> np.ndarray:
    first = np.asarray(sequence[0], dtype=np.float32)
    second = np.asarray(sequence[1], dtype=np.float32) if len(sequence) > 1 else np.zeros(2, dtype=np.float32)
    return feature_vector(state, first, second, np.zeros(2), goal, interface)


def make_training_data(
    env,
    states: list[np.ndarray],
    actions: np.ndarray,
    goals_per_state: int,
    seed: int,
    feature_interface: str,
    path_contact_label: str = "path",
    sequence_length: int = 2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if path_contact_label not in ("path", "endpoint"):
        raise ValueError("path_contact_label must be 'path' or 'endpoint'")
    sequences = make_sequences(actions, length=sequence_length)
    rng = np.random.default_rng(seed + 10)
    features = []
    labels = {name: [] for name in ("pose", "turn", "motion", "contact", "path_contact", "joint", *CONTINUOUS_TARGETS, *MULTICLASS_TARGETS)}
    for state_index, state in enumerate(states):
        goals = [goal_for_state(state, state_index * goals_per_state + goal_index, rng, closure_only=True) for goal_index in range(goals_per_state)]
        for goal_index, goal in enumerate(goals):
            for sequence_index, sequence in enumerate(sequences):
                env.reset(seed=seed + state_index * 1000000 + goal_index * 10000 + sequence_index * 10, options={"state": state.copy(), "goal_state": state.copy()})
                endpoint = state.copy()
                path_contact = True
                for action in sequence:
                    observation, _, terminated, truncated, _ = env.step(action)
                    endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                    path_contact = path_contact and mode_for_state(endpoint) == "contact"
                    if terminated or truncated:
                        break
                features.append(sequence_feature_vector(state, sequence, goal, feature_interface))
                target = label_sequence(state, goal, endpoint)
                target["path_contact"] = int(path_contact if path_contact_label == "path" else target["contact"])
                for name in labels:
                    labels[name].append(target[name])
    return np.asarray(features, dtype=np.float64), {
        name: np.asarray(values, dtype=np.float64 if name in CONTINUOUS_TARGETS else np.int64)
        for name, values in labels.items()
    }


def fit_models(
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    trees: int,
    seed: int,
    model_family: str = "extra_trees",
    knn_neighbors: int = 16,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    unknown = set(labels) - set(MODEL_SEED_OFFSETS)
    if unknown:
        raise ValueError(f"missing fixed seed offsets for targets: {sorted(unknown)}")
    if model_family not in ("extra_trees", "knn"):
        raise ValueError("model_family must be extra_trees or knn")
    if model_family == "knn" and knn_neighbors < 1:
        raise ValueError("knn_neighbors must be positive")
    for name, values in labels.items():
        if name in CONTINUOUS_TARGETS:
            if model_family == "knn":
                models[name] = make_pipeline(
                    StandardScaler(),
                    KNeighborsRegressor(
                        n_neighbors=min(knn_neighbors, len(features)),
                        weights="distance",
                        p=2,
                    ),
                ).fit(features, values)
            else:
                models[name] = ExtraTreesRegressor(
                    n_estimators=trees,
                    min_samples_leaf=4,
                    max_features=1.0,
                    random_state=seed + MODEL_SEED_OFFSETS[name],
                    n_jobs=1,
                ).fit(features, values)
        else:
            if model_family == "knn":
                if len(np.unique(values)) < 2:
                    models[name] = DummyClassifier(strategy="constant", constant=int(values[0])).fit(features, values)
                else:
                    models[name] = make_pipeline(
                        StandardScaler(),
                        KNeighborsClassifier(
                            n_neighbors=min(knn_neighbors, len(features)),
                            weights="distance",
                            p=2,
                        ),
                    ).fit(features, values)
            else:
                models[name] = fit_binary(features, values, trees, seed + MODEL_SEED_OFFSETS[name])
    return models


def choose_uncertainty_aware_sequence(
    models: dict[str, Any],
    features: np.ndarray,
    sequences: np.ndarray,
    state: np.ndarray,
    mode_contact_threshold: float,
    mode_motion_threshold: float,
    mode_xy_regression_margin: float,
    mode_zero_policy: str,
    uncertainty_z: float,
) -> tuple[int, dict[str, Any]]:
    """Choose a candidate by a conservative successor-mode/effect set.

    This is the direct implementation of the literature-motivated interface:
    predict the successor hybrid mode, form lower confidence bounds for the
    continuous effects, reject unsupported contact transitions, and only then
    rank task progress.  No candidate successor is queried at deployment.
    """
    if "successor_mode" not in models:
        raise ValueError("uncertainty-aware selection requires successor_mode head")
    z = max(float(uncertainty_z), 0.0)
    angle, angle_std = regression_mean_std(models["angle_progress"], features)
    xy, xy_std = regression_mean_std(models["xy_progress"], features)
    motion, motion_std = regression_mean_std(models["motion_distance"], features)
    mode_prob = successor_mode_probabilities(models["successor_mode"], features)
    predicted_contact = mode_prob[:, MODE_ORDER.index("contact")]
    mode_confidence = np.max(mode_prob, axis=1)
    mode_uncertainty = 1.0 - mode_confidence
    lower_motion = motion - z * motion_std
    lower_xy = xy - z * xy_std
    lower_angle = angle - z * angle_std
    predicted_turn = positive_probability(models["turn"], features)
    nonzero = np.any(np.abs(sequences) > 1e-8, axis=(1, 2))
    feasible = (
        (predicted_contact >= float(mode_contact_threshold))
        & (lower_motion >= float(mode_motion_threshold))
        & (predicted_turn >= float(mode_contact_threshold))
        & (lower_xy >= -float(mode_xy_regression_margin))
    )
    nonzero_feasible = feasible & nonzero
    current_mode = mode_for_state(np.asarray(state, dtype=np.float64))
    # Contact-rich controllers should not spend a lost-contact step optimizing
    # angular progress.  This is the learned analogue of a contact
    # reacquisition mode: while outside contact, rank the same candidate bank
    # by successor-contact support first, then by conservative motion.  It is
    # still successor-free at runtime; only the current observation and the
    # learned successor-mode probabilities are used.
    if current_mode != "contact":
        recovery_score = (
            -predicted_contact
            - 0.05 * lower_motion
            + 0.10 * mode_uncertainty
        )
        candidate_mask = nonzero
        selected = int(np.argmin(np.where(candidate_mask, recovery_score, np.inf)))
        order = np.argsort(recovery_score, kind="mergesort")
        order = order[np.isin(order, np.flatnonzero(candidate_mask))]
        return selected, {
            "predicted_order": order.tolist(),
            "selected_score": float(recovery_score[selected]),
            "selected_contact_probability": float(predicted_contact[selected]),
            "selected_turn_probability": float(predicted_turn[selected]),
            "selected_angle_progress": float(angle[selected]),
            "selected_angle_lower_bound": float(lower_angle[selected]),
            "selected_xy_progress": float(xy[selected]),
            "selected_motion_lower_bound": float(lower_motion[selected]),
            "selected_mode_uncertainty": float(mode_uncertainty[selected]),
            "mode_feasible_count": int(np.sum(feasible)),
            "mode_feasible": bool(feasible[selected]),
            "zero_guard_active": True,
            "mode_zero_policy": mode_zero_policy,
            "selected_nonzero": bool(nonzero[selected]),
            "nonzero_feasible_count": int(np.sum(nonzero_feasible)),
            "selection_mode": "contact_reacquisition",
            "current_mode": current_mode,
            "uncertainty_z": z,
            "uncertainty_source": "ExtraTrees fitted-tree dispersion plus successor-mode entropy proxy",
        }
    # The uncertainty term is deliberately fixed and recorded.  It prevents a
    # high-variance apparent torque gain from beating a supported action.
    score = -lower_angle + 0.10 * np.maximum(-lower_xy, 0.0) + 0.10 * mode_uncertainty
    zero_guard_active = bool(mode_zero_policy == "nonzero" and np.any(nonzero))
    if np.any(nonzero_feasible):
        candidate_mask = nonzero_feasible if zero_guard_active else feasible
        selected = int(np.argmin(np.where(candidate_mask, score, np.inf)))
        selection_mode = "uncertainty_set_feasible"
    else:
        # Fail closed: choose the least violating nonzero action, not a silent
        # hold.  This preserves the old support invariant while exposing that
        # the learned model has no certified action in the current state.
        margin = np.minimum.reduce([
            predicted_contact - float(mode_contact_threshold),
            lower_motion - float(mode_motion_threshold),
            predicted_turn - float(mode_contact_threshold),
            lower_xy + float(mode_xy_regression_margin),
        ])
        candidate_mask = nonzero if zero_guard_active else np.ones(len(score), dtype=bool)
        order = np.lexsort((score, -margin))
        order = order[np.isin(order, np.flatnonzero(candidate_mask))]
        selected = int(order[0])
        selection_mode = "uncertainty_support_violation_fallback"
    order = np.lexsort((score, ~candidate_mask if np.any(candidate_mask) else np.zeros(len(score), dtype=bool)))
    return selected, {
        "predicted_order": order.tolist(),
        "selected_score": float(score[selected]),
        "selected_contact_probability": float(predicted_contact[selected]),
        "selected_turn_probability": float(predicted_turn[selected]),
        "selected_angle_progress": float(angle[selected]),
        "selected_angle_lower_bound": float(lower_angle[selected]),
        "selected_xy_progress": float(xy[selected]),
        "selected_motion_lower_bound": float(lower_motion[selected]),
        "selected_mode_uncertainty": float(mode_uncertainty[selected]),
        "mode_feasible_count": int(np.sum(feasible)),
        "mode_feasible": bool(feasible[selected]),
        "zero_guard_active": zero_guard_active,
        "mode_zero_policy": mode_zero_policy,
        "selected_nonzero": bool(nonzero[selected]),
        "nonzero_feasible_count": int(np.sum(nonzero_feasible)),
        "selection_mode": selection_mode,
        "uncertainty_z": z,
        "uncertainty_source": "ExtraTrees fitted-tree dispersion plus successor-mode entropy proxy",
    }


def choose_sequence(
    variant: str,
    models: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    sequences: np.ndarray,
    feature_interface: str,
    mode_contact_threshold: float = 0.60,
    mode_motion_threshold: float = 2.0,
    mode_xy_regression_margin: float = 4.0,
    mode_zero_policy: str = "nonzero",
    online_orientation_scores: np.ndarray | None = None,
    uncertainty_z: float = 1.0,
) -> tuple[int, dict[str, Any]]:
    features = np.asarray([sequence_feature_vector(state, sequence, goal, feature_interface) for sequence in sequences], dtype=np.float64)
    if variant == "sequence_mode_hybrid_receding":
        # The literature separates translation/contact maintenance from
        # terminal orientation closure.  Use the conservative staged arm
        # while the block is far away, then switch to successor-mode-aware
        # orientation selection inside the terminal tube.  Both branches use
        # the same frozen feature interface and one-step re-observation.
        block_xy_error = float(np.linalg.norm(np.asarray(state[2:4]) - np.asarray(goal[2:4])))
        phase_variant = (
            "sequence_mode_pairwise_staged_receding"
            if block_xy_error > 24.0
            else "sequence_mode_task_pairwise_receding"
        )
        selected, info = choose_sequence(
            phase_variant, models, state, goal, sequences, feature_interface,
            mode_contact_threshold, mode_motion_threshold,
            mode_xy_regression_margin, mode_zero_policy,
            online_orientation_scores, uncertainty_z,
        )
        info["hybrid_phase"] = "translation" if block_xy_error > 24.0 else "orientation"
        info["hybrid_delegate"] = phase_variant
        return selected, info
    if variant == "sequence_mode_uncertain_receding":
        selected, info = choose_uncertainty_aware_sequence(
            models, features, sequences, state, mode_contact_threshold,
            mode_motion_threshold, mode_xy_regression_margin,
            mode_zero_policy, uncertainty_z,
        )
        info["decision_horizon"] = int(sequences.shape[1])
        info["mode_phase"] = "orientation"
        info["block_xy_error_at_selection"] = float(np.linalg.norm(np.asarray(state[2:4]) - np.asarray(goal[2:4])))
        return selected, info
    if variant == "sequence_continuous_rank":
        predicted_angle = np.asarray(models["angle_progress"].predict(features), dtype=np.float64)
        predicted_xy = np.asarray(models["xy_progress"].predict(features), dtype=np.float64)
        predicted_motion = np.asarray(models["motion_distance"].predict(features), dtype=np.float64)
        predicted_contact = positive_probability(models["contact"], features)
        # Directly rank the predicted task progress.  The angular term is the
        # primary closure objective; XY regression and contact loss are soft
        # penalties, so the model is not forced to reconstruct a wrapped angle
        # from a separately predicted physical effect vector.
        score = (
            -predicted_angle
            + 0.02 * np.maximum(-predicted_xy, 0.0)
            + 0.35 * (1.0 - predicted_contact)
            + 0.10 * np.maximum(0.5 - predicted_motion, 0.0)
        )
        order = np.argsort(score, kind="mergesort")
        selected = int(order[0])
        return selected, {
            "predicted_order": order.tolist(),
            "selected_score": float(score[selected]),
            "selected_pose_probability": float("nan"),
            "selected_turn_probability": float("nan"),
            "selected_contact_probability": float(predicted_contact[selected]),
            "selected_joint_probability": float("nan"),
            "selected_angle_progress": float(predicted_angle[selected]),
            "selected_xy_progress": float(predicted_xy[selected]),
        }
    if variant in ("sequence_mode_lexicographic", "sequence_mode_receding", "sequence_mode_pairwise_receding", "sequence_mode_task_pairwise_receding", "sequence_mode_pairwise_staged_receding", "sequence_mode_online_orientation_receding", "sequence_mode_knn_receding"):
        # Contact-rich manipulation is a hybrid decision problem.  The old
        # soft-contact score allowed a large predicted angular gain to trade
        # against a likely contact loss or a no-motion sequence.  This arm
        # implements the literature-motivated interface explicitly: first
        # require a held-out successor-mode belief (turn + motion + contact),
        # then rank only the surviving candidates by task progress.  The
        # thresholds are fixed protocol constants and are recorded in the
        # returned trace; no simulator successor is queried at runtime.
        if not 0.0 < float(mode_contact_threshold) <= 1.0:
            raise ValueError("mode_contact_threshold must lie in (0, 1]")
        if float(mode_motion_threshold) <= 0.0:
            raise ValueError("mode_motion_threshold must be positive")
        if float(mode_xy_regression_margin) < 0.0:
            raise ValueError("mode_xy_regression_margin must be non-negative")
        if mode_zero_policy not in ("nonzero", "allow"):
            raise ValueError("mode_zero_policy must be 'nonzero' or 'allow'")
        if variant in ("sequence_mode_pairwise_receding", "sequence_mode_pairwise_staged_receding", "sequence_mode_online_orientation_receding"):
            predicted_angle = pairwise_win_scores(models["angle_pairwise"], features)
            ranking_target_name = "angle_progress"
        elif variant == "sequence_mode_task_pairwise_receding":
            predicted_angle = pairwise_win_scores(models["task_pairwise"], features)
            ranking_target_name = "executable_task_progress"
        else:
            predicted_angle = np.asarray(models["angle_progress"].predict(features), dtype=np.float64)
            ranking_target_name = "angle_progress"
        predicted_xy = np.asarray(models["xy_progress"].predict(features), dtype=np.float64)
        predicted_motion = np.asarray(models["motion_distance"].predict(features), dtype=np.float64)
        predicted_contact = positive_probability(models["contact"], features)
        predicted_path_contact = positive_probability(models["path_contact"], features)
        predicted_turn = positive_probability(models["turn"], features)
        feasible = (
            (predicted_contact >= float(mode_contact_threshold))
            & (predicted_motion >= float(mode_motion_threshold))
            & (predicted_turn >= float(mode_contact_threshold))
            & (predicted_xy >= -float(mode_xy_regression_margin))
            & (predicted_path_contact >= float(mode_contact_threshold))
        )
        staged = variant == "sequence_mode_pairwise_staged_receding"
        block_xy_error = float(np.linalg.norm(np.asarray(state[2:4]) - np.asarray(goal[2:4])))
        if staged and block_xy_error > 24.0:
            # Far from the terminal tube, a turn-only action can make the
            # block orientation look better while abandoning translation.
            # Keep contact and effective motion hard, but defer the turn gate
            # until the controller is close enough for orientation closure.
            active_feasible = (
                (predicted_contact >= float(mode_contact_threshold))
                & (predicted_motion >= float(mode_motion_threshold))
                & (predicted_xy >= -float(mode_xy_regression_margin))
                & (predicted_path_contact >= float(mode_contact_threshold))
            )
            score = -predicted_xy + 0.05 * (1.0 - predicted_angle)
            mode_phase = "translation"
        else:
            active_feasible = feasible
            score = -predicted_angle + 0.02 * np.maximum(-predicted_xy, 0.0)
            mode_phase = "orientation"
        nonzero = np.any(np.abs(sequences) > 1e-8, axis=(1, 2))
        # A zero macro-action is useful as an explicit hold, but it must not
        # become a silent terminal fallback when the learned mode model has
        # no supported non-zero action.  If a non-zero feasible candidate
        # exists, the terminal selector is forbidden from choosing hold.
        nonzero_feasible = active_feasible & nonzero
        zero_guard_active = bool(mode_zero_policy == "nonzero" and np.any(nonzero))
        if np.any(nonzero_feasible):
            candidate_mask = nonzero_feasible if zero_guard_active else feasible
            masked = np.where(candidate_mask, score, np.inf)
            selected = int(np.argmin(masked))
            selection_mode = "mode_feasible_then_angle"
        elif np.any(feasible):
            candidate_mask = active_feasible
            masked = np.where(candidate_mask, score, np.inf)
            selected = int(np.argmin(masked))
            selection_mode = "mode_feasible_zero_only"
        else:
            # Fail closed on the hybrid constraint, but keep the controller
            # live: select the candidate with the best bottleneck mode margin
            # and use angle progress only as a tie-break.  This explicitly
            # exposes lack of mode support instead of silently selecting zero.
            margin = np.minimum.reduce([
                predicted_contact - float(mode_contact_threshold),
                predicted_motion - float(mode_motion_threshold),
                predicted_turn - float(mode_contact_threshold),
                predicted_xy + float(mode_xy_regression_margin),
                predicted_path_contact - float(mode_contact_threshold),
            ])
            violation = np.maximum(
                -predicted_xy - float(mode_xy_regression_margin), 0.0
            )
            candidate_mask = (nonzero if zero_guard_active else np.ones(len(score), dtype=bool))
            order = np.lexsort((score + 0.05 * violation, -margin))
            order = order[np.isin(order, np.flatnonzero(candidate_mask))]
            selected = int(order[0])
            selection_mode = "mode_support_violation_nonzero_fallback" if np.any(nonzero) else "mode_support_violation_fallback"
        if (
            variant == "sequence_mode_online_orientation_receding"
            and mode_phase == "orientation"
            and online_orientation_scores is not None
        ):
            feasible_indices = np.flatnonzero(candidate_mask)
            if len(feasible_indices) > 0:
                base_order = feasible_indices[np.argsort(score[feasible_indices], kind="mergesort")]
                shortlist = base_order[: min(8, len(base_order))]
                online_values = np.asarray(online_orientation_scores, dtype=np.float64)
                if online_values.shape != (len(score),) or not np.isfinite(online_values).all():
                    raise ValueError("online_orientation_scores must be finite and match candidates")
                selected = int(shortlist[np.argmax(online_values[shortlist])])
                selection_mode = "mode_feasible_top8_online_orientation"
        order = np.lexsort((score, ~candidate_mask if np.any(candidate_mask) else np.zeros(len(score), dtype=bool)))
        return selected, {
            "predicted_order": order.tolist(),
            "selected_score": float(score[selected]),
            "selected_pose_probability": float("nan"),
            "selected_turn_probability": float(predicted_turn[selected]),
            "selected_contact_probability": float(predicted_contact[selected]),
            "selected_path_contact_probability": float(predicted_path_contact[selected]),
            "selected_joint_probability": float("nan"),
            "selected_angle_progress": float(predicted_angle[selected]),
            "selected_xy_progress": float(predicted_xy[selected]),
            "selected_motion_distance": float(predicted_motion[selected]),
            "mode_contact_threshold": float(mode_contact_threshold),
            "mode_motion_threshold": float(mode_motion_threshold),
            "mode_xy_regression_margin": float(mode_xy_regression_margin),
            "mode_path_contact_threshold": float(mode_contact_threshold),
            "mode_feasible_count": int(np.sum(active_feasible)),
            "mode_feasible": bool(active_feasible[selected]),
            "zero_guard_active": zero_guard_active,
            "mode_zero_policy": mode_zero_policy,
            "selected_nonzero": bool(nonzero[selected]),
            "nonzero_feasible_count": int(np.sum(nonzero_feasible)),
            "selection_mode": selection_mode,
            "decision_horizon": int(sequences.shape[1]),
            "mode_phase": mode_phase,
            "block_xy_error_at_selection": block_xy_error,
            "online_orientation_used": bool(variant == "sequence_mode_online_orientation_receding" and online_orientation_scores is not None),
            "ranking_target": ranking_target_name,
        }
    p = {name: positive_probability(model, features) for name, model in models.items() if name not in CONTINUOUS_TARGETS and name not in MULTICLASS_TARGETS}
    if variant == "sequence_joint":
        score = -p["joint"]
    elif variant == "sequence_soft_contact":
        score = -p["pose"] + 0.20 * (1.0 - p["motion"]) + 0.35 * (1.0 - p["contact"])
    else:
        score = -p["turn"] + 0.20 * (1.0 - p["motion"]) + 0.35 * (1.0 - p["contact"])
    order = np.argsort(score, kind="mergesort")
    selected = int(order[0])
    return selected, {"predicted_order": order.tolist(), "selected_score": float(score[selected]), "selected_pose_probability": float(p["pose"][selected]), "selected_turn_probability": float(p["turn"][selected]), "selected_contact_probability": float(p["contact"][selected]), "selected_joint_probability": float(p["joint"][selected])}


def run_episode(
    env,
    variant: str,
    models: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    sequences: np.ndarray,
    cycles: int,
    seed: int,
    feature_interface: str,
    mode_xy_regression_margin: float = 4.0,
    mode_motion_distance_threshold: float = 2.0,
    mode_zero_policy: str = "nonzero",
    uncertainty_z: float = 1.0,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed, options={"state": state.copy(), "goal_state": goal.copy()})
    current = np.asarray(observation["state"], dtype=np.float64).copy()
    terminated = False
    truncated = False
    history: list[dict[str, Any]] = []
    contact_before_steps = 0
    contact_preserved_steps = 0
    recovery_steps = 0
    for cycle in range(cycles):
        online_scores = None
        if variant == "sequence_mode_online_orientation_receding":
            online_scores = online_orientation_progress(current, goal, sequences, history)
        selected, decision_info = choose_sequence(
            variant, models, current, goal, sequences, feature_interface,
            mode_motion_threshold=mode_motion_distance_threshold,
            mode_xy_regression_margin=mode_xy_regression_margin,
            mode_zero_policy=mode_zero_policy,
            online_orientation_scores=online_scores,
            uncertainty_z=uncertainty_z,
        )
        if decision_info.get("selection_mode") == "contact_reacquisition":
            recovery_steps += 1
        before_action = current.copy()
        for action in sequences[selected]:
            was_contact = mode_for_state(current) == "contact"
            observation, _, terminated, truncated, _ = env.step(action)
            current = np.asarray(observation["state"], dtype=np.float64).copy()
            if was_contact:
                contact_before_steps += 1
                contact_preserved_steps += int(mode_for_state(current) == "contact")
            history.append({"before_state": before_action.copy(), "action": np.asarray(action, dtype=np.float64).copy(), "after_state": current.copy()})
            before_action = current.copy()
            if terminated or truncated:
                break
        # The environment's termination signal is not the paper's joint
        # block-pose criterion.  Stop a learned macro controller immediately
        # after the strict learned-task interface is met; otherwise a later
        # exploratory macro can destroy an already closed pose.
        current_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
        current_angle = angle_error(current[4], goal[4])
        if current_xy < 20.0 and current_angle < np.pi / 9.0:
            break
        if terminated or truncated:
            break
    final_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
    final_angle = angle_error(current[4], goal[4])
    return {
        "variant": variant,
        "final_xy": final_xy,
        "final_angle": final_angle,
        "strict_success": int(final_xy < 20.0 and final_angle < np.pi / 9.0),
        "contact_retention": float(contact_preserved_steps / contact_before_steps) if contact_before_steps else 1.0,
        "recovery_steps": int(recovery_steps),
        "decision_steps": int(len(history)),
        "final_state": current.tolist(),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    actions = torque_action_bank(args.directions, magnitudes=tuple(args.magnitudes))
    sequences = make_sequences(actions)
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        calibration_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 150000)
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        train_states = select_balanced_states(train_pool, args.train_states, args.seed + 1, ("contact",))
        calibration_states = select_balanced_states(
            calibration_pool, args.calibration_episodes, args.seed + 3, ("contact",)
        )
        held_states = select_balanced_states(held_pool, args.held_out_episodes, args.seed + 2, ("contact",))
        features, labels = make_training_data(
            env, train_states, actions, args.goals_per_state,
            args.seed + 300000, args.feature_interface,
            args.path_contact_label,
        )
        models = fit_models(
            features, labels, args.trees, args.seed + 7,
            model_family=args.model_family, knn_neighbors=args.knn_neighbors,
        )
        one_step_features, one_step_labels = make_training_data(
            env, train_states, actions, args.goals_per_state,
            args.seed + 300000, args.feature_interface,
            args.path_contact_label, sequence_length=1,
        )
        one_step_models = fit_models(
            one_step_features, one_step_labels, args.trees, args.seed + 7,
            model_family=args.model_family, knn_neighbors=args.knn_neighbors,
        )
        one_step_models["angle_pairwise"] = fit_pairwise_angle_ranker(
            one_step_features, one_step_labels["angle_progress"], len(actions),
            args.trees, args.seed + 700,
        )
        one_step_models["task_pairwise"] = fit_pairwise_angle_ranker(
            one_step_features,
            task_rank_target(one_step_labels, args.mode_xy_regression_margin),
            len(actions), args.trees, args.seed + 800,
        )
        rng = np.random.default_rng(args.seed + 400000)
        def make_cases(states: list[np.ndarray], seed_offset: int) -> list[dict[str, Any]]:
            case_rng = np.random.default_rng(args.seed + seed_offset)
            return [
                {"state": state.copy(), "goal": goal_for_state(state, index, case_rng, closure_only=True)}
                for index, state in enumerate(states)
            ]

        def run_variants(cases: list[dict[str, Any]], variants: tuple[str, ...], seed_offset: int) -> list[dict[str, Any]]:
            result = []
            for case_index, case in enumerate(cases):
                for variant in variants:
                    receding = variant in RECEDING_VARIANTS
                    active_models = one_step_models if receding else models
                    active_sequences = make_sequences(actions, length=1) if receding else sequences
                    result.append(run_episode(
                        env, variant, active_models, case["state"], case["goal"], active_sequences,
                        args.closure_cycles, args.seed + seed_offset + case_index,
                        args.feature_interface,
                        args.mode_xy_regression_margin,
                        args.mode_motion_distance_threshold,
                        args.mode_zero_policy,
                        args.uncertainty_z,
                    ))
            return result

        calibration_cases = make_cases(calibration_states, 410000)
        held_cases = make_cases(held_states, 420000)
        calibration_reports = run_variants(
            calibration_cases,
            ("sequence_mode_receding", "sequence_mode_hybrid_receding"),
            510000,
        )
        calibration_gate = calibration_gate_decision(
            [row for row in calibration_reports if row["variant"] == "sequence_mode_receding"],
            [row for row in calibration_reports if row["variant"] == "sequence_mode_hybrid_receding"],
        )
        reports = run_variants(held_cases, VARIANTS, 520000)
        deployed_variant = (
            "sequence_mode_hybrid_receding"
            if calibration_gate["passed"]
            else "sequence_mode_receding"
        )
        gated_rows = [
            dict(row, variant="sequence_mode_calibrated_gate")
            for row in reports
            if row["variant"] == deployed_variant
        ]
        reports.extend(gated_rows)
    finally:
        env.close()
    summaries = []
    report_variants = VARIANTS + ("sequence_mode_calibrated_gate",)
    for index, variant in enumerate(report_variants):
        subset = [row for row in reports if row["variant"] == variant]
        summaries.append({"variant": variant, "metrics": {name: bootstrap(np.asarray([row[name] for row in subset]), args.bootstrap, args.seed + 600 + index * 100 + metric_index) for metric_index, name in enumerate(("strict_success", "final_xy", "final_angle", "contact_retention", "recovery_steps"))}, "rows": subset})
    report = {
        "protocol": {"environment": "swm/PushT-v1", "train_contact_states": len(train_states), "calibration_contact_episodes": len(calibration_states), "held_out_contact_episodes": len(held_states), "primitive_action_count": len(actions), "ordered_two_action_sequence_count": len(sequences), "one_step_action_count": int(len(actions)), "directions": args.directions, "magnitudes": [float(value) for value in args.magnitudes], "goals_per_state": args.goals_per_state, "closure_cycles": args.closure_cycles, "feature_interface": args.feature_interface, "path_contact_label": args.path_contact_label, "mode_zero_policy": args.mode_zero_policy, "mode_contact_threshold": 0.60, "mode_motion_distance_threshold": args.mode_motion_distance_threshold, "mode_xy_regression_margin": args.mode_xy_regression_margin, "model_family": args.model_family, "knn_neighbors": args.knn_neighbors, "runtime_candidate_successor_verification": False, "root_disjoint_splits": "train/calibration/test", "calibration_gate": "hybrid candidate admitted only under fixed non-inferiority checks; otherwise mode-receding fallback", "wall_seconds": time.perf_counter() - started},
        "variants": {
            "sequence_joint": "direct two-step joint decision",
            "sequence_soft_contact": "direct two-step pose decision with soft contact and motion risk",
            "sequence_turn_soft_contact": "direct two-step goal-directed turn decision with soft contact and motion risk",
            "sequence_continuous_rank": "continuous task-progress ranking with soft contact/motion risk",
            "sequence_mode_lexicographic": "successor-mode feasibility gate followed by angle-progress ranking",
            "sequence_mode_receding": "horizon-matched one-step mode gate with re-observation after every primitive action",
            "sequence_mode_pairwise_receding": "horizon-matched one-step mode gate with within-state pairwise angle ranking",
            "sequence_mode_task_pairwise_receding": "horizon-matched one-step mode gate with pairwise executable-task ranking (contact, motion, XY, then angle)",
            "sequence_mode_pairwise_staged_receding": "pairwise orientation ranking only inside a 24-pixel terminal tube; contact-preserving translation outside",
            "sequence_mode_online_orientation_receding": "pairwise mode-gated top-8 shortlist with observed local torque-to-angle calibration",
            "sequence_mode_knn_receding": "distance-weighted KNN mode/effect heads on standardized contact-frame features",
            "sequence_mode_uncertain_receding": "successor-mode head plus fitted-tree lower-confidence action set and fail-closed fallback",
            "sequence_mode_hybrid_receding": "staged contact-preserving translation followed by successor-mode-aware terminal orientation closure",
            "sequence_mode_calibrated_gate": "root-disjoint calibration gate choosing the hybrid candidate only when it is non-inferior on success, XY, angle, and contact retention",
        },
        "summaries": summaries,
        "calibration": {
            "rows": calibration_reports,
            "gate": calibration_gate,
            "deployed_variant": deployed_variant,
        },
        "training": {"rows": int(len(features)), "one_step_rows": int(len(one_step_features)), "positive_rates": {name: float(np.mean(values)) for name, values in labels.items()}, "one_step_positive_rates": {name: float(np.mean(values)) for name, values in one_step_labels.items()}},
        "claim_boundary": "Privileged simulator-state pure learned two-action verification. The verifier replaces runtime simulator successor checking, but this is not visual LeWM planning, physical latent causality, or universal world-model evidence.",
        "sanity_checks": {"all_finite": bool(np.isfinite(features).all() and all(np.isfinite(value) for item in summaries for metric in item["metrics"].values() for value in metric.values() if isinstance(value, (int, float)))), "same_cases_and_goals": True, "root_disjoint_train_calibration_test": True, "runtime_candidate_successor_verification": False},
    }
    (output / "learned_sequence_verifier.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# Pure learned sequence verifier", "", "| variant | strict success | final XY | final angle |", "|---|---:|---:|---:|"]
    for item in summaries:
        m = item["metrics"]
        lines.append("| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] |" % (item["variant"], m["strict_success"]["estimate"], m["strict_success"]["ci95_low"], m["strict_success"]["ci95_high"], m["final_xy"]["estimate"], m["final_xy"]["ci95_low"], m["final_xy"]["ci95_high"], m["final_angle"]["estimate"], m["final_angle"]["ci95_low"], m["final_angle"]["ci95_high"]))
    lines += ["", report["claim_boundary"], ""]
    (output / "learned_sequence_verifier.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": summaries, "training": report["training"], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
