"""Matched learned-dynamics planning audit for contact-local state interfaces.

This experiment asks whether the object-centric input found by the paired
contact diagnostic is useful to a closed-loop planner, rather than only to an
offline endpoint regressor.  Three ExtraTrees one-step physical adapters are
trained on the same genuine approach/contact trajectories:

* ``state_action`` uses the complete simulator state and current action;
* ``object_centric`` removes absolute agent coordinates and agent velocity,
  retaining block pose and relative contact geometry;
* ``history`` adds the previous action and action increment to full state;
* ``block_only`` predicts only block pose effects while retaining the actual
  agent state between replans.

Every arm receives the same held-out initial states, candidate action banks,
planning horizon, and action block.  Only the learned adapter used to score
candidate endpoints changes.  The environment executes the selected action
block and is then reset from the actual simulator state before the next
replan.  This is a privileged simulator-state learned-planner diagnostic; it
is not a visual LeWM or frozen-latent planning result.
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
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_contact_history_velocity_diagnostic import (
    MODEL_NAMES as BASE_MODEL_NAMES,
    choose_behavior_action,
    collect_trajectory_data,
    feature_vector,
    fit_models,
    mode_for_state,
    normalize_features,
    predict_step,
    unit,
)
from experiments.run_pusht_learned_physical_dynamics import state_features
from experiments.run_pusht_receding_block_pose_planner import (
    make_state_only_env,
    plan_closure_preserving_return,
    probe_contact_routes,
)
from experiments.run_pusht_obstacle_aware_local_effect import (
    route_to_contact,
    return_agent,
    task_success,
)

gym.logger.min_level = 40

PLANNER_MODEL_NAMES = (
    *BASE_MODEL_NAMES,
    "block_only",
    "block_only_specialist",
    "block_agent_hybrid",
    "block_only_torque_readout",
)


def strict_task_success(state: np.ndarray, goal: np.ndarray) -> bool:
    """Return the paper-level PushT success criterion.

    ``swm/PushT-v1`` can emit Gym ``terminated`` when the translational
    coordinates are close even if the block orientation is still outside the
    protocol margin.  The planner must therefore use the explicit joint
    criterion for both control-state transitions and reported success.
    """
    return bool(task_success(np.asarray(state), np.asarray(goal)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-transitions", type=int, default=768)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument(
        "--direct-train-states",
        type=int,
        default=96,
        help="contact states used to fit the fixed-horizon direct endpoint adapter",
    )
    parser.add_argument(
        "--direct-action-magnitudes",
        nargs="+",
        type=float,
        default=(0.15, 0.25, 0.35, 0.55, 0.75),
        help="absolute action magnitudes used by the direct endpoint fan",
    )
    parser.add_argument(
        "--direct-state-pool",
        choices=("protocol", "mixed", "near_contact_mixed"),
        default="protocol",
        help=(
            "use contact states only, mix contact states from two streams, or "
            "mix all agent-block states within the 65-pixel learned contact manifold"
        ),
    )
    parser.add_argument(
        "--direct-use-previous-action",
        action="store_true",
        help="ablation: expose previous action in the direct endpoint feature interface",
    )
    parser.add_argument(
        "--direct-uncertainty-penalty",
        type=float,
        default=0.0,
        help="risk penalty (in predicted endpoint units) for direct learned ranking",
    )
    parser.add_argument(
        "--direct-contact-mode-penalty",
        type=float,
        default=0.0,
        help=(
            "optional penalty for direct candidates predicted to leave the contact "
            "mode; zero preserves the endpoint-only control"
        ),
    )
    parser.add_argument(
        "--direct-horizon-consistency-penalty",
        type=float,
        default=0.0,
        help=(
            "penalize endpoint candidates whose one-step torque readout does "
            "not support the two-step predicted orientation improvement"
        ),
    )
    parser.add_argument(
        "--direct-goal-conditioned-ranker",
        action="store_true",
        help=(
            "train and use a learned goal-conditioned action-quality ranker "
            "instead of converting a predicted effect into a goal score"
        ),
    )
    parser.add_argument(
        "--direct-goal-conditioned-feasibility",
        action="store_true",
        help=(
            "train and deploy a binary phase-aware feasibility head; selection uses "
            "only its learned probability and no simulator successor verification"
        ),
    )
    parser.add_argument(
        "--direct-goal-conditioned-score",
        action="store_true",
        help=(
            "train phase-specific scalar action-utility heads; this is an "
            "offline-supervised pure-learned ranking arm"
        ),
    )
    parser.add_argument(
        "--direct-goal-conditioned-policy",
        action="store_true",
        help=(
            "distill the best action-bank index into separate translation and "
            "closure policy heads"
        ),
    )
    parser.add_argument(
        "--direct-ranker-goals-per-state",
        type=int,
        default=4,
        help="synthetic target poses per direct training state for the goal ranker",
    )
    parser.add_argument(
        "--direct-dagger-rounds",
        type=int,
        default=0,
        help="DAgger-style pure-learned state-distribution augmentation rounds",
    )
    parser.add_argument(
        "--direct-dagger-states-per-round",
        type=int,
        default=64,
        help="maximum policy-visited contact states added per DAgger round",
    )
    parser.add_argument(
        "--direct-dagger-steps",
        type=int,
        default=6,
        help="action-block steps per DAgger rollout",
    )
    parser.add_argument(
        "--direct-online-calibration",
        action="store_true",
        help=(
            "update a mode-conditioned endpoint residual only from the observed "
            "successor after a committed learned action block"
        ),
    )
    parser.add_argument(
        "--direct-online-local-fit",
        action="store_true",
        help="blend the direct adapter with a low-dimensional per-episode local action-effect fit",
    )
    parser.add_argument(
        "--direct-require-contact-mode",
        action="store_true",
        help=(
            "mode-aware direct endpoint protocol: route back to the strict "
            "contact manifold before applying the learned phase head; the "
            "legacy <=65-pixel near-contact protocol remains the default"
        ),
    )
    parser.add_argument(
        "--direct-online-local-strategy",
        choices=("ridge", "knn", "state_knn", "state_residual_knn", "orientation_gain"),
        default="ridge",
        help=(
            "online local action-effect calibration: ridge is the legacy global local fit; "
            "knn applies corrections only to candidates covered by nearby same-mode actions; "
            "state_knn additionally gates coverage by the observed contact geometry; "
            "state_residual_knn learns only local residuals relative to the frozen adapter; "
            "orientation_gain calibrates only the angular effect channel"
        ),
    )
    parser.add_argument(
        "--direct-online-local-weight",
        type=float,
        default=0.5,
        help="maximum blend weight for an online local calibration (0 disables its effect)",
    )
    parser.add_argument(
        "--direct-online-local-radius",
        type=float,
        default=0.30,
        help="maximum local action/state distance for a KNN correction",
    )
    parser.add_argument(
        "--direct-online-local-neighbors",
        type=int,
        default=3,
        help="number of same-mode action neighbors used by KNN local calibration",
    )
    parser.add_argument(
        "--torque-feature-interface",
        choices=("contact_frame", "legacy", "target_relative", "block_target_frame"),
        default="contact_frame",
        help=(
            "local torque adapter input: contact_frame is the historical alias "
            "for legacy action projections; target_relative and block_target_frame "
            "are explicit alternative interfaces"
        ),
    )
    parser.add_argument("--direct-sequence-directions", type=int, default=8)
    parser.add_argument(
        "--direct-sequence-magnitudes",
        nargs="+",
        type=float,
        default=(0.2, 0.4, 0.6),
        help="action magnitudes used by the learned two-action sequence fan",
    )
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--action-block", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--action-magnitude", type=float, default=0.65)
    parser.add_argument("--noise", type=float, default=0.18)
    parser.add_argument(
        "--training-protocol",
        choices=("trajectory", "contact_intervention"),
        default="trajectory",
        help="genuine behavior transitions or contact-state action interventions",
    )
    parser.add_argument(
        "--planning-objective",
        choices=(
            "joint_block_pose",
            "contact_gated",
            "staged_contact",
            "staged_contact_orient",
            "staged_contact_recovery",
            "staged_contact_feedback",
            "staged_contact_calibrated",
            "staged_contact_route",
            "staged_contact_route_calibrated",
            "staged_contact_local_effect",
            "staged_contact_torque_readout",
            "staged_contact_direct_torque_readout",
            "staged_contact_direct_sequence_readout",
            "staged_contact_verified_torque",
            "staged_contact_verified_torque_hold",
            "staged_contact_verified_torque_stabilized",
            "staged_contact_hierarchical",
            "staged_contact_verified_torque_fullfan",
            "staged_contact_verified_torque_hierarchical_fullfan",
            "staged_contact_verified_hybrid_fullfan",
        ),
        default="joint_block_pose",
        help=(
            "legacy block-pose endpoint score, contact-gated score, or "
            "staged contact acquisition/push/return controller, optionally "
            "with an explicit orientation phase"
        ),
    )
    parser.add_argument(
        "--contact-gate-threshold",
        type=float,
        default=0.85,
        help="probability threshold for allowing predicted block-pose updates",
    )
    parser.add_argument("--route-clearance", type=float, default=140.0)
    parser.add_argument("--route-contact-distance", type=float, default=42.0)
    parser.add_argument("--route-settle-steps", type=int, default=6)
    parser.add_argument(
        "--max-contact-recoveries",
        type=int,
        default=3,
        help="maximum bounded near-goal contact-route recoveries per episode",
    )
    parser.add_argument(
        "--local-effect-directions",
        type=int,
        default=24,
        help="one-step simulator probes used only in the near-goal orientation phase",
    )
    parser.add_argument(
        "--local-effect-scale",
        type=float,
        default=0.55,
        help="probe action magnitude for the near-goal local-effect phase",
    )
    parser.add_argument(
        "--local-effect-xy-margin",
        type=float,
        default=-1.0,
        help="optional per-probe maximum XY increase; negative disables the monotone tube",
    )
    parser.add_argument(
        "--torque-verification-topk",
        type=int,
        default=8,
        help="number of learned torque proposals verified with one simulator step",
    )
    parser.add_argument(
        "--closure-hold-xy-margin",
        type=float,
        default=2.0,
        help="extra block-XY tube allowed after translation first enters the 20-pixel closure region",
    )
    parser.add_argument(
        "--closure-hold-exit-xy",
        type=float,
        default=26.0,
        help="hysteresis exit threshold; translation lock is released only after this XY error",
    )
    parser.add_argument(
        "--closure-return-population", type=int, default=1024,
        help="population for the primary closure-preserving terminal search",
    )
    parser.add_argument(
        "--closure-return-iterations", type=int, default=10,
        help="iterations for the primary closure-preserving terminal search",
    )
    parser.add_argument("--closure-return-horizon", type=int, default=24)
    parser.add_argument("--closure-return-rescue-population", type=int, default=1536)
    parser.add_argument("--closure-return-rescue-iterations", type=int, default=12)
    parser.add_argument("--closure-return-rescue-restarts", type=int, default=1)
    parser.add_argument("--closure-return-deep-population", type=int, default=2048)
    parser.add_argument("--closure-return-deep-iterations", type=int, default=16)
    parser.add_argument("--closure-return-deep-horizon", type=int, default=32)
    parser.add_argument("--closure-return-deep-restarts", type=int, default=1)
    parser.add_argument("--closure-return-path-xy-margin", type=float, default=1.0)
    parser.add_argument("--closure-return-path-angle-margin", type=float, default=0.02)
    parser.add_argument(
        "--terminal-return-retries",
        type=int,
        default=2,
        help="bounded retries after a failed closure-preserving terminal search",
    )
    parser.add_argument("--orientation-path-population", type=int, default=128)
    parser.add_argument("--orientation-path-iterations", type=int, default=3)
    parser.add_argument("--orientation-path-horizon", type=int, default=6)
    parser.add_argument("--return-clearance", type=float, default=140.0)
    parser.add_argument("--return-steps", type=int, default=45)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--feedback-patience",
        type=int,
        default=2,
        help="consecutive model-mismatch steps before contact feedback takes over",
    )
    parser.add_argument(
        "--feedback-steps",
        type=int,
        default=6,
        help="bounded contact-direction actions per feedback intervention",
    )
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def angle_error(value: float, target: float) -> float:
    return float(abs((float(value) - float(target) + np.pi) % (2.0 * np.pi) - np.pi))


def block_pose_cost(state: np.ndarray, goal: np.ndarray) -> float:
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    return xy + 45.0 * angle_error(state[4], goal[4])


def planner_feature_vector(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Build the shared feature interface plus the block-only control arm."""
    if model_name in (
        "block_only", "block_only_specialist", "block_agent_hybrid",
        "block_only_torque_readout",
    ):
        return feature_vector(state, action, previous_action, "object_centric")
    return feature_vector(state, action, previous_action, model_name)


def fit_block_only_model(
    data: dict[str, np.ndarray],
    trees: int,
    seed: int,
) -> ExtraTreesRegressor:
    """Fit a block-pose effect model without extrapolating agent dynamics."""
    features = np.asarray([
        planner_feature_vector(state, action, previous, "block_only")
        for state, action, previous in zip(
            data["states"], data["actions"], data["previous_actions"]
        )
    ])
    targets = np.asarray([
        state_features(next_state)[2:6] - state_features(state)[2:6]
        for state, next_state in zip(data["states"], data["next_states"])
    ])
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(features, targets)
    return model


def fit_torque_effect_model(
    data: dict[str, np.ndarray],
    trees: int,
    seed: int,
) -> ExtraTreesRegressor:
    """Fit a contact-local readout for block ``dx``, ``dy``, and ``dtheta``.

    The ordinary adapter encodes orientation as sine/cosine differences and
    is optimized for endpoint rollout.  This readout instead predicts the
    local control effect used by the near-goal action-ranking rule.  It is fit
    only on simulator-derived one-step contact interventions.
    """
    features = np.asarray([
        planner_feature_vector(state, action, previous, "block_only")
        for state, action, previous in zip(
            data["states"], data["actions"], data["previous_actions"]
        )
    ])
    targets = np.asarray([
        [
            float(next_state[2] - state[2]),
            float(next_state[3] - state[3]),
            float((next_state[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi),
        ]
        for state, next_state in zip(data["states"], data["next_states"])
    ])
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(features, targets)
    return model


def torque_contact_frame_features(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Build translation-normalized contact features for torque effects."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    block_angle = float(state[4])
    relative_angle = float(np.arctan2(relative[1], relative[0]) - block_angle)
    velocity = np.asarray(state[5:7], dtype=np.float64)
    return np.asarray([
        float(np.linalg.norm(relative)),
        float(np.dot(action, normal)),
        float(np.dot(action, tangent)),
        float(np.linalg.norm(action)),
        float(np.dot(previous_action, normal)),
        float(np.dot(previous_action, tangent)),
        float(np.dot(velocity, normal)),
        float(np.dot(velocity, tangent)),
        float(np.cos(relative_angle)),
        float(np.sin(relative_angle)),
        float(np.cos(block_angle)),
        float(np.sin(block_angle)),
    ], dtype=np.float64)


def torque_target_relative_features(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Build contact features aligned with PushT's relative PD target.

    PushT does not interpret ``action`` as an absolute pusher position.  Its
    live step first forms ``target = agent_position + 100 * action`` and then
    applies a position/velocity controller.  The legacy interface exposes the
    action projections but not the resulting target relative to the block.
    This interface makes that quantity explicit while retaining the matched
    legacy covariates and velocity terms.
    """
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    block_angle = float(state[4])
    relative_angle = float(np.arctan2(relative[1], relative[0]) - block_angle)
    velocity = np.asarray(state[5:7], dtype=np.float64)
    target_relative = 100.0 * action - relative
    target_normal = float(np.dot(target_relative, normal))
    target_tangent = float(np.dot(target_relative, tangent))
    target_distance = float(np.linalg.norm(target_relative))
    target_angle = float(np.arctan2(target_relative[1], target_relative[0]) - block_angle)
    previous_target_relative = 100.0 * previous_action - relative
    return np.asarray([
        float(np.linalg.norm(relative)),
        float(np.dot(action, normal)),
        float(np.dot(action, tangent)),
        float(np.linalg.norm(action)),
        float(np.dot(previous_action, normal)),
        float(np.dot(previous_action, tangent)),
        float(np.dot(velocity, normal)),
        float(np.dot(velocity, tangent)),
        float(np.cos(relative_angle)),
        float(np.sin(relative_angle)),
        float(np.cos(block_angle)),
        float(np.sin(block_angle)),
        target_normal,
        target_tangent,
        target_distance,
        float(np.dot(previous_target_relative, normal)),
        float(np.dot(previous_target_relative, tangent)),
        float(np.cos(target_angle)),
        float(np.sin(target_angle)),
    ], dtype=np.float64)


def torque_block_target_frame_features(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Build a compact block-frame interface for the PushT PD target.

    The contact-normal frame changes with the current agent--block ray and
    therefore mixes object orientation and contact location.  The block frame
    keeps the T geometry fixed: it exposes the current agent position, the
    next PD target, velocity, and previous target in coordinates rotated by
    the observed block angle.  No future simulator state is used.
    """
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    angle = float(state[4])
    rotation = np.asarray([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)],
    ], dtype=np.float64)
    agent_relative = rotation @ (state[:2] - state[2:4])
    target_relative = rotation @ (state[:2] + 100.0 * action - state[2:4])
    previous_target_relative = rotation @ (
        state[:2] + 100.0 * previous_action - state[2:4]
    )
    velocity = rotation @ np.asarray(state[5:7], dtype=np.float64)
    action_displacement = rotation @ (100.0 * action)
    previous_displacement = rotation @ (100.0 * previous_action)
    return np.asarray([
        float(agent_relative[0]),
        float(agent_relative[1]),
        float(target_relative[0]),
        float(target_relative[1]),
        float(np.linalg.norm(target_relative)),
        float(velocity[0]),
        float(velocity[1]),
        float(action_displacement[0]),
        float(action_displacement[1]),
        float(previous_displacement[0]),
        float(previous_displacement[1]),
        float(previous_target_relative[0]),
        float(previous_target_relative[1]),
    ], dtype=np.float64)


def torque_feature_vector(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    feature_interface: str = "legacy",
) -> np.ndarray:
    """Dispatch the explicitly named torque feature interface."""
    if feature_interface in ("contact_frame", "legacy"):
        return torque_contact_frame_features(state, action, previous_action)
    if feature_interface == "target_relative":
        return torque_target_relative_features(state, action, previous_action)
    if feature_interface == "block_target_frame":
        return torque_block_target_frame_features(state, action, previous_action)
    raise ValueError(f"unknown torque feature interface: {feature_interface}")


def fit_contact_frame_torque_effect_model(
    data: dict[str, np.ndarray],
    trees: int,
    seed: int,
    feature_interface: str = "legacy",
) -> ExtraTreesRegressor:
    """Fit a local-frame torque readout with equivariant XY targets."""
    features = []
    targets = []
    for state, action, previous, next_state in zip(
        data["states"], data["actions"], data["previous_actions"], data["next_states"]
    ):
        relative = np.asarray(state[2:4] - state[:2], dtype=np.float64)
        normal = unit(relative)
        tangent = np.asarray([-normal[1], normal[0]])
        delta_xy = np.asarray(next_state[2:4] - state[2:4], dtype=np.float64)
        features.append(torque_feature_vector(state, action, previous, feature_interface))
        targets.append([
            float(np.dot(delta_xy, normal)),
            float(np.dot(delta_xy, tangent)),
            float((next_state[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi),
        ])
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(np.asarray(features), np.asarray(targets))
    return model


def fit_contact_frame_torque_effect_specialists(
    data: dict[str, np.ndarray],
    trees: int,
    seed: int,
    feature_interface: str = "legacy",
) -> dict[str, ExtraTreesRegressor]:
    """Fit mode-conditional local-effect readouts with a pooled fallback.

    Contact dynamics are discontinuous at the approach/contact boundary.  A
    pooled torque model is useful as a stable fallback, but it can average
    zero/weak approach effects with genuine contact torque.  Routing by the
    observed simulator state keeps the readout local without introducing a
    future-state label or changing the action verification protocol.
    """
    pooled = fit_contact_frame_torque_effect_model(
        data, trees, seed, feature_interface=feature_interface
    )
    specialists: dict[str, ExtraTreesRegressor] = {"fallback": pooled}
    modes = np.asarray(data["modes"])
    for index, mode in enumerate(("approach", "contact")):
        mask = modes == mode
        if int(mask.sum()) < 32:
            specialists[mode] = pooled
            continue
        subset = {key: value[mask] for key, value in data.items()}
        specialists[mode] = fit_contact_frame_torque_effect_model(
            subset, trees, seed + 101 + index,
            feature_interface=feature_interface,
        )
    specialists["free"] = pooled
    return specialists


def fit_direct_contact_endpoint_model(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    trees: int,
    seed: int,
    previous_actions: np.ndarray | None = None,
    feature_interface: str = "legacy",
) -> ExtraTreesRegressor:
    """Fit a direct contact-frame endpoint model for a fixed action block.

    The target is the net block effect after ``horizon`` repeated actions,
    rather than a one-step delta that is recursively fed back into the model.
    This is intentionally an opt-in adapter so the established one-step
    protocol remains unchanged and can serve as a matched control.
    """
    if horizon < 1:
        raise ValueError("direct endpoint horizon must be positive")
    features = []
    targets = []
    endpoint_contact_labels = []
    states = np.asarray(states, dtype=np.float64)
    if previous_actions is None:
        previous_actions = np.zeros((len(states), 2), dtype=np.float64)
    previous_actions = np.asarray(previous_actions, dtype=np.float64)
    if previous_actions.shape != (len(states), 2):
        raise ValueError("direct endpoint previous_actions must have shape (n, 2)")
    for state_index, state in enumerate(states):
        for action_index, action in enumerate(np.asarray(actions, dtype=np.float64)):
            env.reset(
                seed=seed + state_index * 1000 + action_index,
                options={"state": state.copy(), "goal_state": state.copy()},
            )
            endpoint = state.copy()
            for _ in range(horizon):
                observation, _, terminated, truncated, _ = env.step(action)
                endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                if terminated or truncated:
                    break
            relative = state[2:4] - state[:2]
            normal = unit(relative)
            tangent = np.asarray([-normal[1], normal[0]])
            delta = endpoint[2:4] - state[2:4]
            features.append(torque_feature_vector(
                state, action, previous_actions[state_index], feature_interface
            ))
            targets.append([
                float(np.dot(delta, normal)),
                float(np.dot(delta, tangent)),
                float((endpoint[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi),
            ])
            endpoint_contact_labels.append(int(mode_for_state(endpoint) == "contact"))
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed + horizon,
        n_jobs=1,
    ).fit(np.asarray(features), np.asarray(targets))
    # The endpoint adapter remains a three-dimensional effect regressor.  The
    # optional classifier is attached as an auditable auxiliary head so that
    # existing effect/uncertainty code and matched controls are unchanged.
    labels = np.asarray(endpoint_contact_labels, dtype=np.int64)
    if len(np.unique(labels)) >= 2:
        model.contact_mode_model_ = ExtraTreesClassifier(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed + horizon + 7000,
            n_jobs=1,
        ).fit(np.asarray(features), labels)
    else:
        model.contact_mode_model_ = None
    return model


def goal_conditioned_ranker_features(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    goal: np.ndarray,
    feature_interface: str = "legacy",
) -> np.ndarray:
    """Encode the current task-relative action-quality interface."""
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    angle = float(state[4])
    rotation = np.asarray([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)],
    ])
    goal_relative = rotation @ (goal[2:4] - state[2:4])
    goal_angle_delta = float((goal[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi)
    return np.concatenate([
        torque_feature_vector(state, action, previous_action, feature_interface),
        np.asarray([
            goal_relative[0] / 100.0,
            goal_relative[1] / 100.0,
            np.linalg.norm(goal_relative) / 100.0,
            np.cos(goal_angle_delta),
            np.sin(goal_angle_delta),
            np.linalg.norm(state[2:4] - goal[2:4]) / 100.0,
            # Keep approach and stable-contact states distinct even when
            # their relative distances are similar.  These indicators are
            # computed from the current state only and are available at
            # deployment exactly as they are during label generation.
            np.linalg.norm(state[:2] - state[2:4]) / 65.0,
            float(mode_for_state(state) == "contact"),
            float(mode_for_state(state) == "approach"),
        ]),
    ]).astype(np.float64)


def fit_goal_conditioned_action_ranker(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    trees: int,
    seed: int,
    previous_actions: np.ndarray | None = None,
    feature_interface: str = "legacy",
    goals_per_state: int = 4,
) -> ExtraTreesRegressor:
    """Fit a direct goal-conditioned action-quality readout.

    Each state/action successor is simulated once to produce labels for
    several synthetic target poses.  The model predicts angular and XY
    improvement directly; the planner therefore does not need to reconstruct
    a wrapped orientation error from a separately predicted effect vector.
    Target generation is offline training data construction only.
    """
    if horizon < 1 or goals_per_state < 2:
        raise ValueError("ranker horizon must be positive and goals_per_state >= 2")
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if previous_actions is None:
        previous_actions = np.zeros((len(states), 2), dtype=np.float64)
    previous_actions = np.asarray(previous_actions, dtype=np.float64)
    if previous_actions.shape != (len(states), 2):
        raise ValueError("ranker previous_actions must have shape (n, 2)")
    rng = np.random.default_rng(seed + 99173)
    features = []
    targets = []
    for state_index, state in enumerate(states):
        endpoints = []
        for action_index, action in enumerate(actions):
            env.reset(
                seed=seed + state_index * 1000 + action_index,
                options={"state": state.copy(), "goal_state": state.copy()},
            )
            endpoint = state.copy()
            for _ in range(horizon):
                observation, _, terminated, truncated, _ = env.step(action)
                endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                if terminated or truncated:
                    break
            endpoints.append(endpoint)
        for goal_index in range(goals_per_state):
            radius = float(rng.uniform(20.0, 260.0))
            direction = float(rng.uniform(-np.pi, np.pi))
            goal = state.copy()
            goal[2:4] = np.clip(
                state[2:4] + radius * np.asarray([np.cos(direction), np.sin(direction)]),
                20.0,
                492.0,
            )
            goal[4] = float(
                (state[4] + rng.uniform(-np.pi, np.pi) + np.pi) % (2.0 * np.pi) - np.pi
            )
            before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
            before_angle = angle_error(state[4], goal[4])
            for action_index, (action, endpoint) in enumerate(zip(actions, endpoints)):
                after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
                after_angle = angle_error(endpoint[4], goal[4])
                features.append(goal_conditioned_ranker_features(
                    state, action, previous_actions[state_index], goal, feature_interface
                ))
                targets.append([before_angle - after_angle, before_xy - after_xy])
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed + 8801,
        n_jobs=1,
    )
    return model.fit(np.asarray(features), np.asarray(targets))


def fit_goal_conditioned_feasibility_classifier(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    trees: int,
    seed: int,
    previous_actions: np.ndarray | None = None,
    feature_interface: str = "legacy",
    goals_per_state: int = 8,
) -> ExtraTreesClassifier:
    """Fit a direct, goal-conditioned feasibility decision head.

    The endpoint simulator is used only to create offline supervision.  The
    resulting classifier is deliberately trained on the decision event that
    the planner needs, rather than on a three-channel endpoint effect.  This
    avoids the unstable ``predict effect -> wrap angle -> hand-score`` chain
    used by the legacy direct adapter.

    The label is phase-aware.  When the target is far away, a candidate must
    reduce translation without making orientation materially worse.  In the
    closure regime, it must reduce wrapped orientation error while remaining
    inside the same translation tube.  The current contact mode is also
    retained for contact states so that the learned head does not prefer a
    visually attractive but contact-breaking impulse.
    """
    if horizon < 1 or goals_per_state < 2:
        raise ValueError("feasibility classifier needs positive horizon and goals_per_state >= 2")
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if previous_actions is None:
        previous_actions = np.zeros((len(states), 2), dtype=np.float64)
    previous_actions = np.asarray(previous_actions, dtype=np.float64)
    if previous_actions.shape != (len(states), 2):
        raise ValueError("feasibility previous_actions must have shape (n, 2)")
    rng = np.random.default_rng(seed + 99173)
    features = []
    labels = []
    for state_index, state in enumerate(states):
        endpoints = []
        for action_index, action in enumerate(actions):
            env.reset(
                seed=seed + state_index * 1000 + action_index,
                options={"state": state.copy(), "goal_state": state.copy()},
            )
            endpoint = state.copy()
            for _ in range(horizon):
                observation, _, terminated, truncated, _ = env.step(action)
                endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                if terminated or truncated:
                    break
            endpoints.append(endpoint)
        for goal_index in range(goals_per_state):
            # Include a dense local band because the difficult part of the
            # closed loop is pose closure, while retaining long targets for
            # the translation phase.
            radius = float(rng.uniform(10.0, 220.0))
            direction = float(rng.uniform(-np.pi, np.pi))
            goal = state.copy()
            goal[2:4] = np.clip(
                state[2:4] + radius * np.asarray([np.cos(direction), np.sin(direction)]),
                20.0,
                492.0,
            )
            goal[4] = float(
                (state[4] + rng.uniform(-np.pi, np.pi) + np.pi) % (2.0 * np.pi) - np.pi
            )
            before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
            before_angle = angle_error(state[4], goal[4])
            translation_phase = before_xy >= 45.0
            xy_limit = min(45.0, before_xy + 4.0)
            current_contact = mode_for_state(state) == "contact"
            for action, endpoint in zip(actions, endpoints):
                after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
                after_angle = angle_error(endpoint[4], goal[4])
                if translation_phase:
                    feasible = (
                        after_xy < before_xy - 1e-5
                        and after_angle <= before_angle + 0.12
                    )
                else:
                    feasible = (
                        after_angle < before_angle - 1e-5
                        and after_xy <= xy_limit
                    )
                if current_contact:
                    feasible = feasible and mode_for_state(endpoint) == "contact"
                features.append(goal_conditioned_ranker_features(
                    state, action, previous_actions[state_index], goal, feature_interface
                ))
                labels.append(int(feasible))
    features_array = np.asarray(features, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels_array)) < 2:
        raise ValueError("feasibility classifier training produced only one class")
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        class_weight="balanced",
        random_state=seed + 8803,
        n_jobs=1,
    ).fit(features_array, labels_array)


def fit_goal_conditioned_phase_feasibility_heads(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    trees: int,
    seed: int,
    previous_actions: np.ndarray | None = None,
    feature_interface: str = "legacy",
    goals_per_state: int = 8,
    goal_states: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit phase-specific, action-block-matched feasibility heads.

    The earlier feasibility classifier mixed translation and pose closure
    labels, then selected the largest probability regardless of the current
    phase.  That creates a label/interface mismatch: a candidate that is
    useful for translation can win while the planner is trying to reduce
    orientation error, and vice versa.  This function keeps the same offline
    simulator supervision but fits two independent heads and makes contact
    preservation an explicit part of both labels.

    ``env`` is used only to construct training labels.  The returned heads
    require only the current observed state, action, previous action, and
    goal at deployment.
    """
    if horizon < 1 or goals_per_state < 2:
        raise ValueError("phase feasibility needs positive horizon and goals_per_state >= 2")
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if previous_actions is None:
        previous_actions = np.zeros((len(states), 2), dtype=np.float64)
    previous_actions = np.asarray(previous_actions, dtype=np.float64)
    if previous_actions.shape != (len(states), 2):
        raise ValueError("phase feasibility previous_actions must have shape (n, 2)")
    if goal_states is not None:
        goal_states = np.asarray(goal_states, dtype=np.float64)
        if goal_states.shape != states.shape:
            raise ValueError("phase feasibility goal_states must match states shape")

    rng = np.random.default_rng(seed + 99173)
    features_by_phase = {"translation": [], "closure": []}
    labels_by_phase = {"translation": [], "closure": []}
    label_counts = {
        "translation": {"positive": 0, "negative": 0},
        "closure": {"positive": 0, "negative": 0},
    }
    for state_index, state in enumerate(states):
        endpoints = []
        for action_index, action in enumerate(actions):
            env.reset(
                seed=seed + state_index * 1000 + action_index,
                options={"state": state.copy(), "goal_state": state.copy()},
            )
            endpoint = state.copy()
            for _ in range(horizon):
                observation, _, terminated, truncated, _ = env.step(action)
                endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                if terminated or truncated:
                    break
            endpoints.append(endpoint)

        goals_for_state = []
        if goal_states is not None and np.isfinite(goal_states[state_index]).all():
            # DAgger states carry the actual goal that generated the
            # deployment distribution.  Random goals are still added below
            # so the augmented pool does not collapse to one task direction.
            goals_for_state.append(np.asarray(goal_states[state_index], dtype=np.float64).copy())
        for _ in range(goals_per_state):
            radius = float(rng.uniform(10.0, 220.0))
            direction = float(rng.uniform(-np.pi, np.pi))
            goal = state.copy()
            goal[2:4] = np.clip(
                state[2:4] + radius * np.asarray([np.cos(direction), np.sin(direction)]),
                20.0,
                492.0,
            )
            goal[4] = float(
                (state[4] + rng.uniform(-np.pi, np.pi) + np.pi) % (2.0 * np.pi) - np.pi
            )
            goals_for_state.append(goal)
        for goal in goals_for_state:
            before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
            before_angle = angle_error(state[4], goal[4])
            phase = "translation" if before_xy >= 45.0 else "closure"
            xy_limit = min(45.0, before_xy + 4.0)
            current_mode = mode_for_state(state)
            current_contact = current_mode == "contact"
            for action, endpoint in zip(actions, endpoints):
                after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
                after_angle = angle_error(endpoint[4], goal[4])
                endpoint_mode = mode_for_state(endpoint)
                endpoint_contact_distance = float(
                    np.linalg.norm(endpoint[:2] - endpoint[2:4])
                )
                # A contact state is not allowed to hand the next phase an
                # unconstrained/free state.  The 65-pixel band is the same
                # deployment manifold used by the direct adapter, so this is
                # an action-block-matched recoverability label rather than a
                # hidden successor query at runtime.
                contact_preserved = (
                    endpoint_mode == "contact"
                    and endpoint_contact_distance <= 65.0
                    if current_contact else True
                )
                if phase == "translation":
                    feasible = (
                        after_xy < before_xy - 1e-5
                        and after_angle <= before_angle + 0.08
                        and contact_preserved
                    )
                else:
                    feasible = (
                        after_angle < before_angle - 1e-5
                        and after_xy <= xy_limit
                        and contact_preserved
                    )
                features_by_phase[phase].append(
                    goal_conditioned_ranker_features(
                        state,
                        action,
                        previous_actions[state_index],
                        goal,
                        feature_interface,
                    )
                )
                labels_by_phase[phase].append(int(feasible))
                label_counts[phase]["positive" if feasible else "negative"] += 1

    heads: dict[str, Any] = {}
    for index, phase in enumerate(("translation", "closure")):
        phase_labels = np.asarray(labels_by_phase[phase], dtype=np.int64)
        if len(np.unique(phase_labels)) < 2:
            raise ValueError(
                f"phase feasibility training produced one class for {phase}: "
                f"{label_counts[phase]}"
            )
        heads[phase] = ExtraTreesClassifier(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=1.0,
            class_weight="balanced",
            random_state=seed + 8810 + index,
            n_jobs=1,
        ).fit(np.asarray(features_by_phase[phase]), phase_labels)
    return {
        "translation": heads["translation"],
        "closure": heads["closure"],
        "label_counts": label_counts,
        "protocol": {
            "phase_specific": True,
            "action_block_matched": True,
            "contact_preservation": "required from current contact mode",
            "successor_verification_at_deployment": False,
        },
    }


def fit_goal_conditioned_phase_score_heads(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    trees: int,
    seed: int,
    previous_actions: np.ndarray | None = None,
    feature_interface: str = "legacy",
    goals_per_state: int = 8,
) -> dict[str, Any]:
    """Fit phase-specific scalar utility heads for direct action ranking.

    Binary feasibility is useful for rejecting clearly bad candidates but is
    underdetermined whenever many actions satisfy the same label.  The score
    heads preserve the same no-verification deployment protocol while
    learning the scalar ranking objective actually used by the planner:
    translation progress in the translation phase and wrapped orientation
    progress inside a bounded XY tube in the closure phase.
    """
    if horizon < 1 or goals_per_state < 2:
        raise ValueError("phase score needs positive horizon and goals_per_state >= 2")
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if previous_actions is None:
        previous_actions = np.zeros((len(states), 2), dtype=np.float64)
    previous_actions = np.asarray(previous_actions, dtype=np.float64)
    if previous_actions.shape != (len(states), 2):
        raise ValueError("phase score previous_actions must have shape (n, 2)")
    rng = np.random.default_rng(seed + 99173)
    features_by_phase = {"translation": [], "closure": []}
    targets_by_phase = {"translation": [], "closure": []}
    for state_index, state in enumerate(states):
        endpoints = []
        for action_index, action in enumerate(actions):
            env.reset(
                seed=seed + state_index * 1000 + action_index,
                options={"state": state.copy(), "goal_state": state.copy()},
            )
            endpoint = state.copy()
            for _ in range(horizon):
                observation, _, terminated, truncated, _ = env.step(action)
                endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                if terminated or truncated:
                    break
            endpoints.append(endpoint)
        for _ in range(goals_per_state):
            radius = float(rng.uniform(10.0, 220.0))
            direction = float(rng.uniform(-np.pi, np.pi))
            goal = state.copy()
            goal[2:4] = np.clip(
                state[2:4] + radius * np.asarray([np.cos(direction), np.sin(direction)]),
                20.0,
                492.0,
            )
            goal[4] = float(
                (state[4] + rng.uniform(-np.pi, np.pi) + np.pi) % (2.0 * np.pi) - np.pi
            )
            before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
            before_angle = angle_error(state[4], goal[4])
            phase = "translation" if before_xy >= 45.0 else "closure"
            xy_limit = min(45.0, before_xy + 4.0)
            current_contact = mode_for_state(state) == "contact"
            for action, endpoint in zip(actions, endpoints):
                after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
                after_angle = angle_error(endpoint[4], goal[4])
                current_contact_distance = float(
                    np.linalg.norm(state[:2] - state[2:4])
                )
                endpoint_contact_distance = float(
                    np.linalg.norm(endpoint[:2] - endpoint[2:4])
                )
                # The direct head is deployed on the local contact manifold.
                # Leaving that manifold is therefore a bad action even when
                # the source state is approach rather than strict contact.
                # Stable contact additionally requires retaining contact mode.
                contact_preserved = (
                    endpoint_contact_distance <= 65.0
                    and (
                        mode_for_state(endpoint) == "contact"
                        if current_contact
                        else True
                    )
                ) if current_contact_distance <= 65.0 else True
                if phase == "translation":
                    score = (
                        (before_xy - after_xy) / max(before_xy, 20.0)
                        - 2.0 * max(0.0, after_angle - before_angle - 0.08)
                    )
                else:
                    score = (
                        (before_angle - after_angle) / max(before_angle, 0.25)
                        - 0.02 * max(0.0, after_xy - xy_limit) / max(xy_limit, 20.0)
                    )
                if not contact_preserved:
                    score -= 5.0 if phase == "translation" else 2.0
                features_by_phase[phase].append(
                    goal_conditioned_ranker_features(
                        state,
                        action,
                        previous_actions[state_index],
                        goal,
                        feature_interface,
                    )
                )
                targets_by_phase[phase].append(float(score))
    heads: dict[str, Any] = {}
    for index, phase in enumerate(("translation", "closure")):
        heads[phase] = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed + 8830 + index,
            n_jobs=1,
        ).fit(
            np.asarray(features_by_phase[phase]),
            np.asarray(targets_by_phase[phase], dtype=np.float64),
        )
    return {
        "translation": heads["translation"],
        "closure": heads["closure"],
        "protocol": {
            "phase_specific": True,
            "scalar_task_utility": True,
            "action_block_matched": True,
            "contact_preservation_penalty": True,
            "successor_verification_at_deployment": False,
        },
    }


def fit_goal_conditioned_phase_policy_heads(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    trees: int,
    seed: int,
    previous_actions: np.ndarray | None = None,
    feature_interface: str = "legacy",
    goals_per_state: int = 8,
) -> dict[str, Any]:
    """Distill the simulator-best fixed-bank action into phase heads.

    Unlike candidate-wise regression, this head is trained on one explicit
    action label per state/goal group.  The simulator is used only to produce
    that offline label.  At deployment, the classifier emits an index into the
    same fixed bank and never evaluates a successor.
    """
    if horizon < 1 or goals_per_state < 2:
        raise ValueError("phase policy needs positive horizon and goals_per_state >= 2")
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if previous_actions is None:
        previous_actions = np.zeros((len(states), 2), dtype=np.float64)
    previous_actions = np.asarray(previous_actions, dtype=np.float64)
    if previous_actions.shape != (len(states), 2):
        raise ValueError("phase policy previous_actions must have shape (n, 2)")
    rng = np.random.default_rng(seed + 99173)
    features_by_phase = {"translation": [], "closure": []}
    labels_by_phase = {"translation": [], "closure": []}
    for state_index, state in enumerate(states):
        endpoints = []
        for action_index, action in enumerate(actions):
            env.reset(
                seed=seed + state_index * 1000 + action_index,
                options={"state": state.copy(), "goal_state": state.copy()},
            )
            endpoint = state.copy()
            for _ in range(horizon):
                observation, _, terminated, truncated, _ = env.step(action)
                endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                if terminated or truncated:
                    break
            endpoints.append(endpoint)
        for _ in range(goals_per_state):
            radius = float(rng.uniform(10.0, 220.0))
            direction = float(rng.uniform(-np.pi, np.pi))
            goal = state.copy()
            goal[2:4] = np.clip(
                state[2:4] + radius * np.asarray([np.cos(direction), np.sin(direction)]),
                20.0,
                492.0,
            )
            goal[4] = float(
                (state[4] + rng.uniform(-np.pi, np.pi) + np.pi) % (2.0 * np.pi) - np.pi
            )
            before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
            before_angle = angle_error(state[4], goal[4])
            phase = "translation" if before_xy >= 45.0 else "closure"
            xy_limit = min(45.0, before_xy + 4.0)
            current_contact = mode_for_state(state) == "contact"
            scores = []
            for endpoint in endpoints:
                after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
                after_angle = angle_error(endpoint[4], goal[4])
                preserved = (
                    mode_for_state(endpoint) == "contact"
                    and float(np.linalg.norm(endpoint[:2] - endpoint[2:4])) <= 65.0
                    if current_contact else True
                )
                if phase == "translation":
                    value = (
                        (before_xy - after_xy) / max(before_xy, 20.0)
                        - 2.0 * max(0.0, after_angle - before_angle - 0.08)
                    )
                else:
                    value = (
                        (before_angle - after_angle) / max(before_angle, 0.25)
                        - 0.02 * max(0.0, after_xy - xy_limit) / max(xy_limit, 20.0)
                    )
                if not preserved:
                    value -= 5.0 if phase == "translation" else 2.0
                scores.append(float(value))
            features_by_phase[phase].append(
                goal_conditioned_ranker_features(
                    state,
                    np.zeros(2, dtype=np.float64),
                    previous_actions[state_index],
                    goal,
                    feature_interface,
                )
            )
            labels_by_phase[phase].append(int(np.argmax(np.asarray(scores))))
    heads: dict[str, Any] = {}
    for index, phase in enumerate(("translation", "closure")):
        labels = np.asarray(labels_by_phase[phase], dtype=np.int64)
        if len(np.unique(labels)) < 2:
            raise ValueError(f"phase policy produced one action class for {phase}")
        heads[phase] = ExtraTreesClassifier(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=1.0,
            class_weight="balanced",
            random_state=seed + 8850 + index,
            n_jobs=1,
        ).fit(np.asarray(features_by_phase[phase]), labels)
    return {
        "translation": heads["translation"],
        "closure": heads["closure"],
        "protocol": {
            "phase_specific": True,
            "best_action_distillation": True,
            "fixed_action_bank_count": int(len(actions)),
            "action_block_matched": True,
            "successor_verification_at_deployment": False,
        },
    }


def choose_goal_conditioned_action(
    model: ExtraTreesRegressor,
    state: np.ndarray,
    goal: np.ndarray,
    directions: int = 24,
    translation_first: bool = False,
    magnitudes: tuple[float, ...] = (0.15, 0.25, 0.35, 0.55, 0.75),
    previous_action: np.ndarray | None = None,
    avoid_action: np.ndarray | None = None,
    feature_interface: str = "legacy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select an action from the direct learned goal-quality predictions."""
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    previous = np.zeros(2, dtype=np.float64) if previous_action is None else np.asarray(previous_action, dtype=np.float64).reshape(2)
    actions = torque_action_bank(directions, magnitudes=magnitudes)
    features = np.asarray([
        goal_conditioned_ranker_features(state, action, previous, goal, feature_interface)
        for action in actions
    ])
    predictions = np.asarray(model.predict(features), dtype=np.float64)
    if predictions.shape != (len(actions), 2) or not np.isfinite(predictions).all():
        raise ValueError("goal-conditioned ranker must return finite (angle, XY) improvements")
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    predicted_angle = np.clip(before_angle - predictions[:, 0], 0.0, np.pi)
    predicted_xy = np.maximum(before_xy - predictions[:, 1], 0.0)
    xy_limit = min(45.0, before_xy + 4.0)
    if translation_first:
        feasible = predicted_xy < before_xy - 1e-5
        values = predicted_xy + 0.10 * predicted_angle
        selection_mode = "goal_ranker_translation_progress"
    else:
        feasible = (predictions[:, 0] > 1e-5) & (predicted_xy <= xy_limit)
        values = predicted_angle + 0.02 * predicted_xy
        selection_mode = "goal_ranker_orientation_progress"
    if avoid_action is not None:
        repeat = np.linalg.norm(actions.astype(np.float64) - np.asarray(avoid_action, dtype=np.float64).reshape(1, 2), axis=1) < 0.12
        values = values.copy()
        values[repeat] += 1.0
    if np.any(feasible):
        values = values.copy()
        values[~feasible] = np.inf
    elif not translation_first:
        values = (
            np.maximum(-predictions[:, 0], 0.0)
            + 0.02 * np.maximum(predicted_xy - xy_limit, 0.0)
            + 0.02 * predicted_xy
        )
        selection_mode = "goal_ranker_violation_fallback"
    selected = int(np.argmin(values))
    return np.asarray(actions[selected], dtype=np.float32), {
        "selected": selected,
        "selection_mode": selection_mode,
        "predicted_angle_improvement": float(predictions[selected, 0]),
        "predicted_xy_improvement": float(predictions[selected, 1]),
        "predicted_angle_error": float(predicted_angle[selected]),
        "predicted_block_xy": float(predicted_xy[selected]),
        "predicted_feasible_count": int(np.sum(feasible)),
        "candidate_effects_finite": True,
        "goal_conditioned_ranker": True,
        "feature_interface": feature_interface,
    }


def choose_goal_conditioned_feasibility_action(
    model: ExtraTreesClassifier | dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    directions: int = 24,
    translation_first: bool = False,
    magnitudes: tuple[float, ...] = (0.15, 0.25, 0.35, 0.55, 0.75),
    previous_action: np.ndarray | None = None,
    avoid_action: np.ndarray | None = None,
    feature_interface: str = "legacy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select with the head matching the current translation/closure phase.

    A dict returned by ``fit_goal_conditioned_phase_feasibility_heads`` is the
    new protocol.  The classifier fallback preserves the older ablation and
    makes its use explicit in the returned audit metadata.
    """
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    previous = (
        np.zeros(2, dtype=np.float64)
        if previous_action is None else np.asarray(previous_action, dtype=np.float64).reshape(2)
    )
    actions = torque_action_bank(directions, magnitudes=magnitudes)
    features = np.asarray([
        goal_conditioned_ranker_features(state, action, previous, goal, feature_interface)
        for action in actions
    ])
    phase = "translation" if translation_first else "closure"
    phase_specific = isinstance(model, dict) and phase in model
    selected_model = model[phase] if phase_specific else model
    probabilities = np.asarray(selected_model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(selected_model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1 or probabilities.shape[1] != len(classes):
        raise ValueError("feasibility classifier must expose a binary positive class")
    scores = probabilities[:, int(positive[0])]
    if not np.isfinite(scores).all():
        raise ValueError("feasibility probabilities must be finite")
    values = -scores
    if avoid_action is not None:
        repeat = np.linalg.norm(
            actions.astype(np.float64) - np.asarray(avoid_action, dtype=np.float64).reshape(1, 2),
            axis=1,
        ) < 0.12
        values = values.copy()
        values[repeat] += 0.05
    selected = int(np.argmin(values))
    selection_mode = (
        "phase_head_translation" if phase_specific and translation_first else
        "phase_head_closure" if phase_specific else
        "legacy_goal_feasibility_translation" if translation_first else
        "legacy_goal_feasibility_closure"
    )
    return np.asarray(actions[selected], dtype=np.float32), {
        "selected": selected,
        "selection_mode": selection_mode,
        "feasibility_phase": phase,
        "phase_specific_feasibility_head": bool(phase_specific),
        "predicted_feasibility": float(scores[selected]),
        "predicted_feasible_count": int(np.sum(scores >= 0.5)),
        "candidate_effects_finite": True,
        "goal_conditioned_feasibility": True,
        "feature_interface": feature_interface,
    }


def choose_goal_conditioned_phase_score_action(
    model: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    directions: int = 24,
    translation_first: bool = False,
    magnitudes: tuple[float, ...] = (0.15, 0.25, 0.35, 0.55, 0.75),
    previous_action: np.ndarray | None = None,
    avoid_action: np.ndarray | None = None,
    feature_interface: str = "legacy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the maximum predicted utility from the matching phase head."""
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    previous = (
        np.zeros(2, dtype=np.float64)
        if previous_action is None else np.asarray(previous_action, dtype=np.float64).reshape(2)
    )
    actions = torque_action_bank(directions, magnitudes=magnitudes)
    features = np.asarray([
        goal_conditioned_ranker_features(state, action, previous, goal, feature_interface)
        for action in actions
    ])
    phase = "translation" if translation_first else "closure"
    if not isinstance(model, dict) or phase not in model:
        raise ValueError("phase score model must expose translation and closure heads")
    scores = np.asarray(model[phase].predict(features), dtype=np.float64).reshape(-1)
    if scores.shape != (len(actions),) or not np.isfinite(scores).all():
        raise ValueError("phase score head must return finite scalar scores")
    values = scores.copy()
    if avoid_action is not None:
        repeat = np.linalg.norm(
            actions.astype(np.float64) - np.asarray(avoid_action, dtype=np.float64).reshape(1, 2),
            axis=1,
        ) < 0.12
        values[repeat] -= 0.05
    selected = int(np.argmax(values))
    return np.asarray(actions[selected], dtype=np.float32), {
        "selected": selected,
        "selection_mode": "phase_score_translation" if translation_first else "phase_score_closure",
        "feasibility_phase": phase,
        "phase_specific_score_head": True,
        "predicted_utility": float(scores[selected]),
        "candidate_utility_max": float(np.max(scores)),
        "candidate_utility_min": float(np.min(scores)),
        "candidate_effects_finite": True,
        "goal_conditioned_phase_score": True,
        "feature_interface": feature_interface,
    }


def choose_goal_conditioned_phase_policy_action(
    model: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    directions: int = 24,
    translation_first: bool = False,
    magnitudes: tuple[float, ...] = (0.15, 0.25, 0.35, 0.55, 0.75),
    previous_action: np.ndarray | None = None,
    feature_interface: str = "legacy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose the fixed action-bank index predicted by the phase policy head."""
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    previous = (
        np.zeros(2, dtype=np.float64)
        if previous_action is None else np.asarray(previous_action, dtype=np.float64).reshape(2)
    )
    actions = torque_action_bank(directions, magnitudes=magnitudes)
    phase = "translation" if translation_first else "closure"
    if not isinstance(model, dict) or phase not in model:
        raise ValueError("phase policy model must expose translation and closure heads")
    features = np.asarray([
        goal_conditioned_ranker_features(
            state, np.zeros(2, dtype=np.float64), previous, goal, feature_interface
        )
    ])
    probabilities = np.asarray(model[phase].predict_proba(features), dtype=np.float64)[0]
    classes = np.asarray(model[phase].classes_, dtype=np.int64)
    if len(classes) == 0 or not np.isfinite(probabilities).all():
        raise ValueError("phase policy head returned invalid probabilities")
    selected_class = int(classes[int(np.argmax(probabilities))])
    if selected_class < 0 or selected_class >= len(actions):
        raise ValueError("phase policy predicted an action outside the fixed bank")
    return np.asarray(actions[selected_class], dtype=np.float32), {
        "selected": selected_class,
        "selection_mode": "phase_policy_translation" if translation_first else "phase_policy_closure",
        "feasibility_phase": phase,
        "phase_specific_policy_head": True,
        "predicted_action_probability": float(np.max(probabilities)),
        "candidate_effects_finite": True,
        "goal_conditioned_phase_policy": True,
        "feature_interface": feature_interface,
    }


def choose_direct_torque_effect_action(
    model: ExtraTreesRegressor,
    state: np.ndarray,
    goal: np.ndarray,
    directions: int = 24,
    translation_first: bool = False,
    magnitudes: tuple[float, ...] = (0.15, 0.25, 0.35, 0.55, 0.75),
    avoid_action: np.ndarray | None = None,
    previous_action: np.ndarray | None = None,
    uncertainty_penalty: float = 0.0,
    contact_mode_penalty: float = 0.0,
    horizon_consistency_model: ExtraTreesRegressor | None = None,
    horizon_consistency_penalty: float = 0.0,
    feature_interface: str = "legacy",
    online_effect_bias: np.ndarray | None = None,
    online_local_history: list[dict[str, Any]] | None = None,
    online_local_strategy: str = "ridge",
    online_local_weight: float = 0.5,
    online_local_radius: float = 0.30,
    online_local_neighbors: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rank a fixed action fan using a direct multi-step learned endpoint.

    The optional short-horizon model is an independent one-step effect
    readout.  It is used only as a consistency check: a two-step candidate
    that predicts target rotation while its one-step readout predicts no
    target-oriented rotation is treated as an extrapolation risk.
    """
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if previous_action is None:
        previous_action = np.zeros(2, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64).reshape(2)
    actions = torque_action_bank(directions, magnitudes=magnitudes)
    features = np.asarray([
        torque_feature_vector(state, action, previous_action, feature_interface)
        for action in actions
    ])
    local_effects = np.asarray(model.predict(features), dtype=np.float64)
    if online_effect_bias is not None:
        bias = np.asarray(online_effect_bias, dtype=np.float64).reshape(-1)
        if bias.shape != (3,) or not np.isfinite(bias).all():
            raise ValueError("online endpoint-effect bias must be a finite vector of shape (3,)")
        local_effects = local_effects + bias[None]
    if online_local_strategy not in {
        "ridge", "knn", "state_knn", "state_residual_knn", "orientation_gain"
    }:
        raise ValueError(
            "online local strategy must be 'ridge', 'knn', 'state_knn', "
            "'state_residual_knn', or 'orientation_gain'"
        )
    if not 0.0 <= float(online_local_weight) <= 1.0:
        raise ValueError("online local weight must be in [0, 1]")
    if float(online_local_radius) <= 0.0:
        raise ValueError("online local radius must be positive")
    if int(online_local_neighbors) < 1:
        raise ValueError("online local neighbors must be positive")
    local_fit_used = False
    local_fit_supported_candidates = 0
    local_fit_min_distance = np.inf
    local_fit_neighbor_count = 0
    if online_local_history is not None and len(online_local_history) >= 4:
        history_mode = (
            mode_for_state(state)
            if online_local_strategy == "orientation_gain"
            else "contact"
        )
        usable = [
            row for row in online_local_history[-16:]
            if row.get("mode") == history_mode
            and np.isfinite(np.asarray(row.get("action_local"), dtype=np.float64)).all()
            and np.isfinite(np.asarray(row.get("effect_local"), dtype=np.float64)).all()
        ]
        if len(usable) >= 4:
            relative = state[2:4] - state[:2]
            normal = unit(relative)
            tangent = np.asarray([-normal[1], normal[0]])
            action_local = np.asarray([
                [float(np.dot(action, normal)), float(np.dot(action, tangent))]
                for action in actions
            ])
            x_hist = np.asarray([row["action_local"] for row in usable], dtype=np.float64)
            y_hist = np.asarray([row["effect_local"] for row in usable], dtype=np.float64)
            if online_local_strategy == "ridge":
                # Legacy ablation: a single smooth fit affects every candidate.
                # It is intentionally retained as a negative control.
                def local_design(value: np.ndarray) -> np.ndarray:
                    value = np.asarray(value, dtype=np.float64)
                    an, at = float(value[0]), float(value[1])
                    return np.asarray([
                        an, at, np.hypot(an, at), an * an, at * at, an * at
                    ])
                online_model = Ridge(alpha=5.0).fit(
                    np.asarray([local_design(value) for value in x_hist]), y_hist
                )
                online_effects = online_model.predict(
                    np.asarray([local_design(value) for value in action_local])
                )
                local_effects = (
                    (1.0 - float(online_local_weight)) * local_effects
                    + float(online_local_weight) * online_effects
                )
                local_fit_used = True
                local_fit_supported_candidates = len(actions)
                local_fit_min_distance = float(
                    np.min(np.linalg.norm(action_local[:, None] - x_hist[None], axis=2))
                )
                local_fit_neighbor_count = len(x_hist)
            elif online_local_strategy == "orientation_gain":
                # The direct endpoint adapter can have a systematic angular
                # gain/sign error after the contact geometry changes.  Fit a
                # scalar only for the observed angular channel and leave the
                # learned translation channels untouched; this avoids the
                # translation damage seen with the old global Ridge repair.
                gain_rows = [
                    row for row in usable
                    if row.get("predicted_local") is not None
                    and np.isfinite(np.asarray(row["predicted_local"], dtype=np.float64)).all()
                ]
                if len(gain_rows) >= 4:
                    predicted_angle_effect = np.asarray([
                        float(np.asarray(row["predicted_local"], dtype=np.float64)[2])
                        for row in gain_rows
                    ])
                    actual_angle_effect = np.asarray([
                        float(np.asarray(row["effect_local"], dtype=np.float64)[2])
                        for row in gain_rows
                    ])
                    denominator = float(np.dot(predicted_angle_effect, predicted_angle_effect))
                    if denominator > 1e-8:
                        orientation_gain = float(np.clip(
                            np.dot(predicted_angle_effect, actual_angle_effect) / denominator,
                            -2.0,
                            2.0,
                        ))
                        local_effects = local_effects.copy()
                        local_effects[:, 2] *= orientation_gain
                        local_fit_used = True
                        local_fit_supported_candidates = len(actions)
                        local_fit_min_distance = 0.0
                        local_fit_neighbor_count = len(gain_rows)
            elif mode_for_state(state) == "contact":
                # Candidate-specific repair: do not extrapolate a local contact
                # sample to the whole fan.  Only candidates covered by a nearby
                # action in the same observed contact mode receive a correction.
                neighbor_count = min(int(online_local_neighbors), len(x_hist))
                residual_fit = online_local_strategy == "state_residual_knn"
                if online_local_strategy in {"state_knn", "state_residual_knn"}:
                    # The action-only KNN used in the earlier ablation treats
                    # two contacts with the same local force as interchangeable.
                    # They are not: PushT's T-shaped contact mode changes with
                    # the observed agent/block ray, block angle, and velocity.
                    # Build a compact *current-state + action* chart.  The
                    # state coordinates are normalized before distance search;
                    # no successor state or goal is used to construct them.
                    def state_action_chart(
                        value_state: np.ndarray,
                        value_action: np.ndarray,
                        value_previous: np.ndarray,
                    ) -> np.ndarray:
                        value_state = np.asarray(value_state, dtype=np.float64)
                        value_action = np.asarray(value_action, dtype=np.float64)
                        value_previous = np.asarray(value_previous, dtype=np.float64)
                        relative = value_state[2:4] - value_state[:2]
                        distance = max(float(np.linalg.norm(relative)), 1e-8)
                        normal = relative / distance
                        tangent = np.asarray([-normal[1], normal[0]])
                        velocity = value_state[5:7]
                        angle = float(value_state[4])
                        rotation = np.asarray([
                            [np.cos(angle), np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)],
                        ])
                        agent_block = rotation @ (value_state[:2] - value_state[2:4])
                        velocity_block = rotation @ velocity
                        action_local_value = np.asarray([
                            float(np.dot(value_action, normal)),
                            float(np.dot(value_action, tangent)),
                        ])
                        previous_local_value = np.asarray([
                            float(np.dot(value_previous, normal)),
                            float(np.dot(value_previous, tangent)),
                        ])
                        # Distances are dimensionless and deliberately coarse:
                        # 50 px contact geometry, 10 px/s velocity, and unit
                        # action are comparable neighbourhood scales.
                        return np.asarray([
                            agent_block[0] / 50.0,
                            agent_block[1] / 50.0,
                            velocity_block[0] / 10.0,
                            velocity_block[1] / 10.0,
                            np.cos(angle),
                            np.sin(angle),
                            action_local_value[0],
                            action_local_value[1],
                            previous_local_value[0],
                            previous_local_value[1],
                        ], dtype=np.float64)

                    history_states = [
                        row.get("state_before") for row in usable
                    ]
                    if all(
                        value is not None
                        and np.isfinite(np.asarray(value, dtype=np.float64)).all()
                        for value in history_states
                    ):
                        x_hist = np.asarray([
                            state_action_chart(
                                row["state_before"],
                                row["action_world"],
                                row.get("previous_action", np.zeros(2)),
                            )
                            for row in usable
                        ], dtype=np.float64)
                        x_candidates = np.asarray([
                            state_action_chart(state, action, previous_action)
                            for action in actions
                        ], dtype=np.float64)
                        if residual_fit:
                            residual_rows = [row.get("residual_local") for row in usable]
                            if all(
                                value is not None
                                and np.isfinite(np.asarray(value, dtype=np.float64)).all()
                                for value in residual_rows
                            ):
                                y_hist = np.asarray(residual_rows, dtype=np.float64)
                            else:
                                residual_fit = False
                    else:
                        # Old artifacts may contain action/effect history only;
                        # in that case fail closed to the action-only protocol.
                        x_hist = None
                        x_candidates = None
                else:
                    x_hist = x_hist
                    x_candidates = action_local
                if x_hist is None or x_candidates is None:
                    online_local_strategy = "knn"
                    x_hist = np.asarray([row["action_local"] for row in usable], dtype=np.float64)
                    x_candidates = action_local
                online_model = KNeighborsRegressor(
                    n_neighbors=neighbor_count,
                    weights="distance",
                ).fit(x_hist, y_hist)
                distances, _ = online_model.kneighbors(
                    x_candidates, n_neighbors=neighbor_count, return_distance=True
                )
                nearest_distance = np.asarray(distances[:, 0], dtype=np.float64)
                supported = nearest_distance <= float(online_local_radius)
                if np.any(supported):
                    online_effects = online_model.predict(x_candidates[supported])
                    support_weight = np.clip(
                        1.0 - nearest_distance[supported] / float(online_local_radius),
                        0.0,
                        1.0,
                    ) * float(online_local_weight)
                    local_effects[supported] = (
                        local_effects[supported]
                        + support_weight[:, None] * online_effects
                        if residual_fit
                        else (1.0 - support_weight[:, None]) * local_effects[supported]
                        + support_weight[:, None] * online_effects
                    )
                    local_fit_used = True
                    local_fit_supported_candidates = int(np.sum(supported))
                    local_fit_min_distance = float(np.min(nearest_distance))
                    local_fit_neighbor_count = int(neighbor_count)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    effects = np.asarray([
        [
            local[0] * normal[0] + local[1] * tangent[0],
            local[0] * normal[1] + local[1] * tangent[1],
            local[2],
        ]
        for local in local_effects
    ])
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    predicted_xy = np.linalg.norm(
        state[2:4][None] + effects[:, :2] - goal[2:4][None], axis=1
    )
    predicted_angle = np.asarray([
        angle_error(state[4] + effect[2], goal[4]) for effect in effects
    ])
    if uncertainty_penalty < 0.0:
        raise ValueError("direct uncertainty penalty must be non-negative")
    if contact_mode_penalty < 0.0:
        raise ValueError("direct contact-mode penalty must be non-negative")
    if horizon_consistency_penalty < 0.0:
        raise ValueError("direct horizon-consistency penalty must be non-negative")
    short_predicted_angle = np.zeros(len(actions), dtype=np.float64)
    horizon_inconsistent = np.zeros(len(actions), dtype=bool)
    if horizon_consistency_model is not None and horizon_consistency_penalty > 0.0:
        short_local_effects = np.asarray(
            horizon_consistency_model.predict(features), dtype=np.float64
        )
        if short_local_effects.shape != local_effects.shape:
            raise ValueError(
                "horizon-consistency model must predict the same three local effects"
            )
        short_effects = np.asarray([
            [
                local[0] * normal[0] + local[1] * tangent[0],
                local[0] * normal[1] + local[1] * tangent[1],
                local[2],
            ]
            for local in short_local_effects
        ])
        short_predicted_angle = np.asarray([
            angle_error(state[4] + effect[2], goal[4]) for effect in short_effects
        ])
        # A two-step endpoint improvement unsupported by the one-step chart
        # is the exact extrapolation failure targeted by this ablation.
        long_improves = predicted_angle < before_angle - 1e-5
        short_improves = short_predicted_angle < before_angle - 1e-5
        horizon_inconsistent = long_improves & ~short_improves
    predicted_xy_uncertainty = np.zeros(len(actions), dtype=np.float64)
    predicted_angle_uncertainty = np.zeros(len(actions), dtype=np.float64)
    if uncertainty_penalty > 0.0 and hasattr(model, "estimators_"):
        tree_local = np.asarray([
            estimator.predict(features) for estimator in model.estimators_
        ], dtype=np.float64)
        tree_effects = np.asarray([
            [
                [local[0] * normal[0] + local[1] * tangent[0],
                 local[0] * normal[1] + local[1] * tangent[1], local[2]]
                for local in tree_prediction
            ]
            for tree_prediction in tree_local
        ])
        tree_xy = np.linalg.norm(
            state[2:4][None, None] + tree_effects[:, :, :2]
            - goal[2:4][None, None], axis=2
        )
        tree_angle = np.asarray([
            [angle_error(state[4] + effect[2], goal[4]) for effect in tree_effects[index]]
            for index in range(len(tree_effects))
        ])
        predicted_xy_uncertainty = np.std(tree_xy, axis=0)
        predicted_angle_uncertainty = np.std(tree_angle, axis=0)
    robust_xy = predicted_xy + uncertainty_penalty * predicted_xy_uncertainty
    robust_angle = predicted_angle + uncertainty_penalty * predicted_angle_uncertainty
    predicted_contact_probability = np.ones(len(actions), dtype=np.float64)
    contact_mode_model = getattr(model, "contact_mode_model_", None)
    if contact_mode_penalty > 0.0 and contact_mode_model is not None:
        contact_probability = contact_mode_model.predict_proba(features)
        classes = np.asarray(contact_mode_model.classes_)
        contact_index = np.flatnonzero(classes == 1)
        if len(contact_index) == 1:
            predicted_contact_probability = np.asarray(
                contact_probability[:, int(contact_index[0])], dtype=np.float64
            )
    if translation_first:
        feasible = robust_xy < before_xy - 1e-5
        values = robust_xy + 0.10 * robust_angle
        selection_mode = "direct_predicted_translation_progress"
    else:
        xy_limit = min(45.0, before_xy + 4.0)
        feasible = (robust_angle < before_angle - 1e-5) & (robust_xy <= xy_limit)
        if horizon_consistency_model is not None and horizon_consistency_penalty > 0.0:
            feasible = feasible & ~horizon_inconsistent
        values = robust_angle + 0.02 * robust_xy
        values = values + horizon_consistency_penalty * horizon_inconsistent.astype(float)
        selection_mode = "direct_predicted_feasible"
    if avoid_action is not None:
        # A repeated no-motion action is evidence that the current contact
        # mode is not responding to that direction.  Penalize only a close
        # repeat in the learned ranking; this is an observation-based
        # anti-stall rule, not a simulator verification or geometric route.
        avoid_action = np.asarray(avoid_action, dtype=np.float64).reshape(2)
        repeat = np.linalg.norm(actions.astype(np.float64) - avoid_action[None], axis=1) < 0.12
        values = np.asarray(values, dtype=np.float64).copy()
        values[repeat] += 1.0
    if np.any(feasible):
        values = values.copy()
        values[~feasible] = np.inf
    else:
        if not translation_first:
            # A zero-action fallback is not safe here: it can strand the
            # controller forever after a model misranks every candidate by a
            # small amount.  Stay purely learned, but select the candidate
            # with the smallest predicted constraint violation.  This keeps
            # the fallback auditable and preserves the fixed two-step
            # endpoint protocol without simulator probing.
            violation = (
                np.maximum(robust_angle - before_angle, 0.0)
                + 0.02 * np.maximum(robust_xy - xy_limit, 0.0)
                + 0.02 * robust_xy
            )
            values = violation + horizon_consistency_penalty * horizon_inconsistent.astype(float)
            if avoid_action is not None:
                repeat = np.linalg.norm(
                    actions.astype(np.float64) - avoid_action[None], axis=1
                ) < 0.12
                values[repeat] += 1.0
            selection_mode = "direct_predicted_violation_fallback"
    if contact_mode_penalty > 0.0 and contact_mode_model is not None:
        values = np.asarray(values, dtype=np.float64) + float(contact_mode_penalty) * (
            1.0 - predicted_contact_probability
        )
        if np.any(feasible):
            values[~feasible] = np.inf
    selected = int(np.argmin(values))
    return np.asarray(actions[selected], dtype=np.float32), {
        "selected": selected,
        "selection_mode": selection_mode,
        "predicted_effect": effects[selected].copy(),
        "predicted_block_xy": float(predicted_xy[selected]),
        "predicted_angle_error": float(predicted_angle[selected]),
        "short_horizon_predicted_angle_error": float(short_predicted_angle[selected]),
        "horizon_consistency_penalty": float(horizon_consistency_penalty),
        "horizon_inconsistent": bool(horizon_inconsistent[selected]),
        "horizon_consistency_supported_count": int(np.sum(~horizon_inconsistent)),
        "predicted_xy_uncertainty": float(predicted_xy_uncertainty[selected]),
        "predicted_angle_uncertainty": float(predicted_angle_uncertainty[selected]),
        "uncertainty_penalty": float(uncertainty_penalty),
        "contact_mode_penalty": float(contact_mode_penalty),
        "predicted_contact_probability": float(predicted_contact_probability[selected]),
        "predicted_contact_probability_min": float(np.min(predicted_contact_probability)),
        "feature_interface": feature_interface,
        "online_effect_bias": (
            None if online_effect_bias is None
            else np.asarray(online_effect_bias, dtype=np.float64).copy()
        ),
        "online_local_fit_used": local_fit_used,
        "online_local_strategy": online_local_strategy,
        "online_local_weight": float(online_local_weight),
        "online_local_radius": float(online_local_radius),
        "online_local_neighbors": int(online_local_neighbors),
        "online_local_supported_candidates": int(local_fit_supported_candidates),
        "online_local_min_distance": (
            None if not np.isfinite(local_fit_min_distance)
            else float(local_fit_min_distance)
        ),
        "online_local_neighbor_count": int(local_fit_neighbor_count),
        "predicted_feasible_count": int(feasible.sum()),
        "candidate_effects_finite": bool(np.isfinite(effects).all()),
    }


def direct_sequence_features(
    state: np.ndarray,
    first_action: np.ndarray,
    second_action: np.ndarray,
    previous_action: np.ndarray,
    feature_interface: str = "legacy",
) -> np.ndarray:
    """Encode a two-action contact block in the same local frame."""
    return np.concatenate([
        torque_feature_vector(state, first_action, previous_action, feature_interface),
        torque_feature_vector(state, second_action, first_action, feature_interface),
    ]).astype(np.float64)


def fit_direct_contact_sequence_model(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    trees: int,
    seed: int,
    previous_actions: np.ndarray | None = None,
    feature_interface: str = "legacy",
) -> ExtraTreesRegressor:
    """Fit a learned endpoint map for two distinct constant-action steps.

    Unlike the constant-action endpoint adapter, this map can represent the
    short push--reorient or reorient--push transition that a single repeated
    action cannot express.  Labels are generated offline from the simulator;
    selection at planning time only evaluates the fitted regressor.
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if len(states) == 0 or len(actions) == 0:
        raise ValueError("sequence endpoint training requires non-empty states/actions")
    if previous_actions is None:
        previous_actions = np.zeros((len(states), 2), dtype=np.float64)
    previous_actions = np.asarray(previous_actions, dtype=np.float64)
    if previous_actions.shape != (len(states), 2):
        raise ValueError("sequence endpoint previous_actions must have shape (n, 2)")
    features = []
    targets = []
    for state_index, state in enumerate(states):
        previous = previous_actions[state_index]
        for first_index, first_action in enumerate(actions):
            for second_index, second_action in enumerate(actions):
                env.reset(
                    seed=seed + state_index * 100000 + first_index * 1000 + second_index,
                    options={"state": state.copy(), "goal_state": state.copy()},
                )
                endpoint = state.copy()
                for action in (first_action, second_action):
                    observation, _, terminated, truncated, _ = env.step(action)
                    endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                    if terminated or truncated:
                        break
                relative = state[2:4] - state[:2]
                normal = unit(relative)
                tangent = np.asarray([-normal[1], normal[0]])
                delta = endpoint[2:4] - state[2:4]
                features.append(direct_sequence_features(
                    state, first_action, second_action, previous, feature_interface
                ))
                targets.append([
                    float(np.dot(delta, normal)),
                    float(np.dot(delta, tangent)),
                    float((endpoint[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi),
                ])
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed + 2002,
        n_jobs=1,
    )
    return model.fit(np.asarray(features), np.asarray(targets))


def choose_direct_torque_sequence_action(
    model: ExtraTreesRegressor,
    state: np.ndarray,
    goal: np.ndarray,
    directions: int = 8,
    magnitudes: tuple[float, ...] = (0.2, 0.4, 0.6),
    translation_first: bool = False,
    previous_action: np.ndarray | None = None,
    avoid_action: np.ndarray | None = None,
    feature_interface: str = "legacy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rank two-action sequences using only the learned endpoint map."""
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    previous = (
        np.zeros(2, dtype=np.float64)
        if previous_action is None else np.asarray(previous_action, dtype=np.float64).reshape(2)
    )
    actions = torque_action_bank(directions, magnitudes=magnitudes)
    sequences = np.asarray([
        [first, second] for first in actions for second in actions
    ], dtype=np.float64)
    features = np.asarray([
        direct_sequence_features(
            state, sequence[0], sequence[1], previous, feature_interface
        )
        for sequence in sequences
    ])
    local_effects = np.asarray(model.predict(features), dtype=np.float64)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    effects = np.asarray([
        [
            local[0] * normal[0] + local[1] * tangent[0],
            local[0] * normal[1] + local[1] * tangent[1],
            local[2],
        ]
        for local in local_effects
    ])
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    predicted_xy = np.linalg.norm(
        state[2:4][None] + effects[:, :2] - goal[2:4][None], axis=1
    )
    predicted_angle = np.asarray([
        angle_error(state[4] + effect[2], goal[4]) for effect in effects
    ])
    xy_limit = min(45.0, before_xy + 4.0)
    if translation_first:
        feasible = predicted_xy < before_xy - 1e-5
        values = predicted_xy + 0.10 * predicted_angle
        selection_mode = "direct_sequence_predicted_translation_progress"
    else:
        feasible = (
            (predicted_angle < before_angle - 1e-5)
            & (predicted_xy <= xy_limit)
        )
        values = predicted_angle + 0.02 * predicted_xy
        selection_mode = "direct_sequence_predicted_feasible"
    if avoid_action is not None:
        repeat = np.linalg.norm(
            sequences[:, 0] - np.asarray(avoid_action, dtype=np.float64)[None], axis=1
        ) < 0.12
        values = np.asarray(values, dtype=np.float64).copy()
        values[repeat] += 1.0
    if np.any(feasible):
        values = values.copy()
        values[~feasible] = np.inf
    else:
        if translation_first:
            values = predicted_xy + 0.10 * predicted_angle
            selection_mode = "direct_sequence_translation_violation_fallback"
        else:
            values = (
                np.maximum(predicted_angle - before_angle, 0.0)
                + 0.02 * np.maximum(predicted_xy - xy_limit, 0.0)
                + 0.02 * predicted_xy
            )
            selection_mode = "direct_sequence_violation_fallback"
        if avoid_action is not None:
            values[repeat] += 1.0
    selected = int(np.argmin(values))
    return np.asarray(sequences[selected, 0], dtype=np.float32), {
        "selected": selected,
        "selection_mode": selection_mode,
        "predicted_effect": effects[selected].copy(),
        "predicted_block_xy": float(predicted_xy[selected]),
        "predicted_angle_error": float(predicted_angle[selected]),
        "predicted_feasible_count": int(feasible.sum()),
        "candidate_effects_finite": bool(np.isfinite(effects).all()),
        "execution_action_sequence": sequences[selected].astype(np.float32),
        "sequence_action_bank_size": int(len(actions)),
        "sequence_count": int(len(sequences)),
        "feature_interface": feature_interface,
    }


def select_torque_effect_model(
    model: dict[str, Any], state: np.ndarray
) -> ExtraTreesRegressor:
    """Route the local-effect readout using only the current observed mode."""
    specialists = model.get("torque_effect_specialists")
    if isinstance(specialists, dict):
        return specialists.get(mode_for_state(state), specialists["fallback"])
    return model["torque_effect"]


def fit_agent_only_model(
    data: dict[str, np.ndarray],
    trees: int,
    seed: int,
) -> ExtraTreesRegressor:
    """Fit only the agent XY increment for the hybrid rollout interface."""
    features = np.asarray([
        planner_feature_vector(state, action, previous, "block_agent_hybrid")
        for state, action, previous in zip(
            data["states"], data["actions"], data["previous_actions"]
        )
    ])
    targets = np.asarray([
        state_features(next_state)[:2] - state_features(state)[:2]
        for state, next_state in zip(data["states"], data["next_states"])
    ])
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(features, targets)
    return model


def fit_contact_gate(
    data: dict[str, np.ndarray],
    model_name: str,
    trees: int,
    seed: int,
) -> ExtraTreesClassifier:
    """Fit a gate that decides whether a predicted block update is allowed."""
    features = np.asarray([
        planner_feature_vector(state, action, previous, model_name)
        for state, action, previous in zip(
            data["states"], data["actions"], data["previous_actions"]
        )
    ])
    labels = np.asarray([
        float(
            np.linalg.norm(next_state[2:4] - state[2:4]) > 1e-4
            or angle_error(next_state[4], state[4]) > 1e-4
        )
        for state, next_state in zip(data["states"], data["next_states"])
    ], dtype=np.int64)
    gate = ExtraTreesClassifier(
        n_estimators=max(32, trees // 2),
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    )
    gate.fit(features, labels)
    return gate


def fit_block_only_specialists(
    data: dict[str, np.ndarray],
    trees: int,
    seed: int,
) -> dict[str, ExtraTreesRegressor]:
    """Fit mode-conditional block-pose adapters with a pooled fallback.

    Contact effects are discontinuous across the free/approach/contact
    boundary.  A single regressor averages those modes; the routed models
    keep the feature interface and target fixed while changing only the
    training stratum.  Small or absent strata use the pooled model and are
    recorded as a fallback rather than silently training on too few cases.
    """
    pooled = fit_block_only_model(data, trees, seed)
    specialists: dict[str, ExtraTreesRegressor] = {"fallback": pooled}
    modes = np.asarray(data["modes"])
    for index, mode in enumerate(("free", "approach", "contact")):
        mask = modes == mode
        if int(mask.sum()) < 32:
            specialists[mode] = pooled
            continue
        subset = {key: value[mask] for key, value in data.items()}
        specialists[mode] = fit_block_only_model(subset, trees, seed + 101 + index)
    states = np.asarray(data["states"], dtype=np.float64)
    actions = np.asarray(data["actions"], dtype=np.float64)
    relative = states[:, 2:4] - states[:, :2]
    normal = relative / np.maximum(np.linalg.norm(relative, axis=1, keepdims=True), 1e-8)
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    normal_component = np.abs(np.sum(actions * normal, axis=1))
    tangent_component = np.abs(np.sum(actions * tangent, axis=1))
    torque_mask = (
        np.isin(modes, ("approach", "contact"))
        & (tangent_component >= 0.75 * normal_component)
        & (tangent_component >= 0.15)
    )
    if int(torque_mask.sum()) >= 32:
        specialists["torque"] = fit_block_only_model(
            {key: value[torque_mask] for key, value in data.items()},
            trees,
            seed + 404,
        )
    else:
        specialists["torque"] = pooled
    return specialists


def staged_contact_action(state: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, str] | None:
    """Return a bounded acquire/return action for the staged repair.

    PushT actions are relative target displacements, not velocities.  A
    learned endpoint score is therefore unsafe while the agent is still far
    from the contact manifold: it can repeatedly choose a hallucinated block
    improvement and accumulate agent drift.  This small analytic controller
    handles only the two geometric phases that do not require learned block
    dynamics.  The learned adapter remains responsible for contact-time push
    selection.
    """
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    block = state[2:4]
    agent = state[:2]
    block_error = float(np.linalg.norm(block - goal[2:4]))
    angle = angle_error(state[4], goal[4])
    if block_error < 20.0 and angle < np.pi / 9.0:
        target = goal[:2]
        phase = "return"
    else:
        contact_distance = float(np.linalg.norm(agent - block))
        # The simulator's contact manifold is wider than the strict
        # ``mode_for_state == contact`` diagnostic boundary.  A settled
        # bootstrap can therefore be 45--65 px away while already providing
        # a valid pushing normal.  Treating that state as acquisition sends
        # the agent away from a usable contact and was the hidden stage-gate
        # failure in the previous closed-loop runs.
        if contact_distance <= 65.0:
            return None
        push_direction = unit(goal[2:4] - block)
        contact = block - 42.0 * push_direction
        relative = agent - block
        # If the agent is on the opposite side, use a side waypoint first so
        # the direct target command does not cut through the T.
        if float(np.dot(relative, push_direction)) > -10.0:
            tangent = np.asarray([-push_direction[1], push_direction[0]])
            cross = float(push_direction[0] * relative[1] - push_direction[1] * relative[0])
            side = 1.0 if cross >= 0.0 else -1.0
            target = block + side * 90.0 * tangent - 55.0 * push_direction
        else:
            target = contact
        phase = "acquire"
    # The simulator applies the command through a PD controller.  A 100-pixel
    # target jump carries too much velocity into contact; the smaller gain is
    # the same bounded approach scale used by the validated oracle route.
    action = np.clip((target - agent) / 180.0, -1.0, 1.0).astype(np.float32)
    return action, phase


def update_translation_closure_lock(
    locked: bool,
    state: np.ndarray,
    goal: np.ndarray,
    enter_xy: float = 20.0,
    exit_xy: float = 26.0,
) -> bool:
    """Apply hysteresis to the translation-closed terminal subgoal.

    The lock is entered when block translation satisfies the strict PushT
    margin, but it is not released for a small numerical excursion.  This
    prevents the controller from alternating between torque search and
    translation reacquisition at the boundary, which was the source of the
    repeated block drift in the failed traces.
    """
    if exit_xy < enter_xy:
        raise ValueError("closure hysteresis exit must not be below entry")
    xy = float(np.linalg.norm(np.asarray(state, dtype=np.float64)[2:4] - np.asarray(goal, dtype=np.float64)[2:4]))
    return xy < (exit_xy if locked else enter_xy)


def feedback_contact_action(
    state: np.ndarray,
    goal: np.ndarray,
    sweep_index: int,
) -> tuple[np.ndarray, str, int]:
    """Choose one bounded contact route from observed state only.

    The learned adapter is deliberately bypassed for this short intervention.
    The action is selected from a fixed normal/tangential fan and its effect is
    judged on the next observed simulator state.  This is a feedback control
    diagnostic, not privileged rollout scoring: no future simulator state is
    queried before the action is executed.
    """
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    agent, block = state[:2], state[2:4]
    push_direction = unit(goal[2:4] - block)
    tangent = np.asarray([-push_direction[1], push_direction[0]])
    offsets = (0.0, np.pi / 3.0, -np.pi / 3.0, np.pi / 2.0, -np.pi / 2.0, np.pi)
    offset = offsets[int(sweep_index) % len(offsets)]
    c, s = np.cos(offset), np.sin(offset)
    direction = np.asarray([
        c * push_direction[0] - s * push_direction[1],
        s * push_direction[0] + c * push_direction[1],
    ])
    relative = agent - block
    contact_distance = float(np.linalg.norm(relative))
    if contact_distance > 65.0:
        target = block - 48.0 * direction
        scale = 180.0
        phase = "feedback_acquire"
    else:
        # First move to a usable side of the block if the current contact
        # normal would pull the agent across the object; otherwise apply a
        # bounded push target in the swept direction.
        if float(np.dot(relative, direction)) > -10.0:
            target = block + 90.0 * tangent - 55.0 * direction
            scale = 180.0
            phase = "feedback_reacquire"
        else:
            target = block + 86.0 * direction
            scale = 100.0
            phase = "feedback_push"
    action = np.clip((target - agent) / scale, -1.0, 1.0).astype(np.float32)
    return action, phase, int(sweep_index) + 1


def local_effect_action_bank(
    state: np.ndarray,
    goal: np.ndarray,
    directions: int = 24,
    scale: float = 0.55,
) -> np.ndarray:
    """Construct a fixed contact-force fan for one-step local probing.

    The fan is state-conditioned only through the observed contact geometry;
    it does not query a future simulator state.  Each direction is converted
    to the same target-displacement action convention used by PushT.  The
    resulting bank is shared by all probe decisions and is deliberately small
    enough to expose the method as a local-effect diagnostic rather than a
    hidden global planner.
    """
    if directions < 4 or scale <= 0.0:
        raise ValueError("local-effect bank requires at least four directions")
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    block = state[2:4]
    agent = state[:2]
    desired = unit(goal[2:4] - block)
    angle_delta = float((goal[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi)
    sign = 1.0 if angle_delta >= 0.0 else -1.0
    offsets = np.linspace(-np.pi, np.pi, directions, endpoint=False)
    actions = []
    for offset in offsets:
        c, s = np.cos(offset), np.sin(offset)
        force = np.asarray([
            c * desired[0] - s * desired[1],
            s * desired[0] + c * desired[1],
        ])
        # Include the sign of the required rotation in the ordering, while
        # retaining the full fan so that the simulator—not a hand-coded torque
        # model—selects the useful side of the T.
        if sign < 0.0:
            force = np.asarray([force[0], -force[1]])
        target = block + 82.0 * force
        actions.append(np.clip((target - agent) / 100.0, -1.0, 1.0) * scale / 0.55)
    actions.append(np.zeros(2, dtype=np.float64))
    return np.asarray(actions, dtype=np.float32)


def local_effect_score(
    before: np.ndarray,
    after: np.ndarray,
    goal: np.ndarray,
    tube: float = 45.0,
    xy_margin: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Lexicographically preserve translation while closing orientation."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    before_xy = float(np.linalg.norm(before[2:4] - goal[2:4]))
    after_xy = float(np.linalg.norm(after[2:4] - goal[2:4]))
    after_angle = angle_error(after[4], goal[4])
    before_angle = angle_error(before[4], goal[4])
    success = bool(after_xy < 20.0 and after_angle < np.pi / 9.0)
    translation_limit = float(tube)
    if xy_margin is not None:
        translation_limit = min(translation_limit, before_xy + max(float(xy_margin), 0.0))
    leaves_tube = bool(after_xy > translation_limit)
    # Once translation is near the target, an orientation improvement is the
    # primary objective.  Leaving the tube is a hard violation, not something
    # that a weighted sum may trade for a small angle gain.
    score = (
        1_000_000.0 if leaves_tube else 0.0
    ) + 1_000.0 * after_angle + 0.02 * after_xy
    if after_angle > before_angle + 1e-6:
        score += 0.25
    return float(score), {
        "before_xy": before_xy,
        "after_xy": after_xy,
        "before_angle": before_angle,
        "after_angle": after_angle,
        "translation_limit": translation_limit,
        "leaves_translation_tube": leaves_tube,
        "task_success": success,
    }


def choose_local_effect_action(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    directions: int = 24,
    scale: float = 0.55,
    xy_margin: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Probe one simulator step and return the safest orientation action.

    This is intentionally an online, privileged-state diagnostic.  The
    current state is reset before every probe, and only the observed one-step
    successor is used to rank that probe.  The selected action is committed
    by the caller from the same current state.
    """
    current = np.asarray(state, dtype=np.float64).copy()
    goal = np.asarray(goal, dtype=np.float64).copy()
    bank = local_effect_action_bank(current, goal, directions, scale)
    rows = []
    for index, action in enumerate(bank):
        env.reset(seed=0, options={"state": current.copy(), "goal_state": goal.copy()})
        observation, _, terminated, truncated, _ = env.step(action)
        successor = np.asarray(observation["state"], dtype=np.float64).copy()
        value, metrics = local_effect_score(
            current, successor, goal, xy_margin=xy_margin
        )
        rows.append({
            "index": index,
            "action": action.copy(),
            "value": value,
            "terminated": bool(terminated or truncated),
            **metrics,
        })
    rows.sort(key=lambda item: (not item["task_success"], item["value"]))
    selected = rows[0]
    # Probe rollouts reset the environment and leave it at the last probe
    # successor. Restore the real current state before the caller commits the
    # selected action, keeping probe ranking and execution state-consistent.
    env.reset(seed=0, options={"state": current.copy(), "goal_state": goal.copy()})
    return np.asarray(selected["action"], dtype=np.float32), {
        "phase": "local_effect_orientation",
        "selected_probe": int(selected["index"]),
        "probe_count": len(rows),
        "probe_value": float(selected["value"]),
        "probe_metrics": {key: value for key, value in selected.items()
                          if key not in ("action",)},
        "all_probe_values_finite": bool(np.isfinite([row["value"] for row in rows]).all()),
    }


def collect_contact_intervention_data(
    env: gym.Env,
    count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Collect action-balanced one-step effects on held-out contact states.

    The ordinary behavior policy is useful for representation diagnostics but
    undersamples the torque directions that the planner proposes.  We first
    collect a larger pool of genuine trajectories, then intervene once from
    disjoint approach/contact states using normal, tangential, reverse, and
    mixed actions.  The state pool remains simulator-derived; the intervention
    only changes the one-step action and is never used for evaluation cases.
    """
    pool = collect_trajectory_data(env, max(count * 3, 1024), seed + 300000)
    modes = np.asarray(pool["modes"])
    preferred = np.flatnonzero(np.isin(modes, ("approach", "contact")))
    if len(preferred) == 0:
        preferred = np.arange(len(pool["states"]))
    rng = np.random.default_rng(seed + 310000)
    # Keep a complete local directional fan paired on the same source state.
    # The earlier one-action-per-state schedule left the torque readout to
    # interpolate across unrelated contact geometries.  A balanced block of
    # normal/tangential and mixed actions makes the local effect identifiable
    # without increasing the transition budget.  The three magnitudes are
    # interleaved so every source state sees both direction and scale changes.
    action_templates_count = 12
    base_count = int(np.ceil(count / action_templates_count))
    selected = rng.choice(preferred, size=base_count, replace=len(preferred) < base_count)
    states, actions, previous, next_states, selected_modes = [], [], [], [], []
    for row, index in enumerate(selected):
        state = np.asarray(pool["states"][index], dtype=np.float64).copy()
        relative = state[2:4] - state[:2]
        normal = unit(relative)
        tangent = np.asarray([-normal[1], normal[0]])
        offsets = np.linspace(0.0, 2.0 * np.pi, action_templates_count, endpoint=False)
        magnitudes = (0.35, 0.55, 0.75)
        action_templates = tuple(
            magnitudes[offset % len(magnitudes)]
            * (np.cos(offset_angle) * normal + np.sin(offset_angle) * tangent)
            for offset, offset_angle in enumerate(offsets)
        )
        for offset, template in enumerate(action_templates):
            if len(states) >= count:
                break
            action = np.asarray(template, dtype=np.float64)
            action = np.clip(
                action + 0.03 * rng.normal(size=2), -1.0, 1.0
            ).astype(np.float32)
            observation, _ = env.reset(
                seed=seed + 320000 + row * len(action_templates) + offset,
                options={"state": state.copy()},
            )
            observation, _, _, _, _ = env.step(action)
            states.append(state)
            actions.append(action)
            previous.append(np.asarray(pool["previous_actions"][index], dtype=np.float64).copy())
            next_states.append(np.asarray(observation["state"], dtype=np.float64).copy())
            selected_modes.append(modes[index])
    return {
        "states": np.asarray(states, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.float64),
        "previous_actions": np.asarray(previous, dtype=np.float64),
        "next_states": np.asarray(next_states, dtype=np.float64),
        "modes": np.asarray(selected_modes),
    }


def candidate_bank(
    rng: np.random.Generator,
    state: np.ndarray,
    goal: np.ndarray,
    count: int,
    horizon: int,
    action_magnitude: float,
) -> np.ndarray:
    """Build a fixed guided/random bank shared by all learned interfaces."""
    if count < 8 or horizon < 1:
        raise ValueError("candidate count and horizon are too small")
    bank = rng.uniform(-1.0, 1.0, size=(count, horizon, 2)).astype(np.float32)
    block_direction = unit(np.asarray(goal[2:4] - state[2:4], dtype=np.float64))
    tangent = np.asarray([-block_direction[1], block_direction[0]])
    relative = np.asarray(state[2:4] - state[:2], dtype=np.float64)
    approach = unit(relative)
    templates = [
        np.zeros(2),
        action_magnitude * block_direction,
        -action_magnitude * block_direction,
        action_magnitude * tangent,
        -action_magnitude * tangent,
        action_magnitude * approach,
        -action_magnitude * approach,
        0.75 * action_magnitude * block_direction + 0.45 * action_magnitude * tangent,
    ]
    # Contact-time torque is not determined by the desired block translation
    # alone. Add force-direction proposals around the current contact normal;
    # these are still shared candidates, not privileged per-model choices.
    angle_delta = float((goal[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi)
    tangential_sign = 1.0 if angle_delta >= 0.0 else -1.0
    for offset in (tangential_sign * np.pi / 2.0,
                   tangential_sign * np.pi / 4.0,
                   -tangential_sign * np.pi / 4.0,
                   -tangential_sign * np.pi / 2.0):
        c, s = np.cos(offset), np.sin(offset)
        force = np.asarray([
            c * block_direction[0] - s * block_direction[1],
            s * block_direction[0] + c * block_direction[1],
        ])
        templates.append(np.clip((state[2:4] + 72.0 * force - state[:2]) / 100.0,
                                 -1.0, 1.0))
    for index, action in enumerate(templates):
        if index >= count:
            break
        bank[index] = np.clip(action, -1.0, 1.0)
    # Add a small number of phase-changing candidates to represent approach
    # followed by push, while keeping the bank identical across model arms.
    if count > len(templates):
        phase = np.zeros((horizon, 2), dtype=np.float32)
        split = max(1, horizon // 2)
        phase[:split] = np.clip(0.9 * approach, -1.0, 1.0)
        phase[split:] = np.clip(action_magnitude * block_direction, -1.0, 1.0)
        bank[len(templates)] = phase
    return bank


def rollout_candidate(
    model: ExtraTreesRegressor,
    state: np.ndarray,
    previous_action: np.ndarray,
    actions: np.ndarray,
    model_name: str,
) -> np.ndarray:
    current = np.asarray(state, dtype=np.float64).copy()
    previous = np.asarray(previous_action, dtype=np.float64).copy()
    for action in np.asarray(actions, dtype=np.float64):
        current = predict_step(model, current, action, previous, model_name)
        previous = np.asarray(action, dtype=np.float64).copy()
    return current


def rollout_candidate_batch(
    model: ExtraTreesRegressor,
    state: np.ndarray,
    previous_action: np.ndarray,
    candidates: np.ndarray,
    model_name: str,
    contact_gate: ExtraTreesClassifier | None = None,
    contact_threshold: float = 0.85,
) -> np.ndarray:
    """Roll out all candidates with one tree-ensemble call per horizon step."""
    current = np.repeat(np.asarray(state, dtype=np.float64)[None], len(candidates), axis=0)
    previous = np.repeat(np.asarray(previous_action, dtype=np.float64)[None], len(candidates), axis=0)
    for actions in np.asarray(candidates, dtype=np.float64).transpose(1, 0, 2):
        features = np.asarray([
            planner_feature_vector(row, action, old_action, model_name)
            for row, action, old_action in zip(current, actions, previous)
        ])
        predictor = model["block"] if model_name == "block_agent_hybrid" else model
        delta = np.asarray(predictor.predict(features), dtype=np.float64)
        if contact_gate is not None:
            contact_probability = contact_gate.predict_proba(features)[:, 1]
            # Free-space transitions should not move the block.  Without this
            # gate, small regression noise is compounded over the candidate
            # horizon and becomes a false contact signal.
            delta[contact_probability < contact_threshold, 2:6] = 0.0
        if model_name == "block_agent_hybrid":
            agent_features = np.asarray([
                planner_feature_vector(row, action, old_action, model_name)
                for row, action, old_action in zip(current, actions, previous)
            ])
            agent_delta = np.asarray(model["agent"].predict(agent_features), dtype=np.float64)
            current[:, :2] = np.clip(current[:, :2] + agent_delta, 5.0, 507.0)
            block_delta = np.asarray(model["block"].predict(features), dtype=np.float64)
            block = np.asarray([
                state_features(row)[2:6] for row in current
            ], dtype=np.float64) + block_delta
            block[:, :2] = np.clip(block[:, :2], 5.0, 507.0)
            norm = np.linalg.norm(block[:, 2:4], axis=1, keepdims=True)
            block[:, 2:4] /= np.maximum(norm, 1e-8)
            current[:, 2:4] = block[:, :2]
            current[:, 4] = np.arctan2(block[:, 2], block[:, 3])
            previous = np.asarray(actions, dtype=np.float64)
            continue
        if model_name in (
            "block_only", "block_only_specialist", "block_only_torque_readout"
        ):
            block = np.asarray([
                state_features(row)[2:6] for row in current
            ], dtype=np.float64) + delta
            block[:, :2] = np.clip(block[:, :2], 5.0, 507.0)
            norm = np.linalg.norm(block[:, 2:4], axis=1, keepdims=True)
            block[:, 2:4] /= np.maximum(norm, 1e-8)
            current[:, 2:4] = block[:, :2]
            current[:, 4] = np.arctan2(block[:, 2], block[:, 3])
            previous = np.asarray(actions, dtype=np.float64)
            continue
        physical = np.asarray([state_features(row) for row in current], dtype=np.float64) + delta
        physical = normalize_features(physical)
        current = np.stack([
            physical[:, 0], physical[:, 1], physical[:, 2], physical[:, 3],
            np.arctan2(physical[:, 4], physical[:, 5]), physical[:, 6], physical[:, 7],
        ], axis=1)
        previous = np.asarray(actions, dtype=np.float64)
    return current


def choose_action(
    model: ExtraTreesRegressor,
    model_name: str,
    state: np.ndarray,
    goal: np.ndarray,
    previous_action: np.ndarray,
    candidates: np.ndarray,
    planning_objective: str = "joint_block_pose",
    contact_gate: ExtraTreesClassifier | None = None,
    contact_threshold: float = 0.85,
) -> tuple[np.ndarray, dict[str, Any]]:
    predicted = rollout_candidate_batch(
        model,
        state,
        previous_action,
        candidates,
        model_name,
        contact_gate=contact_gate,
        contact_threshold=contact_threshold,
    )
    current_distance = float(np.linalg.norm(state[:2] - state[2:4]))

    def score(item: np.ndarray) -> float:
        pose = block_pose_cost(item, goal)
        if planning_objective == "joint_block_pose":
            return pose
        if planning_objective not in (
            "contact_gated", "staged_contact_orient", "staged_translation"
        ):
            raise ValueError(f"unknown planning objective: {planning_objective}")
        predicted_distance = float(np.linalg.norm(item[:2] - item[2:4]))
        predicted_motion = float(np.linalg.norm(item[2:4] - state[2:4]))
        # Before contact, block pose is not an actionable objective: a model
        # can hallucinate block motion from a distant state and repeatedly
        # drive the agent away from the contact manifold. Acquire contact
        # first, then optimize pose and reject stationary predictions.
        if current_distance > 45.0:
            return 2.0 * predicted_distance + 0.10 * pose + 0.5 * predicted_motion
        if predicted_motion < 0.5 and pose > 20.0:
            return pose + 1000.0
        if planning_objective == "staged_translation":
            current_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
            predicted_xy = float(np.linalg.norm(item[2:4] - goal[2:4]))
            # Translation is a separate subgoal.  Orientation is deliberately
            # excluded until the block enters the near-goal tube.
            if current_xy >= 45.0:
                return predicted_xy + 0.02 * predicted_distance
            return predicted_xy + 0.01 * angle_error(item[4], goal[4])
        if planning_objective == "staged_contact_orient":
            current_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
            if current_xy < 45.0:
                predicted_xy = float(np.linalg.norm(item[2:4] - goal[2:4]))
                predicted_angle = angle_error(item[4], goal[4])
                # PushT termination is conjunctive.  A weighted sum can
                # accept an angle improvement while leaving XY just outside
                # the 20-pixel boundary (or vice versa).  Use the worst
                # normalized violation, with a small sum tie-breaker.
                xy_margin = predicted_xy / 20.0
                angle_margin = predicted_angle / (np.pi / 9.0)
                return max(xy_margin, angle_margin) + 0.01 * (xy_margin + angle_margin)
        return pose

    values = np.asarray([score(item) for item in predicted])
    selected = int(np.argmin(values))
    selected_action = np.asarray(candidates[selected, 0], dtype=np.float32)
    gate_probability = None
    if contact_gate is not None:
        gate_features = planner_feature_vector(
            state, selected_action, previous_action, model_name
        )[None]
        gate_probability = float(contact_gate.predict_proba(gate_features)[0, 1])
    return selected_action, {
        "selected": selected,
        "predicted_cost": float(values[selected]),
        "predicted_block_xy": float(np.linalg.norm(predicted[selected, 2:4] - goal[2:4])),
        "predicted_contact_distance": float(
            np.linalg.norm(predicted[selected, :2] - predicted[selected, 2:4])
        ),
        "predicted_block_motion": float(
            np.linalg.norm(predicted[selected, 2:4] - state[2:4])
        ),
        "planning_objective": planning_objective,
        "candidate_cost_finite": bool(np.isfinite(values).all()),
        "contact_gate_probability": gate_probability,
        "contact_gate_threshold": float(contact_threshold) if contact_gate is not None else None,
    }


def torque_action_bank(
    directions: int = 24,
    magnitudes: tuple[float, ...] = (0.35, 0.55, 0.75),
) -> np.ndarray:
    """Return a fixed absolute-action fan for learned local-effect ranking."""
    if directions < 8:
        raise ValueError("torque action bank requires at least eight directions")
    offsets = np.linspace(-np.pi, np.pi, directions, endpoint=False)
    actions = [np.zeros(2, dtype=np.float32)]
    for magnitude in magnitudes:
        for offset in offsets:
            actions.append(np.asarray(
                [magnitude * np.cos(offset), magnitude * np.sin(offset)],
                dtype=np.float32,
            ))
    return np.asarray(actions, dtype=np.float32)


def choose_torque_effect_action(
    model: ExtraTreesRegressor,
    state: np.ndarray,
    goal: np.ndarray,
    previous_action: np.ndarray,
    directions: int = 24,
    xy_margin: float = 4.0,
    translation_first: bool = False,
    feature_mode: str = "world_frame",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rank local actions with a learned torque readout and hard XY tube."""
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    actions = torque_action_bank(directions)
    if feature_mode == "world_frame":
        features = np.asarray([
            planner_feature_vector(state, action, previous_action, "block_only")
            for action in actions
        ])
    elif feature_mode in ("contact_frame", "target_relative", "block_target_frame"):
        features = np.asarray([
            torque_feature_vector(
                state, action, previous_action,
                (
                    "legacy" if feature_mode == "contact_frame"
                    else feature_mode
                ),
            )
            for action in actions
        ])
    else:
        raise ValueError(f"unknown torque feature mode: {feature_mode}")
    predicted_effects = np.asarray(model.predict(features), dtype=np.float64)
    if feature_mode in ("contact_frame", "target_relative", "block_target_frame"):
        relative = np.asarray(state[2:4] - state[:2], dtype=np.float64)
        normal = unit(relative)
        tangent = np.asarray([-normal[1], normal[0]])
        effects = np.asarray([
            [
                effect[0] * normal[0] + effect[1] * tangent[0],
                effect[0] * normal[1] + effect[1] * tangent[1],
                effect[2],
            ]
            for effect in predicted_effects
        ], dtype=np.float64)
    else:
        effects = predicted_effects
    current_angle = float(state[4])
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    before_angle = angle_error(current_angle, goal[4])
    xy_limit = min(45.0, before_xy + max(float(xy_margin), 0.0))
    predicted_xy = np.linalg.norm(
        state[2:4][None] + effects[:, :2] - goal[2:4][None], axis=1
    )
    predicted_angle_values = np.asarray([
        angle_error(current_angle + effect[2], goal[4]) for effect in effects
    ])
    angle_improvement = before_angle - predicted_angle_values
    if translation_first:
        # Far from the terminal translation tube, local contact actions should
        # first make progress toward the block position.  Requiring torque
        # closure here recreates the original joint-endpoint conflict.
        xy_improvement = before_xy - predicted_xy
        feasible = xy_improvement > 1e-5
        values = predicted_xy + 0.10 * predicted_angle_values
        selection_mode = "predicted_translation_progress"
    else:
        feasible = (angle_improvement > 1e-5) & (predicted_xy <= xy_limit)
        values = predicted_angle_values + 0.02 * predicted_xy
        selection_mode = "predicted_feasible"
    if np.any(feasible):
        values[~feasible] = np.inf
    else:
        if not translation_first:
            # Once translation is already inside the terminal tube, an
            # uncalibrated torque guess is worse than holding the state.  The
            # zero action is an explicit recoverability safeguard; it keeps
            # the controller from re-opening the translation error merely to
            # satisfy a noisy angle prediction.
            selected = 0
            return np.asarray(actions[selected], dtype=np.float32), {
                "selected": selected,
                "selection_mode": "zero_safety_fallback",
                "predicted_effect": effects[selected].copy(),
                "predicted_block_xy": float(predicted_xy[selected]),
                "predicted_angle_error": float(predicted_angle_values[selected]),
                "predicted_angle_improvement": float(angle_improvement[selected]),
                "predicted_feasible_count": 0,
                "translation_limit": float(xy_limit),
                "candidate_effects_finite": bool(np.isfinite(effects).all()),
            }
        xy_violation = np.maximum(predicted_xy - xy_limit, 0.0) / max(xy_limit, 1.0)
        angle_violation = np.maximum(predicted_angle_values - before_angle, 0.0)
        if translation_first:
            values = predicted_xy + 0.10 * predicted_angle_values
            selection_mode = "least_translation_violation_fallback"
        else:
            values = 100.0 * xy_violation + 100.0 * angle_violation + predicted_angle_values
            selection_mode = "least_violation_fallback"
    selected = int(np.argmin(values))
    return np.asarray(actions[selected], dtype=np.float32), {
        "selected": selected,
        "selection_mode": selection_mode,
        "predicted_effect": effects[selected].copy(),
        "predicted_block_xy": float(predicted_xy[selected]),
        "predicted_angle_error": float(predicted_angle_values[selected]),
        "predicted_angle_improvement": float(angle_improvement[selected]),
        "predicted_feasible_count": int(feasible.sum()),
        "translation_limit": float(xy_limit),
        "candidate_effects_finite": bool(np.isfinite(effects).all()),
        "feature_mode": feature_mode,
    }


def choose_verified_contact_sequence_action(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    action_bank: np.ndarray,
    horizon: int = 2,
    translation_first: bool = True,
    hard_xy_limit: float | None = None,
    path_xy_margin: float = 8.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify short contact sequences and return only the first action.

    A one-step effect can be locally uninformative when the pusher must first
    settle on a new contact normal.  This bounded sequence probe tests that
    hypothesis from the *observed* simulator state.  It is intentionally a
    privileged verification/recovery mechanism: the learned adapter proposes
    actions, while the simulator supplies the short-horizon successor check.
    Intermediate XY errors are retained as a pathwise constraint so that a
    sequence cannot win by making a large first-step excursion and recovering
    only at its endpoint.
    """
    state = np.asarray(state, dtype=np.float64).copy()
    goal = np.asarray(goal, dtype=np.float64).copy()
    actions = np.asarray(action_bank, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2 or len(actions) == 0:
        raise ValueError("sequence action bank must have shape (n, 2)")
    if horizon < 1 or horizon > 4:
        raise ValueError("sequence verification horizon must be in [1, 4]")
    # Deduplicate while preserving deterministic proposal order.  The cap is
    # a computational guard for smoke/full runs, not a hidden random search.
    actions = np.unique(np.round(np.clip(actions, -1.0, 1.0), decimals=7), axis=0)
    if len(actions) > 56:
        actions = actions[:56]
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    limit = None if hard_xy_limit is None else float(hard_xy_limit)
    if limit is not None and limit <= 0.0:
        raise ValueError("hard XY limit must be positive")

    # A full Cartesian product is appropriate for the default two-step
    # fallback (at most 56^2 probes).  For longer horizons, beam expansion
    # keeps the same pathwise criterion without an exponential blow-up.
    prefixes: list[tuple[list[np.ndarray], np.ndarray, float, float]] = [
        ([], state.copy(), before_xy, before_angle)
    ]
    beam_width = min(96, len(actions) ** max(1, horizon - 1))
    for depth in range(horizon):
        expanded = []
        for sequence, current, max_xy, _ in prefixes:
            for action in actions:
                env.reset(seed=0, options={"state": current.copy(), "goal_state": goal.copy()})
                observation, _, terminated, truncated, _ = env.step(action)
                successor = np.asarray(observation["state"], dtype=np.float64).copy()
                successor_xy = float(np.linalg.norm(successor[2:4] - goal[2:4]))
                successor_angle = angle_error(successor[4], goal[4])
                expanded.append((
                    sequence + [action.copy()],
                    successor,
                    max(max_xy, successor_xy),
                    successor_angle,
                    bool(terminated or truncated),
                ))
        if depth + 1 < horizon:
            def beam_key(item):
                sequence, current, max_xy, current_angle, terminated = item
                current_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
                path_penalty = max(0.0, max_xy - before_xy - max(float(path_xy_margin), 0.0))
                return current_xy + 0.10 * current_angle + 20.0 * path_penalty
            expanded.sort(key=beam_key)
            prefixes = [item[:4] for item in expanded[:beam_width]]
        else:
            finals = expanded

    scored = []
    for sequence, final, max_xy, final_angle, terminated in finals:
        final_xy = float(np.linalg.norm(final[2:4] - goal[2:4]))
        path_violation = max(0.0, max_xy - before_xy - max(float(path_xy_margin), 0.0))
        hard_violation = 0.0 if limit is None else max(0.0, max_xy - limit)
        if translation_first:
            feasible = final_xy < before_xy - 1e-5 and hard_violation <= 1e-6
            value = final_xy + 0.10 * final_angle + 100.0 * path_violation + 100.0 * hard_violation
        else:
            feasible = (
                final_angle < before_angle - 1e-5
                and (limit is None or final_xy <= limit + 1e-6)
                and hard_violation <= 1e-6
            )
            value = final_angle + 0.02 * final_xy + 100.0 * path_violation + 100.0 * hard_violation
        scored.append({
            "sequence": sequence,
            "final_xy": final_xy,
            "final_angle": final_angle,
            "max_path_xy": max_xy,
            "path_violation": path_violation,
            "hard_violation": hard_violation,
            "value": float(value),
            "feasible": bool(feasible),
            "terminated": bool(terminated),
        })
    scored.sort(key=lambda item: (not item["feasible"], item["value"]))
    selected = scored[0]
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    return np.asarray(selected["sequence"][0], dtype=np.float32), {
        "selection_mode": "verified_sequence" if selected["feasible"] else "best_sequence_fallback",
        "sequence_horizon": int(horizon),
        "sequence_bank_size": int(len(actions)),
        "sequence_count": int(len(scored)),
        "sequence_feasible_count": int(sum(item["feasible"] for item in scored)),
        "sequence_final_block_xy": float(selected["final_xy"]),
        "sequence_final_angle_error": float(selected["final_angle"]),
        "sequence_max_path_block_xy": float(selected["max_path_xy"]),
        "sequence_path_violation": float(selected["path_violation"]),
        "sequence_hard_violation": float(selected["hard_violation"]),
        "sequence_action": np.asarray(selected["sequence"], dtype=np.float32),
        "sequence_candidate_effects_finite": bool(np.isfinite(actions).all()),
    }


def choose_verified_cem_action(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    seed_actions: np.ndarray,
    translation_first: bool,
    hard_xy_limit: float | None,
    population: int = 96,
    iterations: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a small one-step CEM over observed simulator successors.

    This is a bounded fallback for states where the fixed fan has no useful
    action.  It is intentionally a simulator-state verification procedure,
    not a learned-dynamics claim: the learned and geometric proposals only
    initialize the search distribution, and every sampled action is evaluated
    from the same observed state.
    """
    state = np.asarray(state, dtype=np.float64).copy()
    goal = np.asarray(goal, dtype=np.float64).copy()
    seeds = np.asarray(seed_actions, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 2 or len(seeds) == 0:
        raise ValueError("CEM seed actions must have shape (n, 2)")
    if population < 16 or iterations < 1:
        raise ValueError("CEM population/iterations are too small")
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    limit = None if hard_xy_limit is None else float(hard_xy_limit)
    rng = np.random.default_rng(seed)
    mean = np.mean(np.clip(seeds, -1.0, 1.0), axis=0)
    std = np.maximum(np.std(np.clip(seeds, -1.0, 1.0), axis=0), 0.20)
    best = None
    rollouts = 0
    for iteration in range(iterations):
        samples = np.clip(
            mean[None] + std[None] * rng.normal(size=(population, 2)), -1.0, 1.0
        ).astype(np.float32)
        samples[: min(len(seeds), population)] = np.clip(
            seeds[: min(len(seeds), population)], -1.0, 1.0
        )
        scored = []
        for action in samples:
            env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
            observation, _, terminated, truncated, _ = env.step(action)
            successor = np.asarray(observation["state"], dtype=np.float64).copy()
            after_xy = float(np.linalg.norm(successor[2:4] - goal[2:4]))
            after_angle = angle_error(successor[4], goal[4])
            xy_violation = 0.0 if limit is None else max(0.0, after_xy - limit)
            if translation_first:
                value = after_xy + 0.10 * after_angle + 20.0 * xy_violation
                feasible = after_xy < before_xy - 1e-5
            else:
                value = after_angle + 0.02 * after_xy + 20.0 * xy_violation
                feasible = after_angle < before_angle - 1e-5 and xy_violation <= 1e-6
            row = {
                "action": action.copy(),
                "value": float(value),
                "feasible": bool(feasible),
                "after_xy": after_xy,
                "after_angle": after_angle,
                "terminated": bool(terminated or truncated),
            }
            scored.append(row)
            rollouts += 1
        scored.sort(key=lambda row: (not row["feasible"], row["value"]))
        if best is None or (not scored[0]["feasible"], scored[0]["value"]) < (
            not best["feasible"], best["value"]
        ):
            best = scored[0]
        elite = scored[: max(8, population // 8)]
        mean = 0.35 * mean + 0.65 * np.mean([row["action"] for row in elite], axis=0)
        std = 0.35 * std + 0.65 * np.maximum(
            np.std([row["action"] for row in elite], axis=0), 0.05
        )
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    return np.asarray(best["action"], dtype=np.float32), {
        "selection_mode": "verified_cem" if best["feasible"] else "verified_cem_fallback",
        "cem_population": int(population),
        "cem_iterations": int(iterations),
        "cem_rollouts": int(rollouts),
        "cem_after_xy": float(best["after_xy"]),
        "cem_after_angle_error": float(best["after_angle"]),
        "cem_feasible": bool(best["feasible"]),
        "cem_candidate_effects_finite": bool(np.isfinite(seeds).all()),
    }


def choose_verified_path_cem_action(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    seed_actions: np.ndarray,
    horizon: int = 4,
    population: int = 96,
    iterations: int = 2,
    path_xy_limit: float = 45.0,
    objective: str = "translation",
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Search a short, path-constrained contact sequence from one state.

    ``translation`` is the original near-goal XY repair objective.  The
    ``orientation`` objective is deliberately separate: it searches for a
    multi-step contact-mode change that reduces angle error while keeping the
    block inside an explicit XY tube.  This closes the gap left by a one-step
    torque fan without pretending that the learned readout alone identified
    the whole contact sequence; every candidate is still checked in the
    simulator from the same observed state.
    """
    state = np.asarray(state, dtype=np.float64).copy()
    goal = np.asarray(goal, dtype=np.float64).copy()
    seeds = np.asarray(seed_actions, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 2 or len(seeds) == 0:
        raise ValueError("path CEM seed actions must have shape (n, 2)")
    if horizon < 2 or population < 16 or iterations < 1:
        raise ValueError("path CEM settings are too small")
    if objective not in ("translation", "orientation"):
        raise ValueError("path CEM objective must be translation or orientation")
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    rng = np.random.default_rng(seed)
    mean = np.repeat(np.mean(np.clip(seeds, -1.0, 1.0), axis=0)[None], horizon, axis=0)
    std = np.maximum(np.repeat(np.std(np.clip(seeds, -1.0, 1.0), axis=0)[None], horizon, axis=0), 0.22)
    best = None
    rollouts = 0
    for _ in range(iterations):
        samples = np.clip(
            mean[None] + std[None] * rng.normal(size=(population, horizon, 2)), -1.0, 1.0
        ).astype(np.float32)
        for index in range(min(len(seeds), population)):
            # Repeating a verified contact proposal is a useful sequence
            # warm-start.  Initialising only the first action leaves the
            # remaining horizon centred at the symmetric zero-force mean and
            # makes the search needlessly blind to persistent torque.
            samples[index] = np.repeat(
                np.clip(seeds[index], -1.0, 1.0)[None], horizon, axis=0
            )
        scored = []
        for sequence in samples:
            env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
            current = state.copy()
            path_xy = [before_xy]
            path_angle = [before_angle]
            terminated = False
            for action in sequence:
                observation, _, terminated, truncated, _ = env.step(action)
                current = np.asarray(observation["state"], dtype=np.float64).copy()
                path_xy.append(float(np.linalg.norm(current[2:4] - goal[2:4])))
                path_angle.append(angle_error(current[4], goal[4]))
                if terminated or truncated:
                    break
            final_xy = path_xy[-1]
            final_angle = path_angle[-1]
            path_violation = max(0.0, max(path_xy) - path_xy_limit)
            if objective == "translation":
                # Prefer strict endpoint translation closure, then a safe
                # path, then angle.
                value = final_xy + 0.10 * final_angle + 20.0 * path_violation
                feasible = final_xy < 20.0 and path_violation <= 1e-6
            else:
                # Orientation repair is allowed to use several contact
                # impulses, but it may not buy torque by ejecting the block
                # from the local XY tube.  A small final-XY term breaks ties
                # between equivalent torque solutions.
                value = final_angle + 0.02 * final_xy + 20.0 * path_violation
                feasible = (
                    final_angle < before_angle - 1e-5
                    and final_xy <= path_xy_limit + 1e-6
                    and path_violation <= 1e-6
                )
            scored.append({
                "action": sequence[0].copy(),
                "sequence": sequence.copy(),
                "value": float(value),
                "feasible": bool(feasible),
                "final_xy": float(final_xy),
                "final_angle": float(final_angle),
                "max_path_xy": float(max(path_xy)),
                "max_path_angle": float(max(path_angle)),
            })
            rollouts += 1
        scored.sort(key=lambda row: (not row["feasible"], row["value"]))
        if best is None or (not scored[0]["feasible"], scored[0]["value"]) < (
            not best["feasible"], best["value"]
        ):
            best = scored[0]
        elite = scored[: max(8, population // 8)]
        mean = 0.35 * mean + 0.65 * np.mean([row["sequence"] for row in elite], axis=0)
        std = 0.35 * std + 0.65 * np.maximum(
            np.std([row["sequence"] for row in elite], axis=0), 0.06
        )
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    return np.asarray(best["action"], dtype=np.float32), {
        "selection_mode": "verified_path_cem" if best["feasible"] else "verified_path_cem_fallback",
        "path_cem_horizon": int(horizon),
        "path_cem_population": int(population),
        "path_cem_iterations": int(iterations),
        "path_cem_rollouts": int(rollouts),
        "path_cem_final_xy": float(best["final_xy"]),
        "path_cem_final_angle_error": float(best["final_angle"]),
        "path_cem_max_path_xy": float(best["max_path_xy"]),
        "path_cem_max_path_angle_error": float(best["max_path_angle"]),
        "path_cem_objective": objective,
        "path_cem_before_xy": float(before_xy),
        "path_cem_before_angle_error": float(before_angle),
        "path_cem_feasible": bool(best["feasible"]),
        "path_cem_candidate_effects_finite": bool(np.isfinite(seeds).all()),
    }


def choose_verified_torque_effect_action(
    env: gym.Env,
    model: ExtraTreesRegressor,
    state: np.ndarray,
    goal: np.ndarray,
    previous_action: np.ndarray,
    directions: int = 24,
    xy_margin: float = 4.0,
    topk: int = 8,
    translation_first: bool = False,
    feature_mode: str = "contact_frame",
    hard_xy_limit: float | None = None,
    verify_all: bool = False,
    zero_on_no_feasible: bool = False,
    extra_actions: np.ndarray | None = None,
    sequence_fallback_actions: np.ndarray | None = None,
    sequence_fallback_horizon: int = 2,
    cem_fallback: bool = False,
    cem_seed: int = 0,
    path_cem_fallback: bool = False,
    path_cem_seed: int = 0,
    orientation_path_cem_fallback: bool = False,
    orientation_path_cem_seed: int = 0,
    orientation_path_xy_limit: float | None = None,
    orientation_path_horizon: int = 6,
    orientation_path_population: int = 128,
    orientation_path_iterations: int = 3,
    verification_steps: int = 1,
    online_effect_bias: np.ndarray | None = None,
    online_effect_mode: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify learned torque proposals with current-state simulator probes.

    ``hard_xy_limit`` implements the terminal translation tube.  It is
    deliberately an absolute limit rather than the old per-step
    ``before_xy + margin`` rule: the latter allows small regressions to
    accumulate over many replans.  ``verify_all`` is reserved for the
    terminal stabilizer and turns the learned readout into a proposal stage;
        the simulator supplies a bounded successor check whose horizon is
        explicitly recorded and can match the committed action block.
    """
    if topk < 1:
        raise ValueError("torque verification top-k must be positive")
    if verification_steps < 1 or verification_steps > 8:
        raise ValueError("verification steps must be in [1, 8]")
    state = np.asarray(state, dtype=np.float64).copy()
    goal = np.asarray(goal, dtype=np.float64).copy()
    previous_action = np.asarray(previous_action, dtype=np.float64).copy()
    actions = torque_action_bank(directions)
    if extra_actions is not None:
        extras = np.asarray(extra_actions, dtype=np.float32)
        if extras.ndim == 1:
            extras = extras[None]
        if extras.ndim != 2 or extras.shape[1] != 2:
            raise ValueError("extra actions must have shape (n, 2)")
        actions = np.concatenate([actions, np.clip(extras, -1.0, 1.0)], axis=0)
    if feature_mode in ("contact_frame", "target_relative", "block_target_frame"):
        features = np.asarray([
            torque_feature_vector(
                state, action, previous_action,
                (
                    "legacy" if feature_mode == "contact_frame"
                    else feature_mode
                ),
            )
            for action in actions
        ])
    elif feature_mode == "world_frame":
        features = np.asarray([
            planner_feature_vector(state, action, previous_action, "block_only")
            for action in actions
        ])
    else:
        raise ValueError(f"unknown torque feature mode: {feature_mode}")
    predicted_local = np.asarray(model.predict(features), dtype=np.float64)
    if online_effect_bias is not None:
        bias = np.asarray(online_effect_bias, dtype=np.float64).reshape(-1)
        if bias.shape != (3,) or not np.isfinite(bias).all():
            raise ValueError("online torque-effect bias must be a finite vector of shape (3,)")
        predicted_local = predicted_local + bias[None]
    if feature_mode in ("contact_frame", "target_relative", "block_target_frame"):
        relative = state[2:4] - state[:2]
        normal = unit(relative)
        tangent = np.asarray([-normal[1], normal[0]])
        predicted = np.asarray([
            [effect[0] * normal[0] + effect[1] * tangent[0],
             effect[0] * normal[1] + effect[1] * tangent[1], effect[2]]
            for effect in predicted_local
        ])
    else:
        predicted = predicted_local
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    if hard_xy_limit is not None:
        xy_limit = float(hard_xy_limit)
        if xy_limit <= 0.0:
            raise ValueError("hard XY limit must be positive")
    else:
        xy_limit = min(45.0, before_xy + max(float(xy_margin), 0.0))
    predicted_xy = np.linalg.norm(
        state[2:4][None] + predicted[:, :2] - goal[2:4][None], axis=1
    )
    predicted_angle = np.asarray([
        angle_error(state[4] + effect[2], goal[4]) for effect in predicted
    ])
    predicted_improvement = before_angle - predicted_angle
    if translation_first:
        predicted_values = predicted_xy + 0.10 * predicted_angle
        predicted_feasible = predicted_xy < before_xy
    else:
        predicted_values = predicted_angle + 0.02 * predicted_xy
        predicted_feasible = (predicted_improvement > 1e-5) & (predicted_xy <= xy_limit)
    if np.any(predicted_feasible):
        predicted_values = predicted_values.copy()
        predicted_values[~predicted_feasible] = np.inf
    proposal_indices = np.argsort(predicted_values)
    if not np.isfinite(predicted_values).any():
        proposal_indices = np.argsort(predicted_xy + 0.1 * predicted_angle)
    if verify_all:
        proposal_indices = np.arange(len(actions), dtype=int)
    else:
        proposal_indices = proposal_indices[:min(int(topk), len(actions))]
        # A safe hold must always be available even when the learned ranking
        # omits the zero action from its top-k proposal set.
        if zero_on_no_feasible and 0 not in proposal_indices:
            proposal_indices = np.unique(np.r_[proposal_indices, 0])
    verified = []
    for index in proposal_indices:
        env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
        terminated = False
        truncated = False
        successor = state.copy()
        for _ in range(int(verification_steps)):
            observation, _, terminated, truncated, _ = env.step(actions[int(index)])
            successor = np.asarray(observation["state"], dtype=np.float64).copy()
            if terminated or truncated:
                break
        after_xy = float(np.linalg.norm(successor[2:4] - goal[2:4]))
        after_angle = angle_error(successor[4], goal[4])
        if translation_first:
            value = after_xy + 0.10 * after_angle
            feasible = after_xy < before_xy
        else:
            value, metrics = local_effect_score(
                state,
                successor,
                goal,
                tube=xy_limit if hard_xy_limit is not None else 45.0,
                xy_margin=None if hard_xy_limit is not None else xy_margin,
            )
            feasible = bool(metrics["angle_improved"] if "angle_improved" in metrics else
                            after_angle < before_angle - 1e-5)
            if hard_xy_limit is not None:
                feasible = feasible and after_xy <= xy_limit + 1e-9
        verified.append({
            "index": int(index),
            "value": float(value),
            "after_xy": after_xy,
            "after_angle": after_angle,
            "angle_improvement": float(before_angle - after_angle),
            "feasible": bool(feasible),
            "terminated": bool(terminated or truncated),
            "verification_steps": int(verification_steps),
            "successor_state": successor.copy(),
        })
    verified.sort(key=lambda row: (not row["feasible"], row["value"]))
    if (
        path_cem_fallback
        and translation_first
        and before_xy < 45.0
        and not any(row["feasible"] for row in verified)
    ):
        path_action, path_info = choose_verified_path_cem_action(
            env,
            state,
            goal,
            actions if extra_actions is None else np.concatenate([actions, np.asarray(extra_actions, dtype=np.float32)], axis=0),
            seed=path_cem_seed,
        )
        return path_action, {
            "selected": None,
            "selection_mode": path_info["selection_mode"],
            "verified_count": len(verified),
            "predicted_feasible_count": int(predicted_feasible.sum()),
            "verified_feasible_count": 0,
            "predicted_block_xy": None,
            "predicted_angle_error": None,
            "actual_block_xy": None,
            "actual_angle_error": None,
            "actual_angle_improvement": None,
            "translation_limit": float(xy_limit),
            "candidate_effects_finite": bool(np.isfinite(predicted).all()),
            "feature_mode": feature_mode,
            "hard_xy_limit": None if hard_xy_limit is None else float(hard_xy_limit),
            "verify_all": bool(verify_all),
            "extra_action_count": 0 if extra_actions is None else int(np.asarray(extra_actions).reshape(-1, 2).shape[0]),
            "sequence_fallback": False,
            "cem_fallback": False,
            "path_cem_fallback": True,
            **path_info,
        }
    if (
        orientation_path_cem_fallback
        and not translation_first
        and before_xy < 45.0
        and before_angle >= np.pi / 9.0
        and not any(row["feasible"] for row in verified)
    ):
        orientation_limit = (
            float(orientation_path_xy_limit)
            if orientation_path_xy_limit is not None
            else min(35.0, max(24.0, before_xy + 6.0))
        )
        path_action, path_info = choose_verified_path_cem_action(
            env,
            state,
            goal,
            actions if extra_actions is None else np.concatenate([
                actions, np.asarray(extra_actions, dtype=np.float32)
            ], axis=0),
            horizon=orientation_path_horizon,
            population=orientation_path_population,
            iterations=orientation_path_iterations,
            path_xy_limit=orientation_limit,
            objective="orientation",
            seed=orientation_path_cem_seed,
        )
        return path_action, {
            "selected": None,
            "selection_mode": path_info["selection_mode"],
            "verified_count": len(verified),
            "predicted_feasible_count": int(predicted_feasible.sum()),
            "verified_feasible_count": 0,
            "predicted_block_xy": None,
            "predicted_angle_error": None,
            "actual_block_xy": None,
            "actual_angle_error": None,
            "actual_angle_improvement": None,
            "translation_limit": float(xy_limit),
            "candidate_effects_finite": bool(np.isfinite(predicted).all()),
            "feature_mode": feature_mode,
            "hard_xy_limit": None if hard_xy_limit is None else float(hard_xy_limit),
            "verify_all": bool(verify_all),
            "extra_action_count": 0 if extra_actions is None else int(np.asarray(extra_actions).reshape(-1, 2).shape[0]),
            "sequence_fallback": False,
            "cem_fallback": False,
            "path_cem_fallback": False,
            "orientation_path_cem_fallback": True,
            **path_info,
        }
    if cem_fallback and not any(row["feasible"] for row in verified):
        cem_action, cem_info = choose_verified_cem_action(
            env,
            state,
            goal,
            actions if extra_actions is None else np.concatenate([actions, np.asarray(extra_actions, dtype=np.float32)], axis=0),
            translation_first=translation_first,
            hard_xy_limit=hard_xy_limit,
            seed=cem_seed,
        )
        return cem_action, {
            "selected": None,
            "selection_mode": cem_info["selection_mode"],
            "verified_count": len(verified),
            "predicted_feasible_count": int(predicted_feasible.sum()),
            "verified_feasible_count": 0,
            "predicted_block_xy": None,
            "predicted_angle_error": None,
            "actual_block_xy": None,
            "actual_angle_error": None,
            "actual_angle_improvement": None,
            "translation_limit": float(xy_limit),
            "candidate_effects_finite": bool(np.isfinite(predicted).all()),
            "feature_mode": feature_mode,
            "hard_xy_limit": None if hard_xy_limit is None else float(hard_xy_limit),
            "verify_all": bool(verify_all),
            "extra_action_count": 0 if extra_actions is None else int(np.asarray(extra_actions).reshape(-1, 2).shape[0]),
            "sequence_fallback": False,
            "cem_fallback": True,
            **cem_info,
        }
    sequence_info = None
    if (
        sequence_fallback_actions is not None
        and not any(row["feasible"] for row in verified)
    ):
        sequence_action, sequence_info = choose_verified_contact_sequence_action(
            env,
            state,
            goal,
            sequence_fallback_actions,
            horizon=sequence_fallback_horizon,
            translation_first=translation_first,
            hard_xy_limit=hard_xy_limit,
        )
        env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
        return sequence_action, {
            "selected": None,
            "selection_mode": sequence_info["selection_mode"],
            "verified_count": len(verified),
            "predicted_feasible_count": int(predicted_feasible.sum()),
            "verified_feasible_count": 0,
            "predicted_block_xy": None,
            "predicted_angle_error": None,
            "actual_block_xy": None,
            "actual_angle_error": None,
            "actual_angle_improvement": None,
            "translation_limit": float(xy_limit),
            "candidate_effects_finite": bool(np.isfinite(predicted).all()),
            "feature_mode": feature_mode,
            "hard_xy_limit": None if hard_xy_limit is None else float(hard_xy_limit),
            "verify_all": bool(verify_all),
            "extra_action_count": 0 if extra_actions is None else int(np.asarray(extra_actions).reshape(-1, 2).shape[0]),
            "sequence_fallback": True,
            **sequence_info,
        }
    if zero_on_no_feasible and not any(row["feasible"] for row in verified):
        if hard_xy_limit is not None:
            # A zero target does not instantaneously remove contact momentum.
            # When no candidate satisfies the strict tube, select the
            # actually verified successor with the smallest translation
            # violation; this is a bounded braking decision rather than an
            # assumption that zero action means zero block motion.
            selected = min(
                verified,
                key=lambda row: (row["after_xy"], row["after_angle"]),
            )
            selection_mode = "least_xy_violation_hold"
        else:
            selected = next(row for row in verified if row["index"] == 0)
            selection_mode = "zero_hold"
    else:
        selected = verified[0]
        selection_mode = "verified_all" if verify_all else "verified_topk"
    # Every probe reset the environment to ``state`` and left it at a probe
    # successor.  Restore the actual planning state before the caller commits
    # the selected action; otherwise verification and execution are evaluated
    # from different simulator states, invalidating the hard tube guarantee.
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    residual = None
    if feature_mode in ("contact_frame", "target_relative", "block_target_frame") and online_effect_bias is not None:
        relative = state[2:4] - state[:2]
        normal = unit(relative)
        tangent = np.asarray([-normal[1], normal[0]])
        actual_delta_xy = np.asarray(
            [
                selected["successor_state"][2] - state[2],
                selected["successor_state"][3] - state[3],
            ],
            dtype=np.float64,
        )
        actual_local = np.asarray([
            float(np.dot(actual_delta_xy, normal)),
            float(np.dot(actual_delta_xy, tangent)),
            float((selected["successor_state"][4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi),
        ])
        residual = actual_local - predicted_local[selected["index"]]
    return np.asarray(actions[selected["index"]], dtype=np.float32), {
        "selected": int(selected["index"]),
        "selection_mode": selection_mode,
        "verified_count": len(verified),
        "predicted_feasible_count": int(predicted_feasible.sum()),
        "verified_feasible_count": int(sum(row["feasible"] for row in verified)),
        "predicted_block_xy": float(predicted_xy[selected["index"]]),
        "predicted_angle_error": float(predicted_angle[selected["index"]]),
        "actual_block_xy": float(selected["after_xy"]),
        "actual_angle_error": float(selected["after_angle"]),
        "actual_angle_improvement": float(selected["angle_improvement"]),
        "translation_limit": float(xy_limit),
        "candidate_effects_finite": bool(np.isfinite(predicted).all()),
        "feature_mode": feature_mode,
        "hard_xy_limit": None if hard_xy_limit is None else float(hard_xy_limit),
        "verify_all": bool(verify_all),
        "verification_steps": int(verification_steps),
        "extra_action_count": 0 if extra_actions is None else int(np.asarray(extra_actions).reshape(-1, 2).shape[0]),
        "sequence_fallback": False,
        "online_effect_mode": online_effect_mode,
        "online_effect_residual": None if residual is None else residual,
    }


def held_out_cases(env: gym.Env, episodes: int, seed: int) -> list[dict[str, Any]]:
    cases = []
    for index in range(episodes):
        observation, info = env.reset(seed=seed + 500000 + index)
        state = np.asarray(observation["state"], dtype=np.float64).copy()
        goal = np.asarray(info["goal_state"], dtype=np.float64).copy()
        previous = np.zeros(2, dtype=np.float32)
        rng = np.random.default_rng(seed + 600000 + index)
        for step in range(6 + index % 7):
            action = choose_behavior_action(state, step, rng)
            observation, _, terminated, truncated, _ = env.step(action)
            state = np.asarray(observation["state"], dtype=np.float64).copy()
            previous = np.asarray(action, dtype=np.float32).copy()
            if terminated or truncated:
                break
        reset_observation, reset_info = env.reset(
            seed=seed + 700000 + index,
            options={"state": state.copy(), "goal_state": goal.copy()},
        )
        state = np.asarray(reset_observation["state"], dtype=np.float64).copy()
        cases.append({
            "state": state,
            "goal": np.asarray(reset_info["goal_state"], dtype=np.float64).copy(),
            "previous_action": previous,
            "mode": mode_for_state(state),
        })
    return cases


def collect_pure_learned_dagger_states(
    env: gym.Env,
    model: dict[str, Any],
    seed_states: np.ndarray,
    seed_previous_actions: np.ndarray,
    action_block: int,
    directions: int,
    magnitudes: tuple[float, ...],
    feature_interface: str,
    rounds: int,
    states_per_round: int,
    steps_per_rollout: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect states visited by the current pure-learned direct policy.

    This is training-time distribution augmentation.  Action selection uses
    only the current state, goal, previous action, and the learned phase head;
    the simulator is stepped only to obtain the next training state.  No
    successor is inspected to alter the action selected in the rollout.
    """
    if rounds < 1 or states_per_round < 1 or steps_per_rollout < 1:
        return (
            np.empty((0, 7), dtype=np.float64),
            np.empty((0, 7), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
    states = np.asarray(seed_states, dtype=np.float64)
    previous = np.asarray(seed_previous_actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError("DAgger seed states must have shape (n, 7)")
    if previous.shape != (len(states), 2):
        raise ValueError("DAgger seed previous actions must have shape (n, 2)")
    rng = np.random.default_rng(seed + 70001)
    collected_states: list[np.ndarray] = []
    collected_goals: list[np.ndarray] = []
    collected_previous: list[np.ndarray] = []
    for round_index in range(rounds):
        if len(states) == 0:
            break
        chosen = rng.choice(
            len(states),
            size=min(states_per_round, len(states)),
            replace=len(states) < states_per_round,
        )
        next_states = []
        next_previous = []
        for rollout_index, source_index in enumerate(chosen):
            state = states[int(source_index)].copy()
            previous_action = previous[int(source_index)].copy()
            goal = state.copy()
            radius = float(rng.uniform(10.0, 220.0))
            direction = float(rng.uniform(-np.pi, np.pi))
            goal[2:4] = np.clip(
                state[2:4] + radius * np.asarray([np.cos(direction), np.sin(direction)]),
                20.0,
                492.0,
            )
            goal[4] = float(
                (state[4] + rng.uniform(-np.pi, np.pi) + np.pi) % (2.0 * np.pi) - np.pi
            )
            for step_index in range(steps_per_rollout):
                block_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
                block_angle = angle_error(state[4], goal[4])
                if block_xy < 20.0 and block_angle < np.pi / 9.0:
                    break
                if float(np.linalg.norm(state[:2] - state[2:4])) > 65.0:
                    # The direct learned arm is only defined on its contact
                    # manifold.  Do not inject route-controller states into
                    # this DAgger pool; they belong to a separate interface.
                    break
                action, _ = choose_goal_conditioned_feasibility_action(
                    model,
                    state,
                    goal,
                    directions=directions,
                    translation_first=block_xy >= 45.0,
                    magnitudes=magnitudes,
                    previous_action=previous_action,
                    feature_interface=feature_interface,
                )
                collected_states.append(state.copy())
                collected_goals.append(goal.copy())
                collected_previous.append(previous_action.copy())
                env.reset(
                    seed=seed + round_index * 100000 + rollout_index * 1000 + step_index,
                    options={"state": state.copy(), "goal_state": goal.copy()},
                )
                observation = None
                for _ in range(action_block):
                    observation, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                if observation is None:
                    break
                state = np.asarray(observation["state"], dtype=np.float64).copy()
                previous_action = np.asarray(action, dtype=np.float64).copy()
            next_states.append(state)
            next_previous.append(previous_action)
        if next_states:
            states = np.asarray(next_states, dtype=np.float64)
            previous = np.asarray(next_previous, dtype=np.float64)
    if not collected_states:
        return (
            np.empty((0, 7), dtype=np.float64),
            np.empty((0, 7), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
    return (
        np.asarray(collected_states, dtype=np.float64),
        np.asarray(collected_goals, dtype=np.float64),
        np.asarray(collected_previous, dtype=np.float64),
    )


def run_arm(
    env: gym.Env,
    model: ExtraTreesRegressor | dict[str, ExtraTreesRegressor],
    model_name: str,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    contact_gate: ExtraTreesClassifier | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for episode, case in enumerate(cases):
        observation, _ = env.reset(
            seed=args.seed + 900000 + episode,
            options={"state": case["state"].copy(), "goal_state": case["goal"].copy()},
        )
        state = np.asarray(observation["state"], dtype=np.float64).copy()
        goal = case["goal"].copy()
        previous = case["previous_action"].copy()
        route_records = []
        if args.planning_objective in (
            "staged_contact_route", "staged_contact_route_calibrated",
            "staged_contact_local_effect", "staged_contact_torque_readout",
            "staged_contact_direct_torque_readout",
            "staged_contact_direct_sequence_readout",
            "staged_contact_verified_torque", "staged_contact_verified_torque_hold",
            "staged_contact_verified_torque_stabilized",
            "staged_contact_hierarchical",
            "staged_contact_verified_torque_fullfan",
            "staged_contact_verified_torque_hierarchical_fullfan",
            "staged_contact_verified_hybrid_fullfan",
        ):
            route_args = argparse.Namespace(
                clearance=args.route_clearance,
                contact_distance=args.route_contact_distance,
                settle_steps=args.route_settle_steps,
            )
            route_probe_args = argparse.Namespace(
                recovery_clearance=args.route_clearance,
                recovery_contact_distance=args.route_contact_distance,
                recovery_settle_steps=args.route_settle_steps,
                disable_adaptive_angle=False,
                cost_mode="weighted_pose",
                selection_mode=(
                    "contact_reacquisition_verified"
                    if args.planning_objective == "staged_contact_verified_hybrid_fullfan"
                    else "pose"
                ),
                # Near-goal contact-mode changes may require a bounded
                # temporary translation excursion. The probe must recover
                # within this explicit envelope; it is not free exploration.
                reacquisition_xy_margin=30.0,
            )
            push_direction = unit(goal[2:4] - state[2:4])
            state, initial_route = route_to_contact(
                env, state, goal, push_direction, route_args
            )
            route_records.append(initial_route)
            previous = np.zeros(2, dtype=np.float32)
        trace = []
        terminated = False
        truncated = False
        total_return = 0.0
        model_mismatch_steps = 0
        feedback_steps_remaining = 0
        feedback_sweep_index = 0
        hold_objective = args.planning_objective in (
            "staged_contact_verified_torque_hold",
            "staged_contact_verified_torque_stabilized",
            "staged_contact_verified_torque_fullfan",
            "staged_contact_verified_torque_hierarchical_fullfan",
            "staged_contact_verified_hybrid_fullfan",
            # The direct learned endpoint arms also need an explicit terminal
            # interface.  Previously they executed only one ordinary action
            # block after pose closure, so a solved block with a displaced
            # agent was reported as a planner failure without ever invoking
            # the closure-preserving return controller.
            "staged_contact_direct_torque_readout",
            "staged_contact_direct_sequence_readout",
        )
        stabilized_objective = args.planning_objective in (
            "staged_contact_verified_torque_stabilized",
            "staged_contact_verified_torque_fullfan",
            "staged_contact_verified_torque_hierarchical_fullfan",
            "staged_contact_verified_hybrid_fullfan",
        )
        closure_lock = False
        closure_lock_entries = 0
        closure_lock_breaks = 0
        torque_no_feasible_steps = 0
        torque_recovery_count = 0
        sequence_fallback_count = 0
        path_cem_count = 0
        online_torque_bias = {
            mode: np.zeros(3, dtype=np.float64)
            for mode in ("free", "approach", "contact")
        }
        online_calibration_updates = 0
        direct_online_calibration_updates = 0
        direct_online_local_fit_updates = 0
        online_local_history: list[dict[str, Any]] = []
        no_progress_steps = 0
        simulator_termination_events = 0
        simulator_false_positive_terminations = 0
        terminal_return_attempts = 0
        strict_success = strict_task_success(state, goal)
        for step in range(0, args.max_steps, args.action_block):
            # The direct endpoint adapter is trained on a local contact chart.
            # The legacy protocol allowed it to run throughout the broader
            # <=65-pixel near-contact tube, which conflates approach/contact
            # mode transitions with action-effect ranking.  The opt-in
            # mode-aware protocol makes the chart boundary explicit while
            # preserving the old arm as a matched negative control.
            direct_contact_limit = (
                45.0 if args.direct_require_contact_mode else 65.0
            )
            if (
                args.planning_objective == "staged_contact_verified_hybrid_fullfan"
                and model_name == "block_only_torque_readout"
                and no_progress_steps >= 1
                and torque_recovery_count < args.max_contact_recoveries
                and not closure_lock
                and (
                    float(np.linalg.norm(state[2:4] - goal[2:4])) >= 20.0
                )
            ):
                # The local learned chart has failed online.  Reacquire from
                # the same observed state by probing a fixed set of alternate
                # contact normals; this event is kept separate from learned
                # effects in the trace and report.
                before_recovery = state.copy()
                before_xy = float(np.linalg.norm(before_recovery[2:4] - goal[2:4]))
                recovery_probe_args = argparse.Namespace(**vars(route_probe_args))
                recovery_probe_args.selection_mode = (
                    "contact_reacquisition_verified"
                    if before_xy < 45.0
                    else "contact_reacquisition"
                )
                recovered, recovery_info = probe_contact_routes(
                    env, state, goal, recovery_probe_args
                )
                before_contact = float(np.linalg.norm(before_recovery[:2] - before_recovery[2:4]))
                before_angle = angle_error(before_recovery[4], goal[4])
                before_pose_cost = before_xy + 45.0 * angle_error(before_recovery[4], goal[4])
                recovered = np.asarray(recovered, dtype=np.float64).copy()
                recovered_xy = float(np.linalg.norm(recovered[2:4] - goal[2:4]))
                recovered_contact = float(np.linalg.norm(recovered[:2] - recovered[2:4]))
                recovered_angle = angle_error(recovered[4], goal[4])
                recovered_pose_cost = recovered_xy + 45.0 * angle_error(recovered[4], goal[4])
                if recovery_probe_args.selection_mode in (
                    "contact_reacquisition",
                    "contact_reacquisition_verified",
                ) and before_xy < 45.0:
                    verified_probe = recovery_info.get("verified_probe")
                    global_probe_safe = bool(
                        isinstance(verified_probe, dict)
                        and verified_probe.get("best_global_safe", False)
                    )
                    useful_probe = bool(
                        isinstance(verified_probe, dict)
                        and (
                            verified_probe.get("best_useful", False)
                            or verified_probe.get("best_productive", False)
                        )
                    )
                    accepted_recovery = bool(
                        (
                            recovered_xy <= before_xy + 10.0
                            and recovered_contact < before_contact - 5.0
                            and recovered_angle <= before_angle + 0.25
                        )
                        or (global_probe_safe and useful_probe)
                    )
                else:
                    accepted_recovery = bool(recovered_pose_cost < before_pose_cost - 1e-6)
                probe_commit = {
                    "attempted": False,
                    "accepted": False,
                    "state_after": None,
                    "block_xy_after": None,
                    "angle_after": None,
                }
                if accepted_recovery:
                    state = recovered
                    verified_probe = recovery_info.get("verified_probe")
                    probe_action = (
                        None
                        if not isinstance(verified_probe, dict)
                        else verified_probe.get("best_action")
                    )
                    if (
                        recovery_probe_args.selection_mode == "contact_reacquisition_verified"
                        and probe_action is not None
                    ):
                        probe_commit["attempted"] = True
                        env.reset(
                            seed=0,
                            options={
                                "state": state.copy(),
                                "goal_state": goal.copy(),
                            },
                        )
                        probe_observation, _, probe_terminated, probe_truncated, _ = env.step(
                            np.asarray(probe_action, dtype=np.float32)
                        )
                        probe_state = np.asarray(
                            probe_observation["state"], dtype=np.float64
                        ).copy()
                        probe_xy = float(
                            np.linalg.norm(probe_state[2:4] - goal[2:4])
                        )
                        probe_angle = angle_error(probe_state[4], goal[4])
                        probe_pose = probe_xy + 45.0 * probe_angle
                        global_probe_safe = bool(
                            isinstance(verified_probe, dict)
                            and verified_probe.get("best_global_safe", False)
                        )
                        commit_safe = bool(
                            probe_xy <= (
                                before_xy
                                + float(
                                    getattr(
                                        recovery_probe_args,
                                        "reacquisition_xy_margin",
                                        10.0,
                                    )
                                )
                                if global_probe_safe
                                else recovered_xy + 10.0
                            )
                            and probe_angle <= (
                                before_angle + 0.25
                                if global_probe_safe
                                else recovered_angle + 0.25
                            )
                            and (
                                global_probe_safe
                                or probe_pose <= recovered_pose_cost + 1e-6
                            )
                        )
                        if commit_safe:
                            state = probe_state
                            probe_commit["accepted"] = True
                        probe_commit.update({
                            "state_after": probe_state,
                            "block_xy_after": probe_xy,
                            "angle_after": probe_angle,
                            "pose_cost_after": probe_pose,
                            "terminated": bool(probe_terminated or probe_truncated),
                        })
                route_records.append({
                    "phase": "verified_contact_route_recovery",
                    "recovery_index": int(torque_recovery_count),
                    "state_before": before_recovery,
                    "state_after": state.copy(),
                    "candidate_state_after": recovered,
                    "accepted": accepted_recovery,
                    "block_xy_before": before_xy,
                    "block_xy_after": float(np.linalg.norm(state[2:4] - goal[2:4])),
                    "candidate_block_xy_after": recovered_xy,
                    "before_pose_cost": before_pose_cost,
                    "candidate_pose_cost": recovered_pose_cost,
                    "angle_before": angle_error(before_recovery[4], goal[4]),
                    "angle_after": angle_error(state[4], goal[4]),
                    "probe_commit": probe_commit,
                    "route_info": recovery_info,
                })
                torque_recovery_count += 1
                torque_no_feasible_steps = 0
                no_progress_steps = 0
                previous = np.zeros(2, dtype=np.float32)
                block_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
                block_angle = angle_error(state[4], goal[4])
                contact_distance = float(np.linalg.norm(state[:2] - state[2:4]))
            if args.planning_objective in (
                "staged_contact_route", "staged_contact_route_calibrated",
            "staged_contact_local_effect", "staged_contact_torque_readout",
            "staged_contact_direct_torque_readout",
            "staged_contact_direct_sequence_readout",
            "staged_contact_verified_torque", "staged_contact_verified_torque_hold",
            "staged_contact_verified_torque_stabilized",
            "staged_contact_hierarchical",
            "staged_contact_verified_torque_fullfan",
            "staged_contact_verified_torque_hierarchical_fullfan",
            "staged_contact_verified_hybrid_fullfan",
            ):
                block_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
                block_angle = angle_error(state[4], goal[4])
                contact_distance = float(np.linalg.norm(state[:2] - state[2:4]))
                if hold_objective:
                    prior_lock = closure_lock
                    # Translation lock is a terminal stabilizer, not a
                    # substitute for orientation closure.  Entering it as
                    # soon as XY is small freezes cases whose angle is still
                    # far from the target and turns every useful torque
                    # proposal into a hard-tube violation.
                    if angle_error(state[4], goal[4]) < np.pi / 9.0:
                        closure_lock = update_translation_closure_lock(
                            closure_lock,
                            state,
                            goal,
                            enter_xy=20.0,
                            exit_xy=args.closure_hold_exit_xy,
                        )
                    else:
                        closure_lock = False
                    if closure_lock and not prior_lock:
                        closure_lock_entries += 1
                    if prior_lock and not closure_lock:
                        closure_lock_breaks += 1
                if (
                    block_xy >= 20.0 or block_angle >= np.pi / 9.0
                ) and contact_distance > direct_contact_limit and not closure_lock:
                    push_direction = unit(goal[2:4] - state[2:4])
                    state, reacquisition = route_to_contact(
                        env, state, goal, push_direction, route_args
                    )
                    route_records.append(reacquisition)
                    previous = np.zeros(2, dtype=np.float32)
            if (
                args.planning_objective == "staged_contact_verified_torque_hierarchical_fullfan"
                and model_name == "block_only_torque_readout"
                and torque_no_feasible_steps >= 2
                and torque_recovery_count < args.max_contact_recoveries
                and block_xy < 45.0
                and block_angle >= np.pi / 9.0
                and not closure_lock
            ):
                # No feasible one-step torque action means that the current
                # contact mode cannot realize the requested orientation
                # change.  Holding the state repeats the same failure.  Move
                # to a bounded alternate contact normal, then let the fan
                # re-evaluate from the observed state.  This is an explicit
                # recovery event, not a hidden success path.
                desired = unit(goal[2:4] - state[2:4])
                offsets = (np.pi / 2.0, -np.pi / 2.0, np.pi, 0.0)
                offset = offsets[torque_recovery_count % len(offsets)]
                c, s = np.cos(offset), np.sin(offset)
                recovery_direction = np.asarray([
                    c * desired[0] - s * desired[1],
                    s * desired[0] + c * desired[1],
                ])
                before_recovery = state.copy()
                state, recovery_info = route_to_contact(
                    env, state, goal, recovery_direction, route_args
                )
                route_records.append({
                    "phase": "torque_contact_recovery",
                    "recovery_index": int(torque_recovery_count),
                    "offset": float(offset),
                    "state_before": before_recovery,
                    "state_after": state.copy(),
                    "block_xy_before": float(np.linalg.norm(before_recovery[2:4] - goal[2:4])),
                    "block_xy_after": float(np.linalg.norm(state[2:4] - goal[2:4])),
                    "angle_before": angle_error(before_recovery[4], goal[4]),
                    "angle_after": angle_error(state[4], goal[4]),
                    "route_info": recovery_info,
                })
                torque_recovery_count += 1
                torque_no_feasible_steps = 0
                previous = np.zeros(2, dtype=np.float32)
            if (
                args.planning_objective == "staged_contact_verified_hybrid_fullfan"
                and model_name == "block_only_torque_readout"
                and torque_no_feasible_steps >= 2
                and torque_recovery_count < args.max_contact_recoveries
                and block_xy < 45.0
                and block_angle >= np.pi / 9.0
                and not closure_lock
            ):
                desired = unit(goal[2:4] - state[2:4])
                offsets = (np.pi / 2.0, -np.pi / 2.0, np.pi, 0.0)
                offset = offsets[torque_recovery_count % len(offsets)]
                c, s = np.cos(offset), np.sin(offset)
                recovery_direction = np.asarray([
                    c * desired[0] - s * desired[1],
                    s * desired[0] + c * desired[1],
                ])
                before_recovery = state.copy()
                state, recovery_info = route_to_contact(
                    env, state, goal, recovery_direction, route_args
                )
                route_records.append({
                    "phase": "torque_contact_recovery",
                    "recovery_index": int(torque_recovery_count),
                    "offset": float(offset),
                    "state_before": before_recovery,
                    "state_after": state.copy(),
                    "block_xy_before": float(np.linalg.norm(before_recovery[2:4] - goal[2:4])),
                    "block_xy_after": float(np.linalg.norm(state[2:4] - goal[2:4])),
                    "angle_before": angle_error(before_recovery[4], goal[4]),
                    "angle_after": angle_error(state[4], goal[4]),
                    "route_info": recovery_info,
                })
                torque_recovery_count += 1
                torque_no_feasible_steps = 0
                previous = np.zeros(2, dtype=np.float32)
            bank_rng = np.random.default_rng(args.seed + 1_000_000 + episode * 10_000 + step)
            candidates = candidate_bank(
                bank_rng, state, goal, args.candidates, args.horizon, args.action_magnitude
            )
            feedback_active = (
                args.planning_objective == "staged_contact_feedback"
                and feedback_steps_remaining > 0
            )
            staged = None
            if args.planning_objective in (
                "staged_contact", "staged_contact_orient", "staged_contact_recovery",
                "staged_contact_feedback", "staged_contact_calibrated",
                "staged_contact_route", "staged_contact_route_calibrated",
                "staged_contact_local_effect", "staged_contact_torque_readout",
                "staged_contact_direct_torque_readout",
                "staged_contact_direct_sequence_readout",
                "staged_contact_verified_torque", "staged_contact_verified_torque_hold",
                "staged_contact_verified_torque_stabilized",
                "staged_contact_hierarchical",
                "staged_contact_verified_torque_fullfan",
                "staged_contact_verified_torque_hierarchical_fullfan",
                "staged_contact_verified_hybrid_fullfan",
            ) and not feedback_active:
                staged_candidate = staged_contact_action(state, goal)
                # A translation lock blocks re-acquisition, but a fully closed
                # block is allowed to enter the safe agent-return phase.
                if (
                    not closure_lock
                    or staged_candidate is not None and staged_candidate[1] == "return"
                ):
                    staged = staged_candidate
            terminal_return_handled = False
            if hold_objective and staged is not None and staged[1] == "return":
                terminal_return_attempts += 1
                before_return = state.copy()
                # Every hold objective uses the same explicit terminal interface.
                # A geometric return alone can move the agent successfully while
                # breaking the block-pose gate, so it is not a valid substitute.
                state, return_info = plan_closure_preserving_return(
                    env,
                    state,
                    goal,
                    np.random.default_rng(
                        args.seed + 2_000_000 + episode * 10_000 + step
                    ),
                    args,
                )
                return_terminated = bool(return_info.get("terminated", False))
                # Gym termination is only an optimization hint here. The
                # paper-level endpoint is the explicit joint criterion,
                # including orientation; otherwise the environment can report
                # a false positive while the T is still mis-oriented.
                strict_success = strict_task_success(state, goal)
                terminated = bool(strict_success)
                trace.append({
                    "step": step,
                    "phase": "terminal_return_preserving",
                    "state_before": before_return,
                    "state_after": state.copy(),
                    "block_xy_before": float(np.linalg.norm(before_return[2:4] - goal[2:4])),
                    "block_xy_after": float(np.linalg.norm(state[2:4] - goal[2:4])),
                    "angle_before": angle_error(before_return[4], goal[4]),
                    "angle_after": angle_error(state[4], goal[4]),
                    "terminal_return_terminated": bool(return_terminated),
                    "terminal_return_task_success": bool(strict_success),
                    "terminal_return_attempt": int(terminal_return_attempts),
                    "terminal_return_retry_available": bool(
                        not strict_success
                        and terminal_return_attempts <= int(args.terminal_return_retries)
                    ),
                    "terminal_return_info": return_info,
                    "planning_objective": args.planning_objective,
                })
                # Only a strict joint success closes the episode. A failed
                # search is recoverable for a bounded number of attempts.
                terminal_return_handled = bool(
                    strict_success
                    or terminal_return_attempts > int(args.terminal_return_retries)
                )
                if terminal_return_handled:
                    previous = np.zeros(2, dtype=np.float32)
                    if strict_success:
                        break
                    # Retry exhaustion is recorded as a terminal-interface
                    # failure, but it is still not a simulator success.
                    break
                if hold_objective and staged is not None and staged[1] == "return":
                    previous = np.zeros(2, dtype=np.float32)
                    continue
            if feedback_active:
                route_index = feedback_sweep_index // max(1, args.feedback_steps)
                action, phase, _ = feedback_contact_action(state, goal, route_index)
                feedback_sweep_index += 1
                feedback_steps_remaining -= 1
                info = {
                    "selected": None,
                    "predicted_cost": None,
                    "predicted_block_xy": None,
                    "predicted_contact_distance": None,
                    "predicted_block_motion": None,
                    "planning_objective": args.planning_objective,
                    "phase": phase,
                    "feedback_active": True,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                }
            elif staged is not None:
                action, phase = staged
                info = {
                    "selected": None,
                    "predicted_cost": None,
                    "predicted_block_xy": None,
                    "predicted_contact_distance": None,
                    "predicted_block_motion": None,
                    "planning_objective": args.planning_objective,
                    "phase": phase,
                    "feedback_active": False,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                }
            elif (
                args.planning_objective == "staged_contact_hierarchical"
                and model_name == "block_only_torque_readout"
                and float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                and float(np.linalg.norm(state[:2] - state[2:4])) <= direct_contact_limit
            ):
                # The torque readout is local. Keep long-range translation on
                # the independently fitted block-pose adapter, then switch to
                # verified local effects in the near-goal tube.
                action, info = choose_action(
                    model["base"], "block_only", state, goal, previous,
                    candidates, planning_objective="staged_translation",
                )
                info.update({
                    "planning_objective": args.planning_objective,
                    "phase": "learned_translation",
                    "feedback_active": False,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                    "hierarchical_stage": "translation",
                })
            elif (
                args.planning_objective == "staged_contact_verified_torque_hierarchical_fullfan"
                and model_name == "block_only_torque_readout"
                and float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                and float(np.linalg.norm(state[:2] - state[2:4])) <= direct_contact_limit
            ):
                # This is the corrected hierarchical full-fan arm.  The
                # learned torque readout is only a local orientation proposal;
                # long-range block translation must use the separately fitted
                # block-pose adapter.  The previous full-fan protocol entered
                # the torque branch here and silently held the block whenever
                # its local model found no feasible effect.
                action, info = choose_action(
                    model["base"], "block_only", state, goal, previous,
                    candidates, planning_objective="staged_translation",
                )
                info.update({
                    "planning_objective": args.planning_objective,
                    "phase": "learned_translation",
                    "feedback_active": False,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                    "hierarchical_stage": "translation",
                    "hierarchical_fullfan_corrected": True,
                })
            elif (
                args.planning_objective == "staged_contact_local_effect"
                and float(np.linalg.norm(state[2:4] - goal[2:4])) < 45.0
                and angle_error(state[4], goal[4]) >= np.pi / 9.0
            ):
                action, info = choose_local_effect_action(
                    env,
                    state,
                    goal,
                    directions=args.local_effect_directions,
                    scale=args.local_effect_scale,
                    xy_margin=(
                        args.local_effect_xy_margin
                        if args.local_effect_xy_margin >= 0.0 else None
                    ),
                )
                info.update({
                    "planning_objective": args.planning_objective,
                    "feedback_active": False,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                })
            elif (
                args.planning_objective == "staged_contact_direct_sequence_readout"
                and model_name == "block_only_torque_readout"
                and (
                    float(np.linalg.norm(state[2:4] - goal[2:4])) >= 20.0
                    or angle_error(state[4], goal[4]) >= np.pi / 9.0
                )
                and float(np.linalg.norm(state[:2] - state[2:4])) <= 65.0
            ):
                action, info = choose_direct_torque_sequence_action(
                    model["direct_sequence_endpoint"],
                    state,
                    goal,
                    directions=args.direct_sequence_directions,
                    magnitudes=tuple(args.direct_sequence_magnitudes),
                    translation_first=(
                        float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                    ),
                    previous_action=(
                        previous if args.direct_use_previous_action else None
                    ),
                    avoid_action=(previous if no_progress_steps >= 1 else None),
                    feature_interface=args.torque_feature_interface,
                )
                info.update({
                    "planning_objective": args.planning_objective,
                    "phase": "pure_learned_direct_sequence",
                    "feedback_active": False,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                    "simulator_verification_used": False,
                    "direct_endpoint_horizon": int(args.action_block),
                })
            elif (
                args.planning_objective == "staged_contact_direct_torque_readout"
                and model_name == "block_only_torque_readout"
                and (
                    float(np.linalg.norm(state[2:4] - goal[2:4])) >= 20.0
                    or angle_error(state[4], goal[4]) >= np.pi / 9.0
                )
                and float(np.linalg.norm(state[:2] - state[2:4])) <= 65.0
            ):
                # This is the pure learned endpoint arm.  The model was fit
                # for exactly ``action_block`` simulator steps and scores the
                # complete constant-action block without probing the simulator
                # at selection time.  Keep it before the verified branches so
                # the direct result cannot silently inherit a verifier.
                if args.direct_goal_conditioned_policy:
                    action, info = choose_goal_conditioned_phase_policy_action(
                        model["goal_phase_policy"],
                        state,
                        goal,
                        directions=args.local_effect_directions,
                        translation_first=(
                            float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                        ),
                        magnitudes=tuple(args.direct_action_magnitudes),
                        previous_action=(previous if args.direct_use_previous_action else None),
                        feature_interface=args.torque_feature_interface,
                    )
                elif args.direct_goal_conditioned_score:
                    action, info = choose_goal_conditioned_phase_score_action(
                        model["goal_phase_score"],
                        state,
                        goal,
                        directions=args.local_effect_directions,
                        translation_first=(
                            float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                        ),
                        magnitudes=tuple(args.direct_action_magnitudes),
                        previous_action=(previous if args.direct_use_previous_action else None),
                        avoid_action=(previous if no_progress_steps >= 1 else None),
                        feature_interface=args.torque_feature_interface,
                    )
                elif args.direct_goal_conditioned_feasibility:
                    action, info = choose_goal_conditioned_feasibility_action(
                        model["goal_feasibility"],
                        state,
                        goal,
                        directions=args.local_effect_directions,
                        translation_first=(
                            float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                        ),
                        magnitudes=tuple(args.direct_action_magnitudes),
                        previous_action=(previous if args.direct_use_previous_action else None),
                        avoid_action=(previous if no_progress_steps >= 1 else None),
                        feature_interface=args.torque_feature_interface,
                    )
                elif args.direct_goal_conditioned_ranker:
                    action, info = choose_goal_conditioned_action(
                        model["goal_ranker"],
                        state,
                        goal,
                        directions=args.local_effect_directions,
                        translation_first=(
                            float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                        ),
                        magnitudes=tuple(args.direct_action_magnitudes),
                        previous_action=(previous if args.direct_use_previous_action else None),
                        avoid_action=(previous if no_progress_steps >= 1 else None),
                        feature_interface=args.torque_feature_interface,
                    )
                else:
                    action, info = choose_direct_torque_effect_action(
                        model["direct_endpoint"],
                        state,
                        goal,
                        directions=args.local_effect_directions,
                        translation_first=(
                            float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                        ),
                        magnitudes=tuple(args.direct_action_magnitudes),
                        avoid_action=(previous if no_progress_steps >= 1 else None),
                        previous_action=(previous if args.direct_use_previous_action else None),
                        uncertainty_penalty=args.direct_uncertainty_penalty,
                        contact_mode_penalty=args.direct_contact_mode_penalty,
                        horizon_consistency_model=(
                            select_torque_effect_model(model, state)
                            if args.direct_horizon_consistency_penalty > 0.0 else None
                        ),
                        horizon_consistency_penalty=args.direct_horizon_consistency_penalty,
                        feature_interface=args.torque_feature_interface,
                        online_effect_bias=(
                            online_torque_bias[mode_for_state(state)]
                            if args.direct_online_calibration else None
                        ),
                        online_local_history=(
                            online_local_history if args.direct_online_local_fit else None
                        ),
                        online_local_strategy=args.direct_online_local_strategy,
                        online_local_weight=args.direct_online_local_weight,
                        online_local_radius=args.direct_online_local_radius,
                        online_local_neighbors=args.direct_online_local_neighbors,
                    )
                info.update({
                    "planning_objective": args.planning_objective,
                    "phase": "pure_learned_direct_endpoint",
                    "feedback_active": False,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                    "simulator_verification_used": False,
                    "direct_endpoint_horizon": int(args.action_block),
                    "direct_goal_conditioned_ranker": bool(args.direct_goal_conditioned_ranker),
                    "direct_goal_conditioned_feasibility": bool(args.direct_goal_conditioned_feasibility),
                    "direct_goal_conditioned_score": bool(args.direct_goal_conditioned_score),
                    "direct_goal_conditioned_policy": bool(args.direct_goal_conditioned_policy),
                    "direct_ranker_goals_per_state": int(args.direct_ranker_goals_per_state),
                    "direct_online_calibration": bool(args.direct_online_calibration),
                    "direct_online_local_fit": bool(args.direct_online_local_fit),
                    "direct_online_local_strategy": args.direct_online_local_strategy,
                    "direct_contact_mode_penalty": args.direct_contact_mode_penalty,
                })
            elif (
                args.planning_objective == "staged_contact_torque_readout"
                and model_name == "block_only_torque_readout"
                and (
                    float(np.linalg.norm(state[2:4] - goal[2:4])) >= 20.0
                    or angle_error(state[4], goal[4]) >= np.pi / 9.0
                )
                and float(np.linalg.norm(state[:2] - state[2:4])) <= 65.0
            ):
                torque_model = select_torque_effect_model(model, state)
                action, info = choose_torque_effect_action(
                    torque_model,
                    state,
                    goal,
                    previous,
                    directions=args.local_effect_directions,
                    xy_margin=(
                        args.local_effect_xy_margin
                        if args.local_effect_xy_margin >= 0.0 else 4.0
                    ),
                    translation_first=(
                        float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                    ),
                    feature_mode=args.torque_feature_interface,
                )
                info.update({
                    "planning_objective": args.planning_objective,
                    "phase": "learned_torque_effect",
                    "feedback_active": False,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                })
            elif (
                args.planning_objective == "staged_contact_verified_hybrid_fullfan"
                and model_name == "block_only_torque_readout"
                and (
                    float(np.linalg.norm(state[2:4] - goal[2:4])) >= 20.0
                    or angle_error(state[4], goal[4]) >= np.pi / 9.0
                )
                and (
                    float(np.linalg.norm(state[:2] - state[2:4])) <= 65.0
                    or closure_lock
                )
            ):
                # Learned translation is a proposal, not the final decision.
                # Verify it together with the complete local torque fan from
                # the same observed state.  This directly tests whether the
                # hierarchical failure came from excluding useful contact
                # actions during long-range translation.  The learned
                # translation proposal is not sufficient by itself: when its
                # predicted contact mode is wrong, the old hybrid arm had a
                # zero-action fallback and could remain motionless forever.
                # Add a bounded geometric contact fan as a *recovery proposal*
                # and subject it to exactly the same one-step simulator
                # verification.  This keeps the intervention explicit and
                # auditable; it is not silently counted as a learned effect.
                learned_action, learned_info = choose_action(
                    model["base"], "block_only", state, goal, previous,
                    candidates, planning_objective="staged_translation",
                )
                # Keep the mode-conditional block adapter as an independent
                # learned proposal.  The contact-frame torque readout is
                # strongest on local angular effects, whereas the specialist
                # sees the discontinuous free/approach/contact strata.  It is
                # still only a proposal: the same simulator verification
                # ranks it against the geometric and torque fans.
                specialist_models = model.get("specialist")
                specialist_action = None
                specialist_info = None
                if isinstance(specialist_models, dict):
                    specialist_route = mode_for_state(state)
                    specialist_predictor = specialist_models.get(
                        specialist_route, specialist_models["fallback"]
                    )
                    specialist_action, specialist_info = choose_action(
                        specialist_predictor,
                        "block_only_specialist",
                        state,
                        goal,
                        previous,
                        candidates,
                        planning_objective="staged_contact_orient",
                    )
                geometric_actions = local_effect_action_bank(
                    state,
                    goal,
                    directions=max(8, args.local_effect_directions),
                    scale=max(float(args.local_effect_scale), 0.55),
                )
                fine_actions = torque_action_bank(
                    directions=16,
                    magnitudes=(0.12, 0.20, 0.28),
                )
                micro_actions = torque_action_bank(
                    directions=24,
                    magnitudes=(0.04, 0.08, 0.12),
                )
                extra_actions = np.concatenate(
                    [
                        np.asarray(learned_action, dtype=np.float32)[None],
                        np.asarray(specialist_action, dtype=np.float32)[None]
                        if specialist_action is not None else np.zeros((0, 2), dtype=np.float32),
                        geometric_actions,
                        fine_actions,
                        micro_actions,
                    ],
                    axis=0,
                )
                action, info = choose_verified_torque_effect_action(
                    env,
                    select_torque_effect_model(model, state),
                    state,
                    goal,
                    previous,
                    directions=args.local_effect_directions,
                    xy_margin=(
                        args.local_effect_xy_margin
                        if args.local_effect_xy_margin >= 0.0 else 4.0
                    ),
                    topk=args.torque_verification_topk,
                    translation_first=(
                        float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                        or (
                            float(np.linalg.norm(state[2:4] - goal[2:4])) >= 20.0
                            and angle_error(state[4], goal[4]) < np.pi / 9.0
                        )
                    ),
                    feature_mode=args.torque_feature_interface,
                    hard_xy_limit=(
                        20.0 + max(float(args.closure_hold_xy_margin), 0.0)
                        if closure_lock and angle_error(state[4], goal[4]) < np.pi / 9.0
                        else None
                    ),
                    verify_all=True,
                    # In the terminal tube, never trade a large translation
                    # regression for a small predicted angle improvement.  If
                    # no candidate closes both constraints, choose the
                    # actually verified successor with the smallest XY error.
                    zero_on_no_feasible=bool(
                        closure_lock and angle_error(state[4], goal[4]) < np.pi / 9.0
                    ),
                    extra_actions=extra_actions,
                    sequence_fallback_actions=None,
                    # The CEM fallback is available for isolated diagnostics,
                    # but is disabled in the matched arm because the smoke
                    # audit found no feasible successor and it adds simulator
                    # probes without improving closure.
                    cem_fallback=False,
                    path_cem_fallback=(path_cem_count < 3),
                    path_cem_seed=args.seed + 5_000_000 + episode * 10_000 + step,
                    orientation_path_cem_fallback=(path_cem_count < 5),
                    orientation_path_cem_seed=args.seed + 6_000_000 + episode * 10_000 + step,
                    orientation_path_xy_limit=(
                        min(
                            35.0,
                            max(
                                24.0,
                                float(np.linalg.norm(state[2:4] - goal[2:4])) + 6.0,
                            ),
                        )
                        if not closure_lock else 20.0 + max(float(args.closure_hold_xy_margin), 0.0)
                    ),
                    orientation_path_horizon=args.orientation_path_horizon,
                    orientation_path_population=args.orientation_path_population,
                    orientation_path_iterations=args.orientation_path_iterations,
                    verification_steps=args.action_block,
                    online_effect_bias=online_torque_bias[mode_for_state(state)],
                    online_effect_mode=mode_for_state(state),
                )
                info.update({
                    "planning_objective": args.planning_objective,
                    "phase": "verified_hybrid_effect",
                    "feedback_active": False,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                    "learned_translation_proposal": learned_info,
                    "specialist_proposal": specialist_info,
                    "geometric_contact_recovery_proposals": int(len(geometric_actions)),
                    "fine_torque_proposals": int(len(fine_actions)),
                    "micro_torque_proposals": int(len(micro_actions)),
                    "hybrid_proposal_verified": True,
                    "closure_translation_lock": bool(closure_lock),
                    "online_effect_mode": mode_for_state(state),
                })
            elif (
                args.planning_objective in (
                    "staged_contact_verified_torque",
                    "staged_contact_verified_torque_hold",
                    "staged_contact_verified_torque_stabilized",
                    "staged_contact_hierarchical",
                    "staged_contact_verified_torque_fullfan",
                    "staged_contact_verified_torque_hierarchical_fullfan",
                )
                and model_name == "block_only_torque_readout"
                and (
                    float(np.linalg.norm(state[2:4] - goal[2:4])) >= 20.0
                    or angle_error(state[4], goal[4]) >= np.pi / 9.0
                )
                and (
                    float(np.linalg.norm(state[:2] - state[2:4])) <= 65.0
                    or closure_lock
                )
            ):
                action, info = choose_verified_torque_effect_action(
                    env,
                    select_torque_effect_model(model, state),
                    state,
                    goal,
                    previous,
                    directions=args.local_effect_directions,
                    xy_margin=(
                        args.local_effect_xy_margin
                        if args.local_effect_xy_margin >= 0.0 else 4.0
                    ),
                    topk=args.torque_verification_topk,
                    translation_first=(
                        float(np.linalg.norm(state[2:4] - goal[2:4])) >= 45.0
                    ),
                    feature_mode=args.torque_feature_interface,
                    hard_xy_limit=(
                        20.0 + max(float(args.closure_hold_xy_margin), 0.0)
                        if closure_lock else None
                    ),
                    verify_all=bool(
                        args.planning_objective == "staged_contact_verified_torque_fullfan"
                        or args.planning_objective == "staged_contact_verified_torque_hierarchical_fullfan"
                        or (stabilized_objective and closure_lock)
                    ),
                    zero_on_no_feasible=bool(closure_lock),
                    verification_steps=args.action_block,
                    online_effect_bias=online_torque_bias[mode_for_state(state)],
                    online_effect_mode=mode_for_state(state),
                )
                info.update({
                    "planning_objective": args.planning_objective,
                    "phase": "verified_torque_effect",
                    "feedback_active": False,
                    "model_mismatch_steps": model_mismatch_steps,
                    "candidate_cost_finite": True,
                    "closure_translation_lock": bool(closure_lock),
                })
            else:
                selected_model = model
                model_route = model_name
                if model_name == "block_only_torque_readout":
                    selected_model = model["base"]
                if model_name == "block_only_specialist":
                    current_mode = mode_for_state(state)
                    if (
                        current_mode == "contact"
                        and angle_error(state[4], goal[4]) > np.pi / 9.0
                    ):
                        selected_model = model["torque"]
                        model_route = "torque"
                    else:
                        selected_model = model.get(current_mode, model["fallback"])
                        model_route = current_mode
                action, info = choose_action(
                    selected_model, model_name, state, goal, previous, candidates,
                    planning_objective=(
                        "staged_contact_orient"
                        if args.planning_objective in (
                            "staged_contact_orient", "staged_contact_recovery",
                            "staged_contact_feedback", "staged_contact_calibrated",
                            "staged_contact_route", "staged_contact_route_calibrated",
                            "staged_contact_local_effect", "staged_contact_torque_readout",
                            "staged_contact_direct_torque_readout",
                            "staged_contact_direct_sequence_readout",
                            "staged_contact_verified_torque", "staged_contact_verified_torque_hold",
                            "staged_contact_verified_torque_stabilized",
                            "staged_contact_hierarchical",
                            "staged_contact_verified_torque_fullfan",
                            "staged_contact_verified_torque_hierarchical_fullfan",
                            "staged_contact_verified_hybrid_fullfan",
                        )
                        else "contact_gated"
                        if args.planning_objective == "staged_contact"
                        else args.planning_objective
                    ),
                    contact_gate=(
                        contact_gate
                        if args.planning_objective in (
                            "staged_contact_calibrated", "staged_contact_route_calibrated"
                        )
                        else None
                    ),
                    contact_threshold=args.contact_gate_threshold,
                )
                if args.planning_objective in (
                    "staged_contact", "staged_contact_orient", "staged_contact_recovery",
                    "staged_contact_feedback", "staged_contact_calibrated",
                    "staged_contact_route", "staged_contact_route_calibrated",
                            "staged_contact_local_effect", "staged_contact_torque_readout",
                            "staged_contact_direct_torque_readout",
                            "staged_contact_direct_sequence_readout",
                            "staged_contact_verified_torque", "staged_contact_verified_torque_hold",
                            "staged_contact_verified_torque_stabilized",
                            "staged_contact_hierarchical",
                            "staged_contact_verified_torque_fullfan",
                            "staged_contact_verified_torque_hierarchical_fullfan",
                            "staged_contact_verified_hybrid_fullfan",
                ):
                    info["phase"] = "learned_push"
                info["feedback_active"] = False
                info["model_mismatch_steps"] = model_mismatch_steps
                info["model_route"] = model_route
            before = state.copy()
            execution_sequence = info.get("execution_action_sequence")
            if execution_sequence is not None:
                execution_sequence = np.asarray(execution_sequence, dtype=np.float32)
                if execution_sequence.shape != (args.action_block, 2):
                    raise ValueError(
                        "learned execution_action_sequence must match action_block"
                    )
                applied_actions = execution_sequence
            else:
                applied_actions = np.repeat(
                    np.asarray(action, dtype=np.float32)[None], args.action_block, axis=0
                )
            for applied_action in applied_actions:
                observation, reward, terminated, truncated, _ = env.step(applied_action)
                state = np.asarray(observation["state"], dtype=np.float64).copy()
                total_return += float(reward)
                if terminated or truncated:
                    break
            simulator_terminated = bool(terminated or truncated)
            strict_success = strict_task_success(state, goal)
            if simulator_terminated:
                simulator_termination_events += 1
                if not strict_success:
                    # The simulator has an internal translational stopping
                    # rule that is weaker than this audit's joint pose rule.
                    # Re-open the state-only diagnostic episode so orientation
                    # repair can continue.  This is not a success path: the
                    # strict criterion below remains the sole endpoint gate.
                    simulator_false_positive_terminations += 1
                    env.reset(
                        seed=0,
                        options={
                            "state": state.copy(),
                            "goal_state": goal.copy(),
                        },
                    )
                    terminated = False
                    truncated = False
            if strict_success:
                terminated = True
                truncated = False
            block_xy_before = float(np.linalg.norm(before[2:4] - goal[2:4]))
            block_xy_after = float(np.linalg.norm(state[2:4] - goal[2:4]))
            actual_block_motion = float(np.linalg.norm(state[2:4] - before[2:4]))
            if hold_objective:
                if angle_error(state[4], goal[4]) < np.pi / 9.0:
                    lock_after = update_translation_closure_lock(
                        closure_lock,
                        state,
                        goal,
                        enter_xy=20.0,
                        exit_xy=args.closure_hold_exit_xy,
                    )
                else:
                    lock_after = False
                info["closure_translation_lock_before"] = bool(closure_lock)
                info["closure_translation_lock_after"] = bool(lock_after)
                info["closure_hold_xy_limit"] = float(
                    20.0 + max(float(args.closure_hold_xy_margin), 0.0)
                )
                if closure_lock and not lock_after:
                    closure_lock_breaks += 1
                if lock_after and not closure_lock:
                    closure_lock_entries += 1
                closure_lock = lock_after
            if args.planning_objective in (
                "staged_contact_verified_torque_hierarchical_fullfan",
                "staged_contact_verified_hybrid_fullfan",
            ):
                if (
                    info.get("phase") in ("verified_torque_effect", "verified_hybrid_effect")
                    and int(info.get("verified_feasible_count", 0)) == 0
                ):
                    # An orientation path search is itself a bounded
                    # multi-step recovery attempt.  Do not immediately
                    # trigger the old alternate-normal route after it fails:
                    # that route can eject a near-goal block and conflates
                    # search failure with a changed contact mode.
                    if info.get("orientation_path_cem_fallback", False):
                        torque_no_feasible_steps = 0
                    else:
                        torque_no_feasible_steps += 1
                elif info.get("phase") in ("verified_torque_effect", "verified_hybrid_effect"):
                    torque_no_feasible_steps = 0
            if info.get("sequence_fallback", False):
                sequence_fallback_count += 1
            if info.get("path_cem_fallback", False):
                path_cem_count += 1
            if info.get("orientation_path_cem_fallback", False):
                path_cem_count += 1
            residual = info.get("online_effect_residual")
            if residual is not None:
                residual = np.asarray(residual, dtype=np.float64).reshape(-1)
                residual_mode = str(info.get("online_effect_mode", mode_for_state(before)))
                horizon_compatible = int(info.get("verification_steps", 1)) == 1
                if (
                    residual.shape == (3,)
                    and np.isfinite(residual).all()
                    and residual_mode in online_torque_bias
                    and horizon_compatible
                ):
                    clipped = np.clip(residual, [-5.0, -5.0, -0.5], [5.0, 5.0, 0.5])
                    online_torque_bias[residual_mode] = (
                        0.75 * online_torque_bias[residual_mode] + 0.25 * clipped
                    )
                    online_calibration_updates += 1
                    info["online_effect_bias_after"] = online_torque_bias[residual_mode].copy()
                    info["online_calibration_update"] = True
                else:
                    info["online_calibration_update"] = False
                    info["online_calibration_skip_reason"] = (
                        "verification_horizon_mismatch"
                        if not horizon_compatible
                        else "invalid_or_unrouted_residual"
                    )
            if (
                (args.direct_online_calibration or args.direct_online_local_fit)
                and info.get("phase") == "pure_learned_direct_endpoint"
                and len(applied_actions) == args.action_block
                and not terminated
            ):
                relative = np.asarray(before[2:4] - before[:2], dtype=np.float64)
                normal = unit(relative)
                tangent = np.asarray([-normal[1], normal[0]])
                actual_local = np.asarray([
                    float(np.dot(state[2:4] - before[2:4], normal)),
                    float(np.dot(state[2:4] - before[2:4], tangent)),
                    float((state[4] - before[4] + np.pi) % (2.0 * np.pi) - np.pi),
                ])
                predicted_world = np.asarray(
                    info.get("predicted_effect", np.zeros(3)), dtype=np.float64
                ).reshape(-1)
                predicted_local = np.asarray([
                    float(np.dot(predicted_world[:2], normal)),
                    float(np.dot(predicted_world[:2], tangent)),
                    float(predicted_world[2]),
                ])
                direct_residual = actual_local - predicted_local
                mode = mode_for_state(before)
                action_local = np.asarray([
                    float(np.dot(applied_actions[0], normal)),
                    float(np.dot(applied_actions[0], tangent)),
                ])
                online_local_history.append({
                    "mode": mode,
                    "action_local": action_local.copy(),
                    "effect_local": actual_local.copy(),
                    "predicted_local": predicted_local.copy(),
                    "residual_local": direct_residual.copy(),
                    "state_before": before.copy(),
                    "action_world": np.asarray(applied_actions[0], dtype=np.float64).copy(),
                    "previous_action": np.asarray(previous, dtype=np.float64).copy(),
                })
                direct_online_local_fit_updates += 1
                if args.direct_online_calibration:
                    clipped = np.clip(direct_residual, [-8.0, -8.0, -0.8], [8.0, 8.0, 0.8])
                    online_torque_bias[mode] = (
                        0.60 * online_torque_bias[mode] + 0.40 * clipped
                    )
                    direct_online_calibration_updates += 1
                info["direct_online_effect_residual"] = direct_residual.copy()
                info["direct_online_effect_bias_after"] = online_torque_bias[mode].copy()
                info["direct_online_calibration_update"] = bool(args.direct_online_calibration)
                info["direct_online_local_fit_update"] = True
            if actual_block_motion < 0.5 or block_xy_after > block_xy_before + 0.5:
                no_progress_steps += 1
            else:
                no_progress_steps = 0
            trace.append({
                "step": step,
                "action": action,
                "block_xy_before": block_xy_before,
                "block_xy_after": block_xy_after,
                "actual_block_motion": actual_block_motion,
                **info,
            })
            trace[-1]["simulator_terminated"] = bool(simulator_terminated)
            trace[-1]["strict_task_success"] = bool(strict_success)
            trace[-1]["simulator_false_positive_termination"] = bool(
                simulator_terminated and not strict_success
            )
            if args.planning_objective == "staged_contact_feedback":
                predicted_motion = info.get("predicted_block_motion")
                model_claim_missed = (
                    predicted_motion is not None
                    and float(predicted_motion) >= 2.0
                    and (actual_block_motion < 0.5 or block_xy_after > block_xy_before + 1.0)
                )
                if model_claim_missed:
                    model_mismatch_steps += 1
                elif actual_block_motion >= 0.5 and block_xy_after < block_xy_before:
                    model_mismatch_steps = 0
                if (
                    not feedback_active
                    and model_mismatch_steps >= max(1, args.feedback_patience)
                ):
                    feedback_steps_remaining = max(1, args.feedback_steps)
                    model_mismatch_steps = 0
            previous = np.asarray(applied_actions[-1], dtype=np.float32).copy()
            if terminated or truncated:
                break
        rows.append({
            "episode": episode,
            "model": model_name,
            "mode": case["mode"],
            "success": bool(strict_success),
            "strict_task_success": bool(strict_success),
            "simulator_terminated": bool(simulator_termination_events > 0),
            "simulator_termination_events": int(simulator_termination_events),
            "simulator_false_positive_terminations": int(
                simulator_false_positive_terminations
            ),
            "steps": len(trace) * args.action_block,
            "return": total_return,
            "block_xy_distance": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "angle_error": angle_error(state[4], goal[4]),
            "agent_xy_distance": float(np.linalg.norm(state[:2] - goal[:2])),
            "trace": trace,
            "route_info": route_records,
            "closure_lock_entries": int(closure_lock_entries),
            "closure_lock_breaks": int(closure_lock_breaks),
            "closure_lock_active_at_end": bool(closure_lock),
            "terminal_return_attempts": int(terminal_return_attempts),
            "torque_contact_recovery_count": int(torque_recovery_count),
            "torque_no_feasible_steps_at_end": int(torque_no_feasible_steps),
            "sequence_fallback_count": int(sequence_fallback_count),
            "path_cem_count": int(path_cem_count),
            "online_calibration_updates": int(online_calibration_updates),
            "direct_online_calibration_updates": int(direct_online_calibration_updates),
            "direct_online_local_fit_updates": int(direct_online_local_fit_updates),
            "online_torque_bias_final": online_torque_bias,
            "no_progress_steps_at_end": int(no_progress_steps),
        })
    return rows


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(len(values)),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for model_name in PLANNER_MODEL_NAMES:
        subset = [row for row in rows if row["model"] == model_name]
        result[model_name] = {
            "success_rate": bootstrap(
                np.asarray([float(row["success"]) for row in subset]), args.bootstrap, args.seed + 10
            ),
            "block_xy_distance": bootstrap(
                np.asarray([row["block_xy_distance"] for row in subset]), args.bootstrap, args.seed + 20
            ),
            "angle_error": bootstrap(
                np.asarray([row["angle_error"] for row in subset]), args.bootstrap, args.seed + 30
            ),
            "agent_xy_distance": bootstrap(
                np.asarray([row["agent_xy_distance"] for row in subset]), args.bootstrap, args.seed + 40
            ),
        }
    paired = []
    for target in PLANNER_MODEL_NAMES[1:]:
        baseline = {row["episode"]: row for row in rows if row["model"] == "state_action"}
        candidate = {row["episode"]: row for row in rows if row["model"] == target}
        paired.append({
            "target": target,
            "baseline": "state_action",
            "success_difference": bootstrap(
                np.asarray([float(candidate[i]["success"]) - float(baseline[i]["success"]) for i in baseline]),
                args.bootstrap, args.seed + len(target),
            ),
            "block_xy_difference": bootstrap(
                np.asarray([candidate[i]["block_xy_distance"] - baseline[i]["block_xy_distance"] for i in baseline]),
                args.bootstrap, args.seed + 100 + len(target),
            ),
        })
    return {"by_model": result, "paired_against_state_action": paired}


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
    if args.action_block > args.horizon:
        raise ValueError("action block cannot exceed planning horizon")
    if args.direct_dagger_rounds < 0 or args.direct_dagger_states_per_round < 1:
        raise ValueError("DAgger rounds must be nonnegative and states per round positive")
    if args.direct_dagger_rounds > 0 and not args.direct_goal_conditioned_feasibility:
        raise ValueError("DAgger augmentation currently requires phase feasibility heads")
    if args.planning_objective in (
        "staged_contact_direct_torque_readout",
        "staged_contact_direct_sequence_readout",
    ):
        if (
            args.planning_objective == "staged_contact_direct_sequence_readout"
            and args.action_block != 2
        ):
            raise ValueError(
                "direct learned sequence endpoint requires --action-block 2"
            )
        if args.direct_train_states < 8:
            raise ValueError("--direct-train-states must be at least 8")
        if args.direct_ranker_goals_per_state < 2:
            raise ValueError("--direct-ranker-goals-per-state must be at least 2")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = make_state_only_env()
    try:
        if args.training_protocol == "contact_intervention":
            train = collect_contact_intervention_data(env, args.train_transitions, args.seed)
        else:
            train = collect_trajectory_data(env, args.train_transitions, args.seed)
        models = fit_models(train, args.trees, args.seed)
        models["block_only"] = fit_block_only_model(train, args.trees, args.seed + 1000)
        models["block_only_specialist"] = fit_block_only_specialists(
            train, args.trees, args.seed + 2000
        )
        models["block_agent_hybrid"] = {
            "block": fit_block_only_model(train, args.trees, args.seed + 2500),
            "agent": fit_agent_only_model(train, args.trees, args.seed + 2600),
        }
        models["block_only_torque_readout"] = {
            "base": fit_block_only_model(train, args.trees, args.seed + 2700),
            "torque_effect": fit_contact_frame_torque_effect_model(
                train, args.trees, args.seed + 2800,
                feature_interface=args.torque_feature_interface,
            ),
            "torque_effect_specialists": fit_contact_frame_torque_effect_specialists(
                train, args.trees, args.seed + 2900,
                feature_interface=args.torque_feature_interface,
            ),
            "specialist": models["block_only_specialist"],
        }
        direct_training_count = 0
        direct_dagger_state_count = 0
        direct_training_sources: list[str] = []
        if args.planning_objective in (
            "staged_contact_direct_torque_readout",
            "staged_contact_direct_sequence_readout",
        ):
            # The planner treats agent--block distances up to 65 px as a
            # usable local contact manifold, while the old direct adapter was
            # trained only on the narrower ``mode == contact`` stratum.  The
            # near-contact option makes this deployment domain explicit while
            # keeping the old contact-only protocol available as an ablation.
            def direct_state_mask(data: dict[str, np.ndarray]) -> np.ndarray:
                if args.direct_state_pool == "near_contact_mixed":
                    states = np.asarray(data["states"], dtype=np.float64)
                    distance = np.linalg.norm(states[:, :2] - states[:, 2:4], axis=1)
                    return distance <= 65.0
                return np.asarray(data["modes"]) == "contact"

            primary_mask = direct_state_mask(train)
            contact_states = np.asarray(train["states"])[primary_mask]
            contact_previous = np.asarray(train["previous_actions"])[primary_mask]
            direct_training_sources.append(args.training_protocol)
            state_pools = [(contact_states, contact_previous)]
            if args.direct_state_pool in {"mixed", "near_contact_mixed"}:
                # Intervention data gives directional coverage but can have a
                # narrower velocity/contact-transition distribution.  A
                # trajectory stream can also be augmented with a second
                # independently seeded trajectory stream.  The direct
                # endpoint adapter alone sees this mixture; matched baseline
                # arms retain the requested training protocol unchanged.
                auxiliary = collect_trajectory_data(
                    env, args.train_transitions, args.seed + 500000
                )
                auxiliary_mask = direct_state_mask(auxiliary)
                auxiliary_contact = np.asarray(auxiliary["states"])[auxiliary_mask]
                if len(auxiliary_contact) > 0:
                    auxiliary_previous = np.asarray(auxiliary["previous_actions"])[auxiliary_mask]
                    state_pools.append((auxiliary_contact, auxiliary_previous))
                    direct_training_sources.append("trajectory_auxiliary")
            available_states = np.concatenate(
                [states for states, _ in state_pools], axis=0
            )
            if len(available_states) < args.direct_train_states:
                raise RuntimeError(
                    "insufficient contact states for direct endpoint training: "
                    f"{len(available_states)} < {args.direct_train_states}"
                )
            # Deterministic per-source subsampling preserves both domains
            # instead of letting the larger pool dominate the endpoint map.
            per_source = int(np.ceil(args.direct_train_states / len(state_pools)))
            selected_states = []
            selected_previous = []
            for pool, previous_pool in state_pools:
                indices = np.linspace(0, len(pool) - 1, per_source, dtype=int)
                selected_states.append(pool[indices])
                selected_previous.append(previous_pool[indices])
            direct_states = np.concatenate(selected_states, axis=0)[: args.direct_train_states]
            direct_previous = np.concatenate(selected_previous, axis=0)[: args.direct_train_states]
            if args.planning_objective == "staged_contact_direct_torque_readout":
                direct_actions = torque_action_bank(
                    args.local_effect_directions,
                    magnitudes=tuple(args.direct_action_magnitudes),
                )
                models["block_only_torque_readout"]["direct_endpoint"] = (
                    fit_direct_contact_endpoint_model(
                        env,
                        direct_states,
                        direct_actions,
                        horizon=args.action_block,
                        trees=args.trees,
                        seed=args.seed + 3200,
                        previous_actions=(
                            direct_previous if args.direct_use_previous_action else None
                        ),
                        feature_interface=args.torque_feature_interface,
                    )
                )
                if args.direct_goal_conditioned_policy:
                    models["block_only_torque_readout"]["goal_phase_policy"] = (
                        fit_goal_conditioned_phase_policy_heads(
                            env,
                            direct_states,
                            direct_actions,
                            horizon=args.action_block,
                            trees=args.trees,
                            seed=args.seed + 3465,
                            previous_actions=(
                                direct_previous if args.direct_use_previous_action else None
                            ),
                            feature_interface=args.torque_feature_interface,
                            goals_per_state=args.direct_ranker_goals_per_state,
                        )
                    )
                elif args.direct_goal_conditioned_score:
                    models["block_only_torque_readout"]["goal_phase_score"] = (
                        fit_goal_conditioned_phase_score_heads(
                            env,
                            direct_states,
                            direct_actions,
                            horizon=args.action_block,
                            trees=args.trees,
                            seed=args.seed + 3460,
                            previous_actions=(
                                direct_previous if args.direct_use_previous_action else None
                            ),
                            feature_interface=args.torque_feature_interface,
                            goals_per_state=args.direct_ranker_goals_per_state,
                        )
                    )
                elif args.direct_goal_conditioned_feasibility:
                    models["block_only_torque_readout"]["goal_feasibility"] = (
                        fit_goal_conditioned_phase_feasibility_heads(
                            env,
                            direct_states,
                            direct_actions,
                            horizon=args.action_block,
                            trees=args.trees,
                            seed=args.seed + 3450,
                            previous_actions=(
                                direct_previous if args.direct_use_previous_action else None
                            ),
                            feature_interface=args.torque_feature_interface,
                            goals_per_state=args.direct_ranker_goals_per_state,
                        )
                    )
                    if args.direct_dagger_rounds > 0:
                        dagger_states, dagger_goals, dagger_previous = (
                            collect_pure_learned_dagger_states(
                                env,
                                models["block_only_torque_readout"]["goal_feasibility"],
                                direct_states,
                                direct_previous,
                                action_block=args.action_block,
                                directions=args.local_effect_directions,
                                magnitudes=tuple(args.direct_action_magnitudes),
                                feature_interface=args.torque_feature_interface,
                                rounds=args.direct_dagger_rounds,
                                states_per_round=args.direct_dagger_states_per_round,
                                steps_per_rollout=args.direct_dagger_steps,
                                seed=args.seed + 3470,
                            )
                        )
                        if len(dagger_states) > 0:
                            initial_goals = np.full_like(direct_states, np.nan)
                            augmented_states = np.concatenate(
                                [direct_states, dagger_states], axis=0
                            )
                            augmented_previous = np.concatenate(
                                [direct_previous, dagger_previous], axis=0
                            )
                            augmented_goals = np.concatenate(
                                [initial_goals, dagger_goals], axis=0
                            )
                            models["block_only_torque_readout"]["goal_feasibility"] = (
                                fit_goal_conditioned_phase_feasibility_heads(
                                    env,
                                    augmented_states,
                                    direct_actions,
                                    horizon=args.action_block,
                                    trees=args.trees,
                                    seed=args.seed + 3480,
                                    previous_actions=(
                                        augmented_previous
                                        if args.direct_use_previous_action else None
                                    ),
                                    feature_interface=args.torque_feature_interface,
                                    goals_per_state=args.direct_ranker_goals_per_state,
                                    goal_states=augmented_goals,
                                )
                            )
                            direct_states = augmented_states
                            direct_previous = augmented_previous
                            direct_dagger_state_count = int(len(dagger_states))
                            direct_training_sources.append("pure_learned_dagger")
                elif args.direct_goal_conditioned_ranker:
                    models["block_only_torque_readout"]["goal_ranker"] = (
                        fit_goal_conditioned_action_ranker(
                            env,
                            direct_states,
                            direct_actions,
                            horizon=args.action_block,
                            trees=args.trees,
                            seed=args.seed + 3400,
                            previous_actions=(
                                direct_previous if args.direct_use_previous_action else None
                            ),
                            feature_interface=args.torque_feature_interface,
                            goals_per_state=args.direct_ranker_goals_per_state,
                        )
                    )
            else:
                sequence_actions = torque_action_bank(
                    args.direct_sequence_directions,
                    magnitudes=tuple(args.direct_sequence_magnitudes),
                )
                models["block_only_torque_readout"]["direct_sequence_endpoint"] = (
                    fit_direct_contact_sequence_model(
                        env,
                        direct_states,
                        sequence_actions,
                        trees=args.trees,
                        seed=args.seed + 3300,
                        previous_actions=(
                            direct_previous if args.direct_use_previous_action else None
                        ),
                        feature_interface=args.torque_feature_interface,
                    )
                )
            direct_training_count = int(len(direct_states))
        contact_gates = {}
        for index, model_name in enumerate(PLANNER_MODEL_NAMES):
            gate_name = (
                "object_centric"
                if model_name in (
                    "block_only", "block_only_specialist", "block_only_torque_readout"
                )
                else model_name
            )
            contact_gates[model_name] = fit_contact_gate(
                train, gate_name, args.trees, args.seed + 3000 + index
            )
        cases = held_out_cases(env, args.episodes, args.seed)
        rows = []
        for model_name in PLANNER_MODEL_NAMES:
            rows.extend(
                run_arm(
                    env,
                    models[model_name],
                    model_name,
                    cases,
                    args,
                    contact_gate=contact_gates[model_name],
                )
            )
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "training": (
                "genuine approach/contact trajectories with changing normal/tangential actions"
                if args.training_protocol == "trajectory"
                else "genuine approach/contact states with held-out one-step normal/tangential interventions"
            ),
            "training_protocol": args.training_protocol,
            "models": list(PLANNER_MODEL_NAMES),
            "same_cases": True,
            "same_candidate_bank_seed_schedule": True,
            "candidate_count": args.candidates,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "verification_steps": args.action_block,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "train_transitions": args.train_transitions,
            "trees": args.trees,
            "direct_train_states": direct_training_count,
            "direct_dagger_rounds": int(args.direct_dagger_rounds),
            "direct_dagger_states_per_round": int(args.direct_dagger_states_per_round),
            "direct_dagger_steps": int(args.direct_dagger_steps),
            "direct_dagger_state_count": int(direct_dagger_state_count),
            "direct_training_sources": direct_training_sources,
            "direct_state_pool": args.direct_state_pool,
            "direct_state_selection": (
                "agent_block_distance_le_65"
                if args.direct_state_pool == "near_contact_mixed"
                else "mode_contact"
            ),
            "direct_use_previous_action": args.direct_use_previous_action,
            "direct_uncertainty_penalty": args.direct_uncertainty_penalty,
            "direct_contact_mode_penalty": args.direct_contact_mode_penalty,
            "direct_goal_conditioned_ranker": args.direct_goal_conditioned_ranker,
            "direct_goal_conditioned_feasibility": args.direct_goal_conditioned_feasibility,
            "direct_goal_conditioned_score": args.direct_goal_conditioned_score,
            "direct_goal_conditioned_policy": args.direct_goal_conditioned_policy,
            "direct_feasibility_protocol": (
                "separate_translation_closure_heads_with_contact_preservation"
                if args.direct_goal_conditioned_feasibility else None
            ),
            "direct_ranker_goals_per_state": args.direct_ranker_goals_per_state,
            "direct_horizon_consistency_penalty": args.direct_horizon_consistency_penalty,
            "torque_feature_interface": args.torque_feature_interface,
            "direct_online_calibration": args.direct_online_calibration,
            "direct_online_local_fit": args.direct_online_local_fit,
            "direct_online_local_strategy": args.direct_online_local_strategy,
            "direct_online_local_weight": args.direct_online_local_weight,
            "direct_online_local_radius": args.direct_online_local_radius,
            "direct_online_local_neighbors": args.direct_online_local_neighbors,
            "direct_require_contact_mode": args.direct_require_contact_mode,
            "direct_contact_deployment_boundary": (
                "mode_for_state == contact (distance <=45)"
                if args.direct_require_contact_mode
                else "near_contact distance <=65 (legacy)"
            ),
            "torque_feature_interface": args.torque_feature_interface,
            "direct_endpoint_action_count": (
                int(len(torque_action_bank(
                    args.local_effect_directions,
                    magnitudes=tuple(args.direct_action_magnitudes),
                )))
                if args.planning_objective == "staged_contact_direct_torque_readout"
                else None
            ),
            "direct_sequence_action_count": (
                int(len(torque_action_bank(
                    args.direct_sequence_directions,
                    magnitudes=tuple(args.direct_sequence_magnitudes),
                )))
                if args.planning_objective == "staged_contact_direct_sequence_readout"
                else None
            ),
            "direct_sequence_count": (
                int(len(torque_action_bank(
                    args.direct_sequence_directions,
                    magnitudes=tuple(args.direct_sequence_magnitudes),
                )) ** 2)
                if args.planning_objective == "staged_contact_direct_sequence_readout"
                else None
            ),
            "direct_action_magnitudes": (
                [float(value) for value in args.direct_action_magnitudes]
                if args.planning_objective == "staged_contact_direct_torque_readout"
                else None
            ),
            "direct_sequence_magnitudes": (
                [float(value) for value in args.direct_sequence_magnitudes]
                if args.planning_objective == "staged_contact_direct_sequence_readout"
                else None
            ),
            "direct_endpoint_horizon": (
                int(args.action_block)
                if args.planning_objective in (
                    "staged_contact_direct_torque_readout",
                    "staged_contact_direct_sequence_readout",
                )
                else None
            ),
            "planning_objective": args.planning_objective,
            "contact_gate": args.planning_objective == "staged_contact_calibrated",
            "contact_gate_threshold": args.contact_gate_threshold,
            "local_effect_directions": args.local_effect_directions,
            "local_effect_scale": args.local_effect_scale,
            "local_effect_xy_margin": args.local_effect_xy_margin,
            "torque_verification_topk": args.torque_verification_topk,
            "closure_hold_xy_margin": args.closure_hold_xy_margin,
            "closure_hold_exit_xy": args.closure_hold_exit_xy,
            "terminal_return_retries": args.terminal_return_retries,
            "orientation_path_horizon": args.orientation_path_horizon,
            "orientation_path_population": args.orientation_path_population,
            "orientation_path_iterations": args.orientation_path_iterations,
            "near_goal_contact_reacquisition": (
                "bounded agent-contact recovery accepted by contact-distance decrease,"
                " XY margin, and angle margin"
                if args.planning_objective == "staged_contact_verified_hybrid_fullfan"
                else None
            ),
            "max_contact_recoveries": args.max_contact_recoveries,
            "online_residual_calibration": (
                "mode-conditioned exponential residual bias only when verification and learned-effect horizons match"
                if args.planning_objective in (
                    "staged_contact_verified_hybrid_fullfan",
                    "staged_contact_verified_torque",
                    "staged_contact_verified_torque_hold",
                    "staged_contact_verified_torque_stabilized",
                    "staged_contact_verified_torque_fullfan",
                    "staged_contact_verified_torque_hierarchical_fullfan",
                )
                else None
            ),
            "torque_contact_recovery": (
                "up to three bounded alternate contact-normal routes after two consecutive "
                "no-feasible verified torque decisions"
                if args.planning_objective == "staged_contact_verified_torque_hierarchical_fullfan"
                else None
            ),
            "seed": args.seed,
            "wall_seconds": time.perf_counter() - started,
        },
        "training": {
            "n": int(len(train["states"])),
            "mode_counts": {mode: int(np.sum(train["modes"] == mode)) for mode in ("free", "approach", "contact")},
            "finite": bool(np.isfinite(train["states"]).all() and np.isfinite(train["next_states"]).all()),
        },
        "cases": [{"episode": i, "mode": case["mode"]} for i, case in enumerate(cases)],
        "rows": rows,
        "summary": summarize(rows, args),
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(row[metric])
                for row in rows
                for metric in ("return", "block_xy_distance", "angle_error", "agent_xy_distance")
            )),
            "models_present": sorted({row["model"] for row in rows}) == sorted(PLANNER_MODEL_NAMES),
            "same_case_count": all(sum(row["model"] == model for row in rows) == args.episodes for model in PLANNER_MODEL_NAMES),
            "candidate_bank_shared_schedule": True,
            "torque_feature_interface_recorded": True,
        },
        "claim_boundary": (
            "Privileged simulator-state learned-dynamics planning diagnostic. It tests whether an input interface "
            "supports a learned block-pose planner on this PushT protocol; it does not establish frozen-LeWM latent "
            "causality, visual online planning, or general PushT performance."
        ),
    }
    (output / "object_centric_learned_planner.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True)
    )
    lines = [
        "# Object-centric learned planner",
        "",
        "| model | success | block XY | angle | agent XY |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name in PLANNER_MODEL_NAMES:
        item = report["summary"]["by_model"][model_name]
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] |" % (
                model_name,
                item["success_rate"]["estimate"], item["success_rate"]["ci95_low"], item["success_rate"]["ci95_high"],
                item["block_xy_distance"]["estimate"], item["block_xy_distance"]["ci95_low"], item["block_xy_distance"]["ci95_high"],
                item["angle_error"]["estimate"], item["angle_error"]["ci95_low"], item["angle_error"]["ci95_high"],
                item["agent_xy_distance"]["estimate"], item["agent_xy_distance"]["ci95_low"], item["agent_xy_distance"]["ci95_high"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "object_centric_learned_planner.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summary": report["summary"], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
