"""Mode-aware shared-candidate PushT adapter audit.

This experiment implements the mechanism suggested by the literature audit:
separate successor-mode prediction from within-mode pose-effect prediction,
then rank the *same candidate bank* with a short imagined continuation.  The
runtime pass never queries a simulator successor; simulator successors are
used only to create root-disjoint training labels and to score the final
closed loop.

The experiment is intentionally privileged-state.  It is a prerequisite for
the RGB/frozen-LeWM claim, not evidence for it.  A positive result means that
the remaining failure can plausibly be repaired by a mode-aware control
interface; a negative result means that mode separation alone is insufficient.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_lifted_contact_adapter import (
    contact_aware_pulse_actions,
    execute_pulse,
    make_terminal_roots,
    support_inward_normal,
)
from experiments.run_pusht_obstacle_aware_local_effect import route_to_contact
from experiments.run_pusht_signed_torque_audit import (
    angle_error,
    bootstrap_mean,
    pulse_actions,
    wrap_angle,
)

gym.logger.min_level = 40

MODE_NAMES = ("free", "approach", "contact")
SUCCESSOR_NAMES = ("no_contact", "stalled_contact", "sliding", "separation")


def contact_chart_features(
    state: np.ndarray, action: np.ndarray, previous_action: np.ndarray | None = None,
) -> np.ndarray:
    """Low-dimensional, physics-motivated contact chart.

    The tree model above is deliberately retained as the historical baseline.
    This chart tests a narrower hypothesis: within a contact regime, the
    action effect is smoother when expressed in the block frame and expanded
    by the lever-arm/moment terms.  It does not use a successor label or a
    test-time simulator query.  The chart is intentionally fixed before
    looking at held-out outcomes; it is not a post-hoc feature search.
    """

    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous = np.zeros(2, dtype=np.float64) if previous_action is None else np.asarray(previous_action, dtype=np.float64)
    if state.shape != (7,) or action.shape != (2,) or previous.shape != (2,):
        raise ValueError("contact chart expects state (7), action (2), previous (2)")
    theta = float(state[4])
    rotation = np.asarray([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    lever_world = state[:2] - state[2:4]
    lever = rotation @ lever_world
    body_action = rotation @ action
    body_previous = rotation @ previous
    radius = float(np.linalg.norm(lever))
    phase = float(np.arctan2(lever[1], lever[0]))
    # The signed moment is the invariant first-order torque proxy in this
    # local chart.  Include radial work separately: pushing inward can preserve
    # contact while the tangential component changes orientation.
    moment = float(lever[0] * body_action[1] - lever[1] * body_action[0])
    radial_work = float(np.dot(lever, body_action))
    values: list[float] = [
        1.0, radius / 50.0, lever[0] / 50.0, lever[1] / 50.0,
        body_action[0], body_action[1], body_previous[0], body_previous[1],
        moment, radial_work, moment / max(radius, 8.0),
    ]
    for harmonic in range(1, 5):
        s, c = np.sin(harmonic * phase), np.cos(harmonic * phase)
        values.extend([s, c, s * body_action[0], s * body_action[1], c * body_action[0], c * body_action[1]])
    values.extend([
        lever[0] * body_action[0] / 50.0,
        lever[0] * body_action[1] / 50.0,
        lever[1] * body_action[0] / 50.0,
        lever[1] * body_action[1] / 50.0,
        body_action[0] ** 2, body_action[1] ** 2,
    ])
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("contact chart produced non-finite values")
    return result


def contact_phase_bin(state: np.ndarray, bins: int = 8) -> int:
    """Declared contact-geometry stratum used only for diagnostics."""

    state = np.asarray(state, dtype=np.float64)
    theta = float(state[4])
    rotation = np.asarray([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    lever = rotation @ (state[:2] - state[2:4])
    phase = (float(np.arctan2(lever[1], lever[0])) + np.pi) / (2.0 * np.pi)
    return int(np.floor(np.clip(phase, 0.0, np.nextafter(1.0, 0.0)) * int(bins)))


def physical_contact_geometry(env: gym.Env, state: np.ndarray) -> dict[str, Any]:
    """Return the nearest actual T-shape boundary element to the pusher.

    ``state[:2] - state[2:4]`` is only a center-to-center direction.  PushT's
    object is a union of two rectangles, so that direction aliases different
    edges and vertices.  The environment exposes the fixed local polygon
    geometry; transforming those vertices with the observed block pose gives
    a task-state contact chart without reading a simulator successor.
    """

    state = np.asarray(state, dtype=np.float64)
    if state.shape != (7,):
        raise ValueError("physical contact geometry expects state (7)")
    unwrapped = getattr(env, "unwrapped", env)
    block = np.asarray(state[2:4], dtype=np.float64)
    theta = float(state[4])
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    local_to_world = np.asarray([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    pusher = np.asarray(state[:2], dtype=np.float64)
    candidates: list[tuple[float, np.ndarray, np.ndarray, int, int]] = []
    shape_items = list(getattr(unwrapped.block, "shapes", ()))
    for shape_index, shape in enumerate(shape_items[:2]):
        vertices = [np.asarray([float(vertex.x), float(vertex.y)], dtype=np.float64) for vertex in shape.get_vertices()]
        for edge_index, (local_a, local_b) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
            a = block + local_to_world @ local_a
            b = block + local_to_world @ local_b
            edge = b - a
            denominator = float(np.dot(edge, edge))
            fraction = float(np.clip(np.dot(pusher - a, edge) / max(denominator, 1.0e-12), 0.0, 1.0))
            point = a + fraction * edge
            distance = float(np.linalg.norm(pusher - point))
            candidates.append((distance, point, edge / max(np.linalg.norm(edge), 1.0e-12), shape_index, edge_index))
    if not candidates:
        raise ValueError("PushT block exposes no polygon edges")
    distance, point, tangent, shape_index, edge_index = min(candidates, key=lambda item: item[0])
    # Recompute the position along the winning edge so contact-point bins can
    # distinguish two contacts on the same long edge.
    selected_shape = shape_items[shape_index]
    selected_vertices = [np.asarray([float(vertex.x), float(vertex.y)], dtype=np.float64) for vertex in selected_shape.get_vertices()]
    local_a, local_b = selected_vertices[edge_index], selected_vertices[(edge_index + 1) % len(selected_vertices)]
    world_a = block + local_to_world @ local_a
    world_b = block + local_to_world @ local_b
    world_edge = world_b - world_a
    edge_fraction = float(np.clip(np.dot(point - world_a, world_edge) / max(float(np.dot(world_edge, world_edge)), 1.0e-12), 0.0, 1.0))
    normal_world = (pusher - point) / max(float(np.linalg.norm(pusher - point)), 1.0e-8)
    world_to_body = local_to_world.T
    return {
        "distance": distance,
        "contact_point_world": point,
        "normal_world": normal_world,
        "lever_body": world_to_body @ (point - block),
        "pusher_to_contact_body": world_to_body @ (pusher - point),
        "edge_tangent_body": world_to_body @ tangent,
        "normal_body": world_to_body @ normal_world,
        "shape_index": int(shape_index),
        "edge_index": int(edge_index),
        "edge_fraction": edge_fraction,
    }


def boundary_inward_normal(env: gym.Env, state: np.ndarray) -> np.ndarray:
    """Return the action direction from the pusher into the nearest boundary.

    ``physical_contact_geometry()['normal_world']`` points from the boundary
    point towards the pusher.  The contact-preserving action is its negative:
    from the pusher towards the actual polygon boundary.  This is deliberately
    different from the legacy center/vertex heuristic and is only valid for
    the declared boundary-contact action frame.
    """

    geometry = physical_contact_geometry(env, state)
    point = np.asarray(geometry["contact_point_world"], dtype=np.float64)
    pusher = np.asarray(state, dtype=np.float64)[:2]
    direction = point - pusher
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-8:
        direction = -np.asarray(geometry["normal_world"], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
    result = direction / max(norm, 1.0e-8)
    if not np.isfinite(result).all() or float(np.linalg.norm(result)) < 0.5:
        raise ValueError("boundary inward normal is not finite/nonzero")
    return result


def boundary_recovery_direction(env: gym.Env, state: np.ndarray) -> np.ndarray:
    """Return the shortest state-only direction back to the nearest edge."""

    return boundary_inward_normal(env, state)


def action_frame_normal(
    env: gym.Env, state: np.ndarray, boundary_action_frame: bool,
) -> np.ndarray:
    """Select the declared contact action frame without hidden test access."""

    return (
        boundary_inward_normal(env, state)
        if bool(boundary_action_frame)
        else support_inward_normal(env, state)
    )


def physical_contact_chart_features(
    env: gym.Env, state: np.ndarray, action: np.ndarray,
    previous_action: np.ndarray | None = None,
) -> np.ndarray:
    """Build a chart conditioned on the actual polygon edge/lever arm."""

    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous = np.zeros(2, dtype=np.float64) if previous_action is None else np.asarray(previous_action, dtype=np.float64)
    geometry = physical_contact_geometry(env, state)
    theta = float(state[4])
    rotation = np.asarray([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    body_action = rotation @ action
    body_previous = rotation @ previous
    lever = np.asarray(geometry["lever_body"], dtype=np.float64)
    contact_vector = np.asarray(geometry["pusher_to_contact_body"], dtype=np.float64)
    tangent = np.asarray(geometry["edge_tangent_body"], dtype=np.float64)
    normal = np.asarray(geometry["normal_body"], dtype=np.float64)
    moment = float(lever[0] * body_action[1] - lever[1] * body_action[0])
    edge_tangent_action = float(np.dot(body_action, tangent))
    normal_action = float(np.dot(body_action, normal))
    inward_body = -normal
    inward_action = float(np.dot(body_action, inward_body))
    # Commanded tangential velocity is not the tangential contact force once
    # the friction cone saturates.  These fixed Coulomb-cone lifts expose the
    # effective force/moment under several declared friction hypotheses.
    friction_lift: list[float] = [inward_action, edge_tangent_action]
    for friction_mu in (0.2, 0.5, 1.0):
        tangential_force = float(np.clip(
            edge_tangent_action,
            -friction_mu * max(inward_action, 0.0),
            friction_mu * max(inward_action, 0.0),
        ))
        effective_force = inward_action * inward_body + tangential_force * tangent
        effective_moment = float(lever[0] * effective_force[1] - lever[1] * effective_force[0])
        friction_lift.extend([
            tangential_force, tangential_force / max(abs(inward_action), 1.0e-4),
            effective_moment, float(np.sign(effective_moment)),
        ])
    values: list[float] = [
        1.0, float(geometry["distance"]) / 50.0,
        *(lever / 50.0), *(contact_vector / 50.0), *tangent, *normal,
        *body_action, *body_previous, moment, edge_tangent_action, normal_action,
        float(geometry["shape_index"]), float(geometry["edge_index"]) / 4.0,
        *friction_lift,
    ]
    values.extend([
        lever[0] * body_action[0] / 50.0, lever[0] * body_action[1] / 50.0,
        lever[1] * body_action[0] / 50.0, lever[1] * body_action[1] / 50.0,
        edge_tangent_action * moment, normal_action * moment,
    ])
    for shape_index in range(2):
        values.append(float(geometry["shape_index"]) == shape_index)
    for edge_index in range(4):
        values.append(float(geometry["edge_index"]) == edge_index)
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("physical contact chart produced non-finite values")
    return result


def contact_region_key(env: gym.Env, state: np.ndarray) -> str:
    """Stable coarse key for local sign calibration, never a successor label."""

    geometry = physical_contact_geometry(env, state)
    position_bin = int(np.floor(np.clip(float(geometry["edge_fraction"]), 0.0, np.nextafter(1.0, 0.0)) * 4.0))
    return "%d:%d:%d" % (geometry["shape_index"], geometry["edge_index"], position_bin)


def current_mode(state: np.ndarray) -> int:
    """Current mode inferred from current geometry only."""

    distance = float(np.linalg.norm(np.asarray(state, dtype=np.float64)[:2] - np.asarray(state, dtype=np.float64)[2:4]))
    if distance > 75.0:
        return 0
    if distance > 45.0:
        return 1
    return 2


def current_mode_for_env(env: gym.Env, state: np.ndarray) -> int:
    """Boundary-aware mode for the non-circular T-shaped contact manifold."""

    boundary_distance = float(physical_contact_geometry(env, state)["distance"])
    # The pusher radius is 15 px.  A 9 px tolerance covers solver separation
    # and small RGB/state readout noise while keeping a real gap in approach.
    if boundary_distance <= 24.0:
        return 2
    if boundary_distance <= 70.0:
        return 1
    return 0


def successor_mode(before: np.ndarray, after: np.ndarray, env: gym.Env | None = None) -> int:
    """Offline label for the mode reached by one executed pulse.

    The label is deliberately based on externally observable simulator state,
    and is never passed to the runtime selector.
    """

    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if env is None:
        start_distance = float(np.linalg.norm(before[:2] - before[2:4]))
        end_distance = float(np.linalg.norm(after[:2] - after[2:4]))
    else:
        start_distance = float(physical_contact_geometry(env, before)["distance"])
        end_distance = float(physical_contact_geometry(env, after)["distance"])
    block_motion = float(np.linalg.norm(after[2:4] - before[2:4]))
    if start_distance > 70.0:
        return 0
    if end_distance > 70.0:
        return 3
    if block_motion > 1.0:
        return 2
    return 1


def make_boundary_contact_roots(
    env: gym.Env, count: int, seed: int,
) -> list[dict[str, np.ndarray]]:
    """Generate roots at actual T-boundary contact, not center-radius contact."""

    base_roots = make_terminal_roots(env, count, seed, root_protocol="terminal_pose")
    rng = np.random.default_rng(int(seed + 70000))
    shape_items = list(getattr(env.unwrapped.block, "shapes", ()))
    if not shape_items:
        raise ValueError("PushT block has no polygon shapes")
    roots: list[dict[str, np.ndarray]] = []
    for root in base_roots:
        state = np.asarray(root["state"], dtype=np.float64).copy()
        theta = float(state[4])
        local_to_world = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        accepted = False
        for _ in range(64):
            shape_index = int(rng.integers(0, min(2, len(shape_items))))
            shape = shape_items[shape_index]
            vertices = [np.asarray([float(vertex.x), float(vertex.y)], dtype=np.float64) for vertex in shape.get_vertices()]
            edge_index = int(rng.integers(0, len(vertices)))
            local_a = vertices[edge_index]
            local_b = vertices[(edge_index + 1) % len(vertices)]
            edge = local_b - local_a
            fraction = float(rng.uniform(0.15, 0.85))
            local_point = local_a + fraction * edge
            world_point = state[2:4] + local_to_world @ local_point
            edge_unit = edge / max(float(np.linalg.norm(edge)), 1.0e-8)
            signed_area = 0.5 * sum(
                vertices[index][0] * vertices[(index + 1) % len(vertices)][1]
                - vertices[(index + 1) % len(vertices)][0] * vertices[index][1]
                for index in range(len(vertices))
            )
            # Pymunk polygons are CCW here, so the right-hand normal is
            # outward; retain the winding branch for portability.
            outward_local = np.asarray([edge_unit[1], -edge_unit[0]])
            if signed_area < 0.0:
                outward_local = -outward_local
            outward = local_to_world @ outward_local
            candidate = state.copy()
            candidate[:2] = np.clip(world_point + 15.0 * outward, 25.0, 485.0)
            gap = float(physical_contact_geometry(env, candidate)["distance"])
            state = candidate
            if 12.0 <= gap <= 18.0:
                accepted = True
                break
        if not accepted:
            raise RuntimeError("could not sample a valid boundary-contact root")
        roots.append({"state": state, "goal": np.asarray(root["goal"], dtype=np.float64).copy()})
    return roots


def boundary_lifted_contact_features(
    env: gym.Env, state: np.ndarray, action: np.ndarray,
    previous_action: np.ndarray | None = None,
    previous_observed_angle_delta: float = 0.0,
) -> np.ndarray:
    """Lift the learned feature map with the same polygon frame used to act.

    The historical lifted map is center-relative.  For the boundary protocol,
    actions are generated from the nearest polygon edge, so the predictor must
    receive edge-normal/tangent coordinates as well.  The old map is retained
    as a prefix to preserve the baseline features; the appended terms are
    fixed before held-out evaluation and contain no successor information.
    """

    from experiments.run_pusht_lifted_contact_adapter import lifted_contact_features

    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous = np.zeros(2, dtype=np.float64) if previous_action is None else np.asarray(previous_action, dtype=np.float64)
    geometry = physical_contact_geometry(env, state)
    inward = boundary_inward_normal(env, state)
    tangent = np.asarray([-inward[1], inward[0]], dtype=np.float64)
    edge_tangent = np.asarray(geometry["edge_tangent_body"], dtype=np.float64)
    theta = float(state[4])
    rotation = np.asarray([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    body_action = rotation @ action
    body_previous = rotation @ previous
    lever = np.asarray(geometry["lever_body"], dtype=np.float64)
    values = lifted_contact_features(state, action, previous).tolist()
    values.extend([
        float(geometry["distance"]) / 50.0,
        *(np.asarray(geometry["pusher_to_contact_body"], dtype=np.float64) / 50.0),
        *(lever / 50.0), *inward, *tangent, *edge_tangent,
        float(np.dot(action, inward)), float(np.dot(action, tangent)),
        float(np.dot(previous, inward)), float(np.dot(previous, tangent)),
        float(np.dot(body_action, edge_tangent)),
        float(geometry["shape_index"]), float(geometry["edge_index"]) / 4.0,
        float(geometry["edge_fraction"]),
    ])
    # Reuse the same friction-cone lift as the explicit chart so tree and
    # Ridge predictors see identical action mechanics under this protocol.
    values.extend(physical_contact_chart_features(env, state, action, previous).tolist())
    # The current state does not expose block angular velocity.  Retain the
    # last observed angle increment as a causal finite-memory coordinate so a
    # delayed impulse is not mistaken for the current action effect.
    values.extend([
        float(previous_observed_angle_delta),
        abs(float(previous_observed_angle_delta)),
        float(np.sign(previous_observed_angle_delta)),
    ])
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("boundary lifted features produced non-finite values")
    return result


def lifted_row_features(
    row: dict[str, Any], env: gym.Env | None = None,
    boundary_action_frame: bool = False,
) -> np.ndarray:
    from experiments.run_pusht_lifted_contact_adapter import lifted_contact_features

    if boundary_action_frame:
        if env is None:
            raise ValueError("boundary action features require the PushT environment geometry")
        return boundary_lifted_contact_features(
            env, np.asarray(row["before"], dtype=np.float64),
            np.asarray(row["action"], dtype=np.float64),
            np.asarray(row.get("previous_action", np.zeros(2)), dtype=np.float64),
            float(row.get("previous_observed_angle_delta", 0.0)),
        )
    return lifted_contact_features(
        np.asarray(row["before"], dtype=np.float64),
        np.asarray(row["action"], dtype=np.float64),
        np.asarray(row.get("previous_action", np.zeros(2)), dtype=np.float64),
    )


def transition_target(row: dict[str, Any]) -> np.ndarray:
    before = np.asarray(row["before"], dtype=np.float64)
    after = np.asarray(row["after"], dtype=np.float64)
    before_distance = np.linalg.norm(before[:2] - before[2:4])
    after_distance = np.linalg.norm(after[:2] - after[2:4])
    return np.asarray([
        float(wrap_angle(after[4] - before[4])),
        *(after[2:4] - before[2:4]),
        *(after[:2] - before[:2]),
        float(after_distance - before_distance),
    ], dtype=np.float64)


def transition_target_from_states(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Observed transition target for the runtime effect cache."""

    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    before_distance = np.linalg.norm(before[:2] - before[2:4])
    after_distance = np.linalg.norm(after[:2] - after[2:4])
    return np.asarray([
        float(wrap_angle(after[4] - before[4])),
        *(after[2:4] - before[2:4]),
        *(after[:2] - before[:2]),
        float(after_distance - before_distance),
    ], dtype=np.float64)


def online_effect_key(env: gym.Env, state: np.ndarray, action: np.ndarray) -> str:
    """Key an observed effect by the full local action chart.

    A previous version keyed only by contact region, moment sign, and action
    norm.  That aliases candidates with different normal/tangent components:
    they can have the same first-order moment while inducing different
    sticking/sliding transitions.  Runtime observations must therefore only
    transfer to the same quantized local action direction and edge location.
    """

    region = contact_region_key(env, state)
    geometry = physical_contact_geometry(env, state)
    inward = boundary_inward_normal(env, state)
    tangent = np.asarray([-inward[1], inward[0]], dtype=np.float64)
    vector = np.asarray(action, dtype=np.float64)
    normal_component = round(float(np.dot(vector, inward)), 3)
    tangent_component = round(float(np.dot(vector, tangent)), 3)
    edge_fraction = round(float(geometry["edge_fraction"]), 2)
    return "%s:n%.3f:t%.3f:e%.2f" % (
        region, normal_component, tangent_component, edge_fraction,
    )


def fit_regressor(
    rows: list[dict[str, Any]], seed: int, trees: int,
    env: gym.Env | None = None, boundary_action_frame: bool = False,
) -> ExtraTreesRegressor:
    features = np.asarray([
        lifted_row_features(row, env=env, boundary_action_frame=boundary_action_frame)
        for row in rows
    ], dtype=np.float64)
    targets = np.asarray([transition_target(row) for row in rows], dtype=np.float64)
    model = ExtraTreesRegressor(
        n_estimators=int(trees), min_samples_leaf=2, max_features=0.8,
        random_state=int(seed), n_jobs=1,
    )
    model.fit(features, targets)
    return model


def goal_ranker_features(
    row_or_state: dict[str, Any] | np.ndarray,
    goal: np.ndarray | None = None,
    action: np.ndarray | None = None,
    previous_action: np.ndarray | None = None,
    previous_observed_angle_delta: float = 0.0,
    env: gym.Env | None = None,
    boundary_action_frame: bool = False,
) -> np.ndarray:
    """Build a goal-conditioned candidate-ranking interface.

    The ordinary adapter predicts a transition and then applies a second
    scoring function.  This interface is a preregistered diagnostic of that
    composition: it predicts the one-step pose cost directly from training
    candidates.  It remains successor-free at runtime and uses only the
    observed state, declared goal, action, and finite-memory coordinates.
    """

    if isinstance(row_or_state, dict):
        row = row_or_state
        state = np.asarray(row["before"], dtype=np.float64)
        goal_value = np.asarray(row["goal"], dtype=np.float64)
        action_value = np.asarray(row["action"], dtype=np.float64)
        previous_value = np.asarray(row.get("previous_action", np.zeros(2)), dtype=np.float64)
        delta_value = float(row.get("previous_observed_angle_delta", 0.0))
    else:
        state = np.asarray(row_or_state, dtype=np.float64)
        if goal is None or action is None:
            raise ValueError("goal and action are required for state-based ranking features")
        goal_value = np.asarray(goal, dtype=np.float64)
        action_value = np.asarray(action, dtype=np.float64)
        previous_value = np.zeros(2, dtype=np.float64) if previous_action is None else np.asarray(previous_action, dtype=np.float64)
        delta_value = float(previous_observed_angle_delta)
    base = lifted_row_features(
        {"before": state, "action": action_value, "previous_action": previous_value,
         "previous_observed_angle_delta": delta_value},
        env=env, boundary_action_frame=boundary_action_frame,
    )
    goal_delta = goal_value[2:4] - state[2:4]
    task = np.asarray([
        float(wrap_angle(goal_value[4] - state[4])) / np.pi,
        float(goal_delta[0]) / 100.0,
        float(goal_delta[1]) / 100.0,
        float(np.linalg.norm(goal_delta)) / 100.0,
    ], dtype=np.float64)
    result = np.concatenate([base, task])
    if not np.isfinite(result).all():
        raise ValueError("goal ranker features produced non-finite values")
    return result


def fit_goal_ranker(
    rows: list[dict[str, Any]], seed: int, trees: int,
    env: gym.Env | None = None, boundary_action_frame: bool = False,
) -> ExtraTreesRegressor:
    """Fit a training-only direct one-step pose-cost ranker."""

    features = np.asarray([
        goal_ranker_features(row, env=env, boundary_action_frame=boundary_action_frame)
        for row in rows
    ], dtype=np.float64)
    targets = np.asarray([
        pose_cost(np.asarray(row["after"], dtype=np.float64), np.asarray(row["goal"], dtype=np.float64))
        for row in rows
    ], dtype=np.float64)
    model = ExtraTreesRegressor(
        n_estimators=int(trees), min_samples_leaf=2, max_features=0.8,
        random_state=int(seed), n_jobs=1,
    )
    model.fit(features, targets)
    return model


def fit_contact_chart(
    rows: list[dict[str, Any]], seed: int, alpha: float = 10.0,
    env: gym.Env | None = None,
) -> dict[str, Any]:
    """Fit a fixed contact-chart effect model on training rows only.

    A global chart plus current-mode charts gives the model a controlled
    fallback hierarchy without fitting a separate model for every contact
    phase (which would be unidentified with the small root-disjoint smoke).
    Standardization is fit inside each pipeline on the training subset only.
    """

    def fit_subset(subset: list[dict[str, Any]], local_seed: int) -> Any:
        if env is None:
            features = np.asarray([
                contact_chart_features(row["before"], row["action"], row.get("previous_action"))
                for row in subset
            ], dtype=np.float64)
        else:
            features = np.asarray([
                physical_contact_chart_features(env, row["before"], row["action"], row.get("previous_action"))
                for row in subset
            ], dtype=np.float64)
        targets = np.asarray([transition_target(row) for row in subset], dtype=np.float64)
        # Ridge is deterministic for a fixed design; the seed is recorded in
        # the artifact to make the protocol explicit even though it is not
        # consumed by the estimator.
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(features, targets)
        return model

    by_mode: dict[int, Any] = {}
    counts: dict[str, int] = {}
    for mode in range(3):
        subset = [row for row in rows if (current_mode_for_env(env, row["before"]) if env is not None else current_mode(row["before"])) == mode]
        counts[MODE_NAMES[mode]] = len(subset)
        if len(subset) >= 24:
            by_mode[mode] = fit_subset(subset, seed + 100 + mode)
    return {
        "global": fit_subset(rows, seed),
        "by_mode": by_mode,
        "alpha": float(alpha),
        "feature_dim": int((
            physical_contact_chart_features(env, rows[0]["before"], rows[0]["action"], rows[0].get("previous_action"))
            if env is not None else contact_chart_features(rows[0]["before"], rows[0]["action"], rows[0].get("previous_action"))
        ).shape[0]),
        "geometry_source": "PushT polygon boundary nearest edge" if env is not None else "center-to-center fallback",
        "counts": counts,
    }


def fit_mode_aware(
    rows: list[dict[str, Any]], seed: int, trees: int, env: gym.Env | None = None,
    boundary_action_frame: bool = False, contact_chart_alpha: float = 10.0,
) -> dict[str, Any]:
    features = np.asarray([
        lifted_row_features(row, env=env, boundary_action_frame=boundary_action_frame)
        for row in rows
    ], dtype=np.float64)
    labels = np.asarray([successor_mode(row["before"], row["after"], env=env) for row in rows], dtype=np.int64)
    angle_labels = np.asarray([int(transition_target(row)[0] > 1.0e-5) for row in rows], dtype=np.int64)
    classifier = ExtraTreesClassifier(
        n_estimators=int(trees), min_samples_leaf=2, max_features=0.8,
        class_weight="balanced", random_state=int(seed + 1), n_jobs=1,
    )
    classifier.fit(features, labels)
    direction_classifier = ExtraTreesClassifier(
        n_estimators=int(trees), min_samples_leaf=2, max_features=0.8,
        class_weight="balanced", random_state=int(seed + 2), n_jobs=1,
    )
    direction_classifier.fit(features, angle_labels)
    by_mode: dict[int, ExtraTreesRegressor] = {}
    by_successor: dict[int, ExtraTreesRegressor] = {}
    by_mode_successor: dict[tuple[int, int], ExtraTreesRegressor] = {}
    counts: dict[str, int] = {}
    for mode in range(3):
        subset = [row for row in rows if (current_mode_for_env(env, row["before"]) if env is not None else current_mode(row["before"])) == mode]
        counts[MODE_NAMES[mode]] = len(subset)
        if len(subset) >= 24:
            by_mode[mode] = fit_regressor(subset, seed + 10 + mode, trees, env, boundary_action_frame)
    # A single regressor per current mode still averages incompatible
    # successor modes (sticking, sliding, separation).  Fit successor
    # experts as well; inference will marginalize them with the learned mode
    # probabilities.  The minimum is fixed before seeing held-out rows so a
    # sparse mode fails back to a broader expert instead of overfitting.
    successor_counts: dict[str, int] = {}
    for successor in range(len(SUCCESSOR_NAMES)):
        subset = [row for row, label in zip(rows, labels) if int(label) == successor]
        successor_counts[SUCCESSOR_NAMES[successor]] = len(subset)
        if len(subset) >= 12:
            by_successor[successor] = fit_regressor(subset, seed + 60 + successor, trees, env, boundary_action_frame)
        for mode in range(3):
            mode_subset = [
                row for row, label in zip(rows, labels)
                if (current_mode_for_env(env, row["before"]) if env is not None else current_mode(row["before"])) == mode and int(label) == successor
            ]
            if len(mode_subset) >= 12:
                by_mode_successor[(mode, successor)] = fit_regressor(
                    mode_subset, seed + 80 + 4 * mode + successor, trees,
                    env, boundary_action_frame,
                )
    return {
        "global": fit_regressor(rows, seed + 40, trees, env, boundary_action_frame),
        "goal_ranker": fit_goal_ranker(
            rows, seed + 180, trees, env=env,
            boundary_action_frame=boundary_action_frame,
        ),
        "contact_chart": fit_contact_chart(
            rows, seed + 140, alpha=contact_chart_alpha, env=env,
        ),
        "by_mode": by_mode,
        "by_successor": by_successor,
        "by_mode_successor": by_mode_successor,
        "classifier": classifier,
        "direction_classifier": direction_classifier,
        "boundary_action_frame": bool(boundary_action_frame),
        "counts": {**counts, "successor": successor_counts},
    }


def predict_transition(
    model: dict[str, Any], state: np.ndarray, action: np.ndarray,
    previous_action: np.ndarray, *, mode_conditioned: bool, predictor: str = "tree",
    env: gym.Env | None = None,
    online_effects: dict[str, dict[str, Any]] | None = None,
    boundary_action_frame: bool = False,
    previous_observed_angle_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    from experiments.run_pusht_lifted_contact_adapter import lifted_contact_features

    if boundary_action_frame:
        if env is None:
            raise ValueError("boundary action prediction requires the PushT environment geometry")
        features = boundary_lifted_contact_features(
            env, state, action, previous_action, previous_observed_angle_delta,
        )[None]
    else:
        features = lifted_contact_features(state, action, previous_action)[None]
    mode = current_mode_for_env(env, state) if env is not None else current_mode(state)
    probabilities = np.zeros(4, dtype=np.float64)
    probabilities[np.asarray(model["classifier"].classes_, dtype=np.int64)] = np.asarray(
        model["classifier"].predict_proba(features)[0], dtype=np.float64
    )
    # The physical chart is identified from polygon contact geometry.  It is
    # intentionally not extrapolated into approach/free space; those states
    # use the historical tree predictor.  This is an applicability-domain
    # gate, not a test-outcome-tuned weight.
    if predictor == "contact_chart" and mode == 2:
        chart = model["contact_chart"]
        chart_model = chart["by_mode"].get(mode, chart["global"])
        chart_features = np.asarray([(
            physical_contact_chart_features(env, state, action, previous_action)
            if env is not None else contact_chart_features(state, action, previous_action)
        )], dtype=np.float64)
        target = np.asarray(chart_model.predict(chart_features)[0], dtype=np.float64)
    elif mode_conditioned and model.get("by_successor"):
        # The predictor is a finite mixture of successor-mode experts rather
        # than one regressor trained across discontinuous transitions.  When
        # a mode-specific expert is supported, it takes precedence; otherwise
        # the broader successor expert is used, then the current-mode/global
        # fallback.  This keeps the runtime interface fail-closed for sparse
        # successor classes.
        fallback = model["by_mode"].get(mode, model["global"])
        target_components = []
        for successor in range(len(SUCCESSOR_NAMES)):
            expert = model["by_mode_successor"].get((mode, successor))
            if expert is None:
                expert = model["by_successor"].get(successor, fallback)
            target_components.append(np.asarray(expert.predict(features)[0], dtype=np.float64))
        target = np.sum(
            np.asarray(probabilities, dtype=np.float64)[:, None] * np.asarray(target_components),
            axis=0,
        )
    else:
        regressor = model["by_mode"].get(mode, model["global"]) if mode_conditioned else model["global"]
        target = np.asarray(regressor.predict(features)[0], dtype=np.float64)
    if online_effects is not None and env is not None and mode == 2:
        cache = online_effects.get(online_effect_key(env, state, action))
        if cache is not None and int(cache.get("count", 0)) > 0:
            # The action bank uses fixed magnitudes.  Use the observed local
            # transition for a repeated region/sign/magnitude and retain the
            # learned model for unseen candidates (fail-closed coverage).
            target = np.asarray(cache["mean"], dtype=np.float64).copy()
    direction_probability = 0.5
    direction_classes = np.asarray(model["direction_classifier"].classes_, dtype=np.int64)
    direction_values = np.asarray(model["direction_classifier"].predict_proba(features)[0], dtype=np.float64)
    positive = np.flatnonzero(direction_classes == 1)
    if len(positive):
        direction_probability = float(direction_values[int(positive[0])])
    elif len(direction_classes):
        direction_probability = float(direction_classes[0] == 1)
    return target, probabilities, direction_probability


def predicted_state(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).copy()
    state[4] = float(wrap_angle(state[4] + target[0]))
    state[2:4] += target[1:3]
    state[:2] += target[3:5]
    return state


def pose_cost(state: np.ndarray, goal: np.ndarray) -> float:
    return angle_error(state[4], goal[4]) / (np.pi / 9.0) + np.linalg.norm(state[2:4] - goal[2:4]) / 20.0


def planning_pose_cost(
    current: np.ndarray, imagined: np.ndarray, goal: np.ndarray, phase_aware: bool,
) -> float:
    """Score an imagined transition without letting rotation consume translation.

    The legacy weighted score can prefer a large angle improvement even after
    the block has started leaving its XY tube.  The phase-aware variant is a
    fixed lexicographic-style diagnostic: translate while XY error is large;
    rotate only after translation is close and penalize leaving the tube.
    """

    if not phase_aware:
        return pose_cost(imagined, goal)
    current_xy = float(np.linalg.norm(np.asarray(current)[2:4] - np.asarray(goal)[2:4]))
    imagined_xy = float(np.linalg.norm(np.asarray(imagined)[2:4] - np.asarray(goal)[2:4]))
    current_angle = angle_error(current[4], goal[4])
    imagined_angle = angle_error(imagined[4], goal[4])
    if current_xy > 28.0:
        # Translation is primary; avoid accepting a large angular regression.
        return imagined_xy / 20.0 + 2.0 * max(0.0, imagined_angle - current_angle) / (np.pi / 9.0)
    tube_penalty = max(0.0, imagined_xy - current_xy - 8.0) / 20.0
    return (
        6.0 * imagined_angle / (np.pi / 9.0)
        + 0.8 * imagined_xy / 20.0
        + 0.25 * tube_penalty
    )


def candidate_actions(
    env: gym.Env, state: np.ndarray, tangent_magnitudes: tuple[float, ...], inward_magnitude: float,
    free_recovery_magnitude: float = 0.5,
    goal: np.ndarray | None = None,
    goal_translation_magnitudes: tuple[float, ...] = (),
    boundary_action_frame: bool = False,
) -> np.ndarray:
    if current_mode_for_env(env, state) == 0:
        direction = np.asarray(state, dtype=np.float64)[2:4] - np.asarray(state, dtype=np.float64)[:2]
        direction /= max(float(np.linalg.norm(direction)), 1.0e-8)
        return np.asarray([float(free_recovery_magnitude) * direction], dtype=np.float64)
    normal = action_frame_normal(env, state, boundary_action_frame)
    mode = current_mode_for_env(env, state)
    if boundary_action_frame:
        # The corrected protocol already has an explicit polygon-gap mode
        # gate.  Reusing contact_aware_pulse_actions here would introduce the
        # legacy center-distance (38 px) gate a second time, making boundary
        # contacts with different T edges receive different pulse amplitudes.
        actions, _ = pulse_actions(state, tangent_magnitudes, inward_magnitude, normal)
    else:
        actions, _ = contact_aware_pulse_actions(
            state, tangent_magnitudes, inward_magnitude, normal,
        )
    # The old bank contained only inward+tangent pulses.  Consequently the
    # declared ``separation`` successor had zero support and could not be
    # learned or penalized.  Add one explicit outward probe for contact states
    # and use the identical bank in collection, ranking, lookahead, and
    # closed-loop execution.  It is a safety candidate, not a preferred
    # action; the learned mode risk must learn to reject it when unnecessary.
    if mode == 2:
        outward = (-float(inward_magnitude) * np.asarray(normal, dtype=np.float64))[None, :]
        actions = np.concatenate((np.asarray(actions, dtype=np.float64), outward), axis=0)
    if (
        goal is not None
        and len(goal_translation_magnitudes)
        and mode in (1, 2)
        and angle_error(state[4], np.asarray(goal, dtype=np.float64)[4]) <= np.pi / 6.0
    ):
        # The original bank only varied the contact tangent.  Add a declared
        # task-space translation direction so that a nearly correct angle does
        # not force the controller to keep rotating the object to reach its XY
        # target.  The normal component remains inward, so these candidates are
        # still subject to the same contact lease and are not privileged-state
        # successor queries.
        goal_delta = np.asarray(goal, dtype=np.float64)[2:4] - np.asarray(state, dtype=np.float64)[2:4]
        tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
        tangential_goal = goal_delta - float(np.dot(goal_delta, normal)) * normal
        tangent_norm = float(np.linalg.norm(tangential_goal))
        if tangent_norm > 1.0e-8:
            tangent_sign = float(np.sign(np.dot(tangential_goal, tangent)))
            if abs(tangent_sign) > 0.5:
                for magnitude in goal_translation_magnitudes:
                    actions = np.concatenate((
                        np.asarray(actions, dtype=np.float64),
                        (float(inward_magnitude) * normal
                         + tangent_sign * float(magnitude) * tangent)[None, :],
                    ), axis=0)
        goal_norm = float(np.linalg.norm(goal_delta))
        if goal_norm > 1.0e-8:
            goal_direction = goal_delta / goal_norm
            if float(np.dot(goal_direction, normal)) >= 0.0:
                for magnitude in goal_translation_magnitudes:
                    actions = np.concatenate((
                        np.asarray(actions, dtype=np.float64),
                        (float(magnitude) * goal_direction)[None, :],
                    ), axis=0)
    return np.asarray(actions, dtype=np.float64)


def collect_mode_balanced_rows(
    env: gym.Env,
    roots: list[dict[str, np.ndarray]],
    tangent_magnitudes: tuple[float, ...],
    inward_magnitude: float,
    free_recovery_magnitude: float,
    pulse_steps: int,
    sequence_steps: int,
    seed: int,
    goal_translation_magnitudes: tuple[float, ...] = (),
    boundary_action_frame: bool = False,
) -> list[dict[str, Any]]:
    """Collect candidate groups while deliberately covering recovery states."""

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    next_root_id = 0
    for root in roots:
        env.reset(seed=0, options={"state": root["state"].copy(), "goal_state": root["goal"].copy()})
        current = np.asarray(root["state"], dtype=np.float64).copy()
        goal = np.asarray(root["goal"], dtype=np.float64).copy()
        previous = np.zeros(2, dtype=np.float64)
        previous_observed_angle_delta = 0.0
        for _ in range(int(sequence_steps)):
            actions = candidate_actions(
                env, current, tangent_magnitudes, inward_magnitude,
                free_recovery_magnitude,
                goal=goal,
                goal_translation_magnitudes=goal_translation_magnitudes,
                boundary_action_frame=boundary_action_frame,
            )
            support_normal = action_frame_normal(env, current, boundary_action_frame)
            group_id = int(next_root_id)
            next_root_id += 1
            outcomes = []
            for action_id, action in enumerate(actions):
                result = execute_pulse(env, current, goal, action, pulse_steps)
                row = {
                    "root_id": group_id,
                    "action_id": int(action_id),
                    "state": current.copy(),
                    "before": result["before"].copy(),
                    "goal": goal.copy(),
                    "action": np.asarray(action, dtype=np.float64),
                    "previous_action": previous.copy(),
                    "previous_observed_angle_delta": float(previous_observed_angle_delta),
                    "moment_sign": action_moment_sign(current, action),
                    "support_normal": support_normal.copy(),
                    "action_frame": "boundary_edge" if boundary_action_frame else "legacy_support_vertex",
                    **result,
                }
                rows.append(row)
                outcomes.append(result)
            # A free-state group must advance through the explicit recovery
            # arm; otherwise aggregation immediately returns to the same
            # uninformative free state and never learns its transition.
            chosen = 0 if current_mode_for_env(env, current) == 0 else int(rng.integers(0, len(actions)))
            current = np.asarray(outcomes[chosen]["after"], dtype=np.float64).copy()
            previous = np.asarray(actions[chosen], dtype=np.float64).copy()
            previous_observed_angle_delta = float(wrap_angle(
                current[4] - np.asarray(outcomes[chosen]["before"], dtype=np.float64)[4]
            ))
    return rows


def action_moment_sign(
    state: np.ndarray, action: np.ndarray, env: gym.Env | None = None,
) -> float:
    """Signed moment proxy; use the actual polygon contact lever when given."""

    if env is not None:
        theta = float(np.asarray(state, dtype=np.float64)[4])
        rotation = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        lever = rotation @ np.asarray(physical_contact_geometry(env, state)["lever_body"], dtype=np.float64)
    else:
        lever = np.asarray(state, dtype=np.float64)[:2] - np.asarray(state, dtype=np.float64)[2:4]
    action = np.asarray(action, dtype=np.float64)
    value = float(lever[0] * action[1] - lever[1] * action[0])
    return float(np.sign(value))


def smallest_action_for_moment_sign(
    state: np.ndarray, actions: np.ndarray, desired_sign: float,
    env: gym.Env | None = None,
) -> int | None:
    """Return the least invasive candidate with a requested geometric sign.

    The sign is only a probe direction.  It is intentionally not treated as
    the physical torque sign until an observed angle increment calibrates the
    local contact convention.
    """

    if abs(float(desired_sign)) < 0.5:
        return None
    signs = np.asarray([action_moment_sign(state, action, env=env) for action in actions])
    valid = np.flatnonzero(signs == float(np.sign(desired_sign)))
    if not len(valid):
        return None
    return int(valid[np.argmin([np.linalg.norm(actions[index]) for index in valid])])


def update_observed_sign_calibration(
    current_gain_sign: float | None, action_sign: float, observed_angle_delta: float,
    threshold: float,
) -> tuple[float | None, bool]:
    """Infer the local mapping from geometric moment sign to angle sign.

    PushT contact geometry can reverse the naive world-frame cross-product
    convention.  The update uses only the committed action and the next RGB/
    state-derived angle readout; a near-zero response leaves the calibration
    unresolved and therefore fail-closed.
    """

    if abs(float(observed_angle_delta)) <= float(threshold) or abs(float(action_sign)) < 0.5:
        return current_gain_sign, False
    sample = float(np.sign(observed_angle_delta) * np.sign(action_sign))
    if current_gain_sign is None:
        return sample, True
    # A later informative transition can correct a one-off contact impulse,
    # but no averaging is hidden here: the sign remains an explicit runtime
    # state and is reported in the trace.
    return sample, True


def update_mode_sign_consensus(
    votes: dict[int, int], mode: int, action_sign: float,
    observed_angle_delta: float, threshold: float,
) -> tuple[float | None, bool]:
    """Update a mode-local sign vote from any informative executed pulse."""

    if abs(float(observed_angle_delta)) <= float(threshold) or abs(float(action_sign)) < 0.5:
        return None, False
    sample = int(np.sign(observed_angle_delta) * np.sign(action_sign))
    votes[int(mode)] = int(votes.get(int(mode), 0) + sample)
    total = int(votes[int(mode)])
    return (float(np.sign(total)) if total != 0 else None), True


def update_contact_sign_consensus(
    votes: dict[str, int], region_key: str, action_sign: float,
    observed_angle_delta: float, threshold: float,
) -> tuple[float | None, bool]:
    """Update a sign vote for one coarse physical contact region."""

    if abs(float(observed_angle_delta)) <= float(threshold) or abs(float(action_sign)) < 0.5:
        return None, False
    sample = int(np.sign(observed_angle_delta) * np.sign(action_sign))
    votes[region_key] = int(votes.get(region_key, 0) + sample)
    total = int(votes[region_key])
    return (float(np.sign(total)) if total != 0 else None), True


def imagined_score(
    env: gym.Env,
    model: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    tangent_magnitudes: tuple[float, ...],
    inward_magnitude: float,
    goal_translation_magnitudes: tuple[float, ...],
    depth: int,
    mode_conditioned: bool,
    phase_aware: bool = False,
    predictor: str = "tree",
    online_effects: dict[str, dict[str, Any]] | None = None,
    boundary_action_frame: bool = False,
    strict_direction_gate: bool = False,
    previous_observed_angle_delta: float = 0.0,
) -> float:
    target, probabilities, direction_probability = predict_transition(
        model, state, action, previous_action, mode_conditioned=mode_conditioned,
        predictor=predictor, env=env, online_effects=online_effects,
        boundary_action_frame=boundary_action_frame,
        previous_observed_angle_delta=previous_observed_angle_delta,
    )
    imagined = predicted_state(state, target)
    score = planning_pose_cost(state, imagined, goal, phase_aware)
    current = current_mode_for_env(env, state)
    # The outward probe is necessary for identifying the separation expert,
    # but it is not a valid committed action while the pusher is in contact or
    # re-acquiring contact.  Without this lease the learned selector repeatedly chose the
    # probe (the last shared-bank action), turning a support-identification
    # intervention into an execution-time contact break.  Keep the candidate
    # in the shared bank and apply a fixed safety gate only at commitment.
    if current in (1, 2):
        support_normal = action_frame_normal(env, state, boundary_action_frame)
        inward_component = float(np.dot(np.asarray(action, dtype=np.float64), support_normal))
        if inward_component < 0.0:
            return float(score + 100.0 + 10.0 * abs(inward_component))
    # A contact-breaking successor is a hard risk, not a weighted substitute
    # for pose progress.  The penalty is smaller in free/approach mode.
    if current == 2:
        score += 8.0 * probabilities[3] + 1.5 * probabilities[0]
        desired_sign = float(np.sign(wrap_angle(goal[4] - state[4])))
        direction_match = direction_probability if desired_sign >= 0.0 else (1.0 - direction_probability)
        if strict_direction_gate and desired_sign != 0.0 and direction_match < 0.5:
            return float(score + 100.0 + 10.0 * (0.5 - direction_match))
        # A sign error is more damaging than a small magnitude error in the
        # contact regime.  Keep this as a mode-internal gate, not a global
        # weighted angle/XY trade-off.
        score += 4.0 * (1.0 - direction_match)
    else:
        score += 2.0 * probabilities[3]
    if depth <= 1:
        return float(score)
    second_scores = []
    for second in candidate_actions(
        env, imagined, tangent_magnitudes, inward_magnitude,
        goal=goal, goal_translation_magnitudes=goal_translation_magnitudes,
        boundary_action_frame=boundary_action_frame,
    ):
        second_target, second_probabilities, second_direction_probability = predict_transition(
            model, imagined, second, action, mode_conditioned=mode_conditioned,
            predictor=predictor, env=env, online_effects=online_effects,
            boundary_action_frame=boundary_action_frame,
            previous_observed_angle_delta=float(target[0]),
        )
        second_state = predicted_state(imagined, second_target)
        second_score = planning_pose_cost(imagined, second_state, goal, phase_aware)
        if current_mode_for_env(env, imagined) == 2:
            second_score += 8.0 * second_probabilities[3] + 1.5 * second_probabilities[0]
            desired_sign = float(np.sign(wrap_angle(goal[4] - imagined[4])))
            direction_match = second_direction_probability if desired_sign >= 0.0 else (1.0 - second_direction_probability)
            if strict_direction_gate and desired_sign != 0.0 and direction_match < 0.5:
                second_score += 100.0 + 10.0 * (0.5 - direction_match)
            second_score += 4.0 * (1.0 - direction_match)
        second_scores.append(second_score)
    return float(score + 0.5 * min(second_scores))


def select_action(
    env: gym.Env, model: dict[str, Any], state: np.ndarray, goal: np.ndarray,
    actions: np.ndarray, previous_action: np.ndarray, args: argparse.Namespace,
    *, mode_conditioned: bool,
    predictor: str = "tree",
    online_effects: dict[str, dict[str, Any]] | None = None,
    previous_observed_angle_delta: float = 0.0,
) -> int:
    if predictor == "goal_ranker" and model.get("goal_ranker") is not None:
        ranker_features = np.asarray([
            goal_ranker_features(
                state, goal=goal, action=action, previous_action=previous_action,
                previous_observed_angle_delta=previous_observed_angle_delta,
                env=env,
                boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            )
            for action in actions
        ], dtype=np.float64)
        predicted_costs = np.asarray(model["goal_ranker"].predict(ranker_features), dtype=np.float64)
        # Preserve the same successor-risk gate used by the effect planner;
        # only the within-bank pose ranking is changed by this ablation.
        risk_costs = predicted_costs.copy()
        current_mode = current_mode_for_env(env, state)
        if current_mode == 2:
            for index, action in enumerate(actions):
                _, probabilities, direction_probability = predict_transition(
                    model, state, action, previous_action,
                    mode_conditioned=mode_conditioned, predictor="tree", env=env,
                    online_effects=online_effects,
                    boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                    previous_observed_angle_delta=previous_observed_angle_delta,
                )
                risk_costs[index] += 8.0 * probabilities[3] + 1.5 * probabilities[0]
                desired_sign = float(np.sign(wrap_angle(goal[4] - state[4])))
                direction_match = direction_probability if desired_sign >= 0.0 else (1.0 - direction_probability)
                risk_costs[index] += 4.0 * (1.0 - direction_match)
        return int(np.argmin(risk_costs))
    scores = [
        imagined_score(
            env, model, state, goal, action, previous_action,
            args.tangent_magnitudes, args.inward_magnitude,
            args.goal_translation_magnitudes,
            args.lookahead, mode_conditioned, args.phase_aware_cost, predictor,
            online_effects, bool(getattr(args, "boundary_action_frame", False)),
            bool(getattr(args, "strict_direction_gate", False)),
            previous_observed_angle_delta,
        )
        for action in actions
    ]
    return int(np.argmin(np.asarray(scores, dtype=np.float64)))


def rank_rows(env: gym.Env, rows: list[dict[str, Any]], model: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["root_id"]), []).append(row)
    results: dict[str, list[float]] = {
        "global_one_step": [], "mode_aware_lookahead": [],
        "contact_chart_one_step": [], "contact_chart_lookahead": [],
    }
    strata: dict[str, dict[int, list[float]]] = {name: {} for name in results}
    oracle: list[float] = []
    for root_rows in grouped.values():
        actual = np.asarray([pose_cost(np.asarray(row["after"]), np.asarray(row["goal"])) for row in root_rows])
        oracle.append(float(actual.min()))
        state = np.asarray(root_rows[0]["before"], dtype=np.float64)
        goal = np.asarray(root_rows[0]["goal"], dtype=np.float64)
        previous = np.asarray(root_rows[0].get("previous_action", np.zeros(2)), dtype=np.float64)
        phase_bin = contact_phase_bin(state)
        variants = (
            ("global_one_step", False, 1, "tree"),
            ("mode_aware_lookahead", True, args.lookahead, "tree"),
            ("contact_chart_one_step", False, 1, "contact_chart"),
            ("contact_chart_lookahead", True, args.lookahead, "contact_chart"),
        )
        for name, conditioned, depth, predictor in variants:
            scores = [
                imagined_score(
                    env, model, state, goal, row["action"], previous,
                    args.tangent_magnitudes, args.inward_magnitude,
                    args.goal_translation_magnitudes,
                    depth, conditioned, args.phase_aware_cost, predictor,
                    boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                    strict_direction_gate=bool(getattr(args, "strict_direction_gate", False)),
                )
                for row in root_rows
            ]
            selected = int(np.argmin(np.asarray(scores)))
            regret = float(actual[selected] - actual.min())
            results[name].append(regret)
            strata[name].setdefault(phase_bin, []).append(regret)
    stratum_metrics = {
        name: {
            str(index): bootstrap_mean(values, 94120 + 10 * variant_index + index)
            for index, values in sorted(bins.items())
        }
        for variant_index, (name, bins) in enumerate(strata.items())
    }
    return {
        "metrics": {
            name: bootstrap_mean(values, 94100 + index)
            for index, (name, values) in enumerate(results.items())
        },
        "per_contact_phase_bin": stratum_metrics,
        "oracle_pose_cost": bootstrap_mean(oracle, 94110),
        "n_roots": len(grouped),
    }


def closed_loop(
    env: gym.Env, roots: list[dict[str, np.ndarray]], model: dict[str, Any], args: argparse.Namespace,
    *, mode_conditioned: bool, contact_recovery: bool, observed_sign_feedback: bool,
    observed_sign_calibration: bool, contact_geometry_sign_feedback: bool = False,
    recent_contact_sign_fallback: bool = False,
    strict_contact_sign_gate: bool = False,
    direct_observed_sign_reversal: bool = False,
    predictor: str = "tree",
) -> dict[str, Any]:
    records = []
    for root_index, root in enumerate(roots):
        observation, _ = env.reset(seed=0, options={"state": root["state"].copy(), "goal_state": root["goal"].copy()})
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        goal = np.asarray(root["goal"], dtype=np.float64).copy()
        previous = np.zeros(2, dtype=np.float64)
        last_observed_delta = 0.0
        moment_to_angle_sign: float | None = None
        moment_to_angle_sign_by_mode: dict[int, float] = {}
        sign_votes_by_mode: dict[int, int] = {}
        moment_to_angle_sign_by_region: dict[str, float] = {}
        sign_votes_by_region: dict[str, int] = {}
        recent_contact_sign: float | None = None
        last_action_moment_sign: float | None = None
        online_effects: dict[str, dict[str, Any]] = {}
        calibration_attempts = 0
        calibration_successes = 0
        calibration_attempted_modes: set[int] = set()
        calibration_attempted_regions: set[str] = set()
        wrong_sign_events = 0
        trace = []
        for step in range(args.closure_steps):
            if pose_cost(current, goal) < 1.0:
                break
            contact_distance = float(np.linalg.norm(current[:2] - current[2:4]))
            boundary_distance = float(physical_contact_geometry(env, current)["distance"])
            current_mode_id = current_mode_for_env(env, current)
            current_region_key = (
                contact_region_key(env, current) if current_mode_id == 2 else None
            )
            if contact_geometry_sign_feedback and current_region_key is not None:
                moment_to_angle_sign = moment_to_angle_sign_by_region.get(current_region_key)
                if (
                    moment_to_angle_sign is None
                    and recent_contact_sign_fallback
                    and not bool(getattr(args, "boundary_action_frame", False))
                ):
                    # A sign learned on another polygon edge/position is not
                    # transportable across a hybrid contact transition.  The
                    # corrected boundary protocol therefore fails closed;
                    # the legacy protocol retains the historical fallback as
                    # an explicit ablation.
                    moment_to_angle_sign = recent_contact_sign
            else:
                moment_to_angle_sign = moment_to_angle_sign_by_mode.get(current_mode_id)
            recovery = bool(contact_recovery and boundary_distance > args.recovery_threshold)
            recovery_macro = False
            recovery_info: dict[str, Any] | None = None
            recovery_before = current.copy()
            terminal_handoff = bool(
                args.terminal_handoff
                and not recovery
                and np.linalg.norm(current[2:4] - goal[2:4]) <= args.terminal_xy_gate
                and angle_error(current[4], goal[4]) > args.terminal_angle_gate
            )
            calibration_probe = False
            if recovery:
                # This is an explicit physical-interface branch.  It is not
                # learned action-effect evidence: once the observed state is
                # outside the contact tube, re-acquire the support boundary
                # before asking the learned ranker for a torque pulse.
                # The environment's asymmetric support vertex is useful for
                # torque pulses, but it is not guaranteed to point toward the
                # block when the pusher is already outside the contact tube.
                # Recovery therefore uses the unambiguous center-directed
                # vector; this branch is reported separately from the learned
                # effect model.
                if args.validated_route_recovery:
                    goal_delta = goal[2:4] - current[2:4]
                    goal_norm = float(np.linalg.norm(goal_delta))
                    desired_translation = goal_delta / max(goal_norm, 1.0e-8)
                    route_args = argparse.Namespace(
                        clearance=args.route_clearance,
                        contact_distance=args.route_contact_distance,
                        settle_steps=args.route_settle_steps,
                    )
                    current, recovery_info = route_to_contact(
                        env, current, goal, desired_translation, route_args,
                    )
                    actions = np.zeros((1, 2), dtype=np.float64)
                    selected = 0
                    recovery_macro = True
                elif args.goal_directed_recovery:
                    goal_delta = goal[2:4] - current[2:4]
                    goal_norm = float(np.linalg.norm(goal_delta))
                    if goal_norm > 1.0e-8:
                        desired_translation = goal_delta / goal_norm
                        target_contact = current[2:4] - args.recovery_contact_radius * desired_translation
                        center_direction = target_contact - current[:2]
                    else:
                        center_direction = current[2:4] - current[:2]
                else:
                    center_direction = (
                        boundary_recovery_direction(env, current)
                        if bool(getattr(args, "boundary_action_frame", False))
                        else current[2:4] - current[:2]
                    )
                if not recovery_macro:
                    center_direction = center_direction / max(float(np.linalg.norm(center_direction)), 1.0e-8)
                    actions = np.asarray([
                        args.recovery_magnitude * center_direction,
                    ], dtype=np.float64)
                    selected = 0
            else:
                actions = candidate_actions(
                    env, current, args.tangent_magnitudes, args.inward_magnitude,
                    goal=goal, goal_translation_magnitudes=args.goal_translation_magnitudes,
                    boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                )
                runtime_mode_conditioned = bool(mode_conditioned and current_mode_id == 2)
                calibration_probe = bool(
                    observed_sign_calibration
                    and moment_to_angle_sign is None
                    and current_mode_id == 2
                    and (
                        current_region_key not in calibration_attempted_regions
                        if contact_geometry_sign_feedback and current_region_key is not None
                        else current_mode_id not in calibration_attempted_modes
                    )
                )
                if calibration_probe:
                    # Use one of the smallest available signed candidates and
                    # observe its angle response before trusting the geometric
                    # cross-product convention.  This is a real runtime probe,
                    # not a simulator successor query.
                    probe = smallest_action_for_moment_sign(
                        current, actions, 1.0,
                        env=env if contact_geometry_sign_feedback else None,
                    )
                    if probe is None:
                        probe = smallest_action_for_moment_sign(
                            current, actions, -1.0,
                            env=env if contact_geometry_sign_feedback else None,
                        )
                    if probe is not None:
                        selected = probe
                        calibration_attempts += 1
                        if contact_geometry_sign_feedback and current_region_key is not None:
                            calibration_attempted_regions.add(current_region_key)
                        else:
                            calibration_attempted_modes.add(current_mode_id)
                    else:
                        calibration_probe = False
                elif terminal_handoff:
                    desired_sign = float(np.sign(wrap_angle(goal[4] - current[4])))
                    desired_moment_sign = desired_sign if moment_to_angle_sign is None else desired_sign * moment_to_angle_sign
                    selected = smallest_action_for_moment_sign(
                        current, actions, desired_moment_sign,
                        env=env if contact_geometry_sign_feedback else None,
                    )
                    if selected is None:
                        selected = select_action(
                            env, model, current, goal, actions, previous, args,
                            mode_conditioned=runtime_mode_conditioned,
                            predictor=predictor,
                            online_effects=online_effects if args.online_effect_correction else None,
                            previous_observed_angle_delta=last_observed_delta,
                        )
                else:
                    selected = select_action(
                        env, model, current, goal, actions, previous, args,
                        mode_conditioned=runtime_mode_conditioned,
                        predictor=predictor,
                        online_effects=online_effects if args.online_effect_correction else None,
                        previous_observed_angle_delta=last_observed_delta,
                    )
                if (
                    not calibration_probe
                    and observed_sign_feedback
                    and boundary_distance <= args.feedback_contact_threshold
                ):
                    desired_sign = float(np.sign(wrap_angle(goal[4] - current[4])))
                    observed_sign = float(np.sign(last_observed_delta))
                    if desired_sign != 0.0 and observed_sign != 0.0 and observed_sign != desired_sign:
                        # The previous committed pulse turned the block the
                        # wrong way.  Re-rank only candidates with the
                        # opposite observed moment sign; no simulator probe
                        # or successor is used here.
                        if direct_observed_sign_reversal and last_action_moment_sign is not None:
                            # Do not assume a portable geometry-to-torque
                            # mapping in a sticking/sliding transition.  Use
                            # the only local causal observation available: the
                            # action sign that just produced the wrong angle.
                            desired_moment_sign = -float(last_action_moment_sign)
                        elif moment_to_angle_sign is not None:
                            desired_moment_sign = desired_sign * moment_to_angle_sign
                        else:
                            desired_moment_sign = 0.0
                        replacement = smallest_action_for_moment_sign(
                            current, actions, desired_moment_sign,
                            env=env if contact_geometry_sign_feedback else None,
                        )
                        if replacement is not None:
                            selected = replacement
                if (
                    strict_contact_sign_gate
                    and not calibration_probe
                    and not direct_observed_sign_reversal
                    and current_mode_id == 2
                    and moment_to_angle_sign is not None
                ):
                    # This is a declared runtime intervention: once a local
                    # moment-to-angle sign has been identified from an actual
                    # observation, learned ranking is allowed only among
                    # candidates with the requested torque sign.  The model
                    # still ranks magnitude/translation within that subset.
                    desired_sign = float(np.sign(wrap_angle(goal[4] - current[4])))
                    desired_moment_sign = desired_sign * float(moment_to_angle_sign)
                    valid = np.asarray([
                        index for index, candidate in enumerate(actions)
                        if action_moment_sign(
                            current, candidate,
                            env=env if contact_geometry_sign_feedback else None,
                        ) == float(np.sign(desired_moment_sign))
                    ], dtype=np.int64)
                    if len(valid):
                        gated_actions = np.asarray(actions[valid], dtype=np.float64)
                        local_selected = select_action(
                            env, model, current, goal, gated_actions, previous, args,
                            mode_conditioned=runtime_mode_conditioned,
                            predictor=predictor,
                            online_effects=online_effects if args.online_effect_correction else None,
                            previous_observed_angle_delta=last_observed_delta,
                        )
                        selected = int(valid[int(local_selected)])
            action = np.asarray(actions[selected], dtype=np.float32)
            before = recovery_before if recovery_macro else current.copy()
            terminated = truncated = False
            execution_steps = (
                0 if recovery_macro
                else args.recovery_pulse_steps if recovery
                else args.calibration_pulse_steps if calibration_probe else args.pulse_steps
            )
            for _ in range(execution_steps):
                observation, _, terminated, truncated, _ = env.step(action)
                current = np.asarray(observation["state"], dtype=np.float64).copy()
                if terminated or truncated:
                    break
            trace.append({
                "step": step, "selected": selected, "action": action.tolist(),
                "block_before": before[2:4].tolist(),
                "block_goal": goal[2:4].tolist(),
                "desired_angle_sign": float(np.sign(wrap_angle(goal[4] - before[4]))),
                "action_moment_sign": float(action_moment_sign(
                    before, action, env=env if contact_geometry_sign_feedback else None,
                )),
                "mode_before": MODE_NAMES[current_mode_for_env(env, before)],
                "contact_phase_bin": int(contact_phase_bin(before)),
                "contact_geometry": {
                    "pusher_block_vector": (before[:2] - before[2:4]).tolist(),
                    "block_angle": float(before[4]),
                    "boundary_distance_before": float(physical_contact_geometry(env, before)["distance"]),
                },
                "contact_recovery": recovery,
                "goal_directed_recovery": bool(recovery and args.goal_directed_recovery),
                "validated_route_recovery": bool(recovery and args.validated_route_recovery),
                "recovery_info": None if recovery_info is None else {
                    "route_steps": int(recovery_info.get("route_steps", 0)),
                    "contact_distance": float(recovery_info.get("contact_distance", np.nan)),
                    "terminated": bool(recovery_info.get("terminated", False)),
                    "retreat_steps": int(recovery_info.get("retreat_steps", 0)),
                },
                "runtime_mode_conditioned": bool(
                    mode_conditioned and current_mode_id == 2 and not recovery
                ),
                "terminal_handoff": terminal_handoff,
                "observed_sign_feedback": bool(observed_sign_feedback),
                "observed_sign_calibration": bool(observed_sign_calibration),
                "strict_contact_sign_gate": bool(strict_contact_sign_gate),
                "direct_observed_sign_reversal": bool(direct_observed_sign_reversal),
                "calibration_probe": bool(calibration_probe),
                "moment_to_angle_sign": moment_to_angle_sign,
                "execution_steps": int(execution_steps),
                "predicted_mode_probabilities": predict_transition(
                    model, before, action, previous,
                    mode_conditioned=bool(mode_conditioned and current_mode_id == 2 and not recovery),
                    predictor=predictor, env=env,
                    online_effects=online_effects if args.online_effect_correction else None,
                    boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                    previous_observed_angle_delta=last_observed_delta,
                )[1].tolist(),
                "predicted_direction_probability": float(predict_transition(
                    model, before, action, previous,
                    mode_conditioned=bool(mode_conditioned and current_mode_id == 2 and not recovery),
                    predictor=predictor, env=env,
                    online_effects=online_effects if args.online_effect_correction else None,
                    boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                    previous_observed_angle_delta=last_observed_delta,
                )[2]),
                "pose_cost_after": pose_cost(current, goal),
                "angle_after": angle_error(current[4], goal[4]),
                "observed_angle_delta": float(wrap_angle(current[4] - before[4])),
                "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
                "contact_distance_after": float(np.linalg.norm(current[:2] - current[2:4])),
                "boundary_distance_after": float(physical_contact_geometry(env, current)["distance"]),
                "boundary_mode_after": MODE_NAMES[current_mode_for_env(env, current)],
            })
            previous = action.astype(np.float64)
            last_observed_delta = float(wrap_angle(current[4] - before[4]))
            if args.online_effect_correction and not recovery_macro and current_mode_id == 2:
                key = online_effect_key(env, before, action)
                observed_target = transition_target_from_states(before, current)
                cache = online_effects.get(key)
                if cache is None:
                    online_effects[key] = {"mean": observed_target.tolist(), "count": 1}
                else:
                    count = int(cache["count"])
                    mean = np.asarray(cache["mean"], dtype=np.float64)
                    cache["mean"] = (mean + (observed_target - mean) / float(count + 1)).tolist()
                    cache["count"] = count + 1
            action_sign = action_moment_sign(
                before, action, env=env if contact_geometry_sign_feedback else None,
            )
            last_action_moment_sign = action_sign
            if calibration_probe:
                moment_to_angle_sign, updated = update_observed_sign_calibration(
                    moment_to_angle_sign, action_sign, last_observed_delta,
                    args.calibration_angle_threshold,
                )
                if updated:
                    calibration_successes += 1
                    if contact_geometry_sign_feedback and current_region_key is not None:
                        moment_to_angle_sign_by_region[current_region_key] = float(moment_to_angle_sign)
                    else:
                        moment_to_angle_sign_by_mode[current_mode_id] = float(moment_to_angle_sign)
                else:
                    if contact_geometry_sign_feedback and current_region_key is not None:
                        calibration_attempted_regions.add(current_region_key)
                    else:
                        calibration_attempted_modes.add(current_mode_id)
                trace[-1]["moment_to_angle_sign"] = moment_to_angle_sign
            consensus_sign, consensus_updated = update_mode_sign_consensus(
                sign_votes_by_mode, current_mode_id, action_sign,
                last_observed_delta, args.calibration_angle_threshold,
            )
            if consensus_updated and consensus_sign is not None:
                moment_to_angle_sign = consensus_sign
                if contact_geometry_sign_feedback and current_region_key is not None:
                    moment_to_angle_sign_by_region[current_region_key] = consensus_sign
                else:
                    moment_to_angle_sign_by_mode[current_mode_id] = consensus_sign
                trace[-1]["moment_to_angle_sign"] = consensus_sign
            if contact_geometry_sign_feedback and current_region_key is not None:
                region_sign, region_updated = update_contact_sign_consensus(
                    sign_votes_by_region, current_region_key, action_sign,
                    last_observed_delta, args.calibration_angle_threshold,
                )
                if region_updated and region_sign is not None:
                    moment_to_angle_sign_by_region[current_region_key] = region_sign
                    trace[-1]["moment_to_angle_sign"] = region_sign
            if recent_contact_sign_fallback and current_mode_id == 2:
                recent_sample = update_observed_sign_calibration(
                    None, action_sign, last_observed_delta,
                    args.calibration_angle_threshold,
                )[0]
                if recent_sample is not None:
                    recent_contact_sign = recent_sample
            if (
                moment_to_angle_sign is not None
                and abs(last_observed_delta) > args.calibration_angle_threshold
                and abs(float(np.sign(last_observed_delta))) > 0.5
                and abs(float(np.sign(wrap_angle(goal[4] - before[4])))) > 0.5
                and float(np.sign(last_observed_delta)) != float(np.sign(wrap_angle(goal[4] - before[4])))
            ):
                wrong_sign_events += 1
            if terminated or truncated:
                break
        records.append({
            "root_index": root_index,
            "final_xy": float(np.linalg.norm(current[2:4] - goal[2:4])),
            "final_angle": angle_error(current[4], goal[4]),
            "pose_success": bool(
                np.linalg.norm(current[2:4] - goal[2:4]) < 20.0
                and angle_error(current[4], goal[4]) < np.pi / 9.0
            ),
            "trace": trace,
            "moment_to_angle_sign": moment_to_angle_sign,
            "moment_to_angle_sign_by_mode": {
                str(key): value for key, value in moment_to_angle_sign_by_mode.items()
            },
            "moment_to_angle_sign_by_region": moment_to_angle_sign_by_region,
            "sign_votes_by_mode": {
                str(key): value for key, value in sign_votes_by_mode.items()
            },
            "sign_votes_by_region": sign_votes_by_region,
            "calibration_attempts": calibration_attempts,
            "calibration_successes": calibration_successes,
            "calibration_unresolved": bool(
                observed_sign_calibration
                and (
                    (
                        len(calibration_attempted_regions) > len(moment_to_angle_sign_by_region)
                        if contact_geometry_sign_feedback
                        else 2 in calibration_attempted_modes and 2 not in moment_to_angle_sign_by_mode
                    )
                )
            ),
            "wrong_sign_events": wrong_sign_events,
        })
    return {
        "metrics": {
            "pose_success": bootstrap_mean([float(row["pose_success"]) for row in records], 94201),
            "final_xy": bootstrap_mean([row["final_xy"] for row in records], 94202),
            "final_angle": bootstrap_mean([row["final_angle"] for row in records], 94203),
            "contact_loss_rate": bootstrap_mean([
                float(any(item["contact_distance_after"] > 80.0 for item in row["trace"]))
                for row in records
            ], 94204),
            "boundary_contact_loss_rate": bootstrap_mean([
                float(any(item["boundary_distance_after"] > 70.0 for item in row["trace"]))
                for row in records
            ], 94209),
            "recovery_steps": bootstrap_mean([
                float(sum(bool(item["contact_recovery"]) for item in row["trace"]))
                for row in records
            ], 94205),
            "calibration_success_rate": bootstrap_mean([
                float(row["calibration_successes"] > 0) for row in records
            ], 94206),
            "calibration_unresolved_rate": bootstrap_mean([
                float(row["calibration_unresolved"]) for row in records
            ], 94208),
            "wrong_sign_events": bootstrap_mean([
                float(row["wrong_sign_events"]) for row in records
            ], 94207),
        },
        "predictor": predictor,
        "contact_geometry_sign_feedback": bool(contact_geometry_sign_feedback),
        "recent_contact_sign_fallback": bool(recent_contact_sign_fallback),
        "online_effect_correction": bool(args.online_effect_correction),
        "online_effect_entries": [
            {"key": key, "count": int(value["count"])}
            for key, value in sorted(online_effects.items())
        ],
        "rows": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root-protocol", choices=("terminal_pose", "boundary_contact"), default="terminal_pose",
                        help="terminal_pose preserves the legacy center-radius roots; boundary_contact samples the actual T polygon boundary")
    parser.add_argument("--boundary-action-frame", action="store_true",
                        help="use the nearest polygon-edge normal for contact actions and recovery; boundary_contact enables it automatically")
    parser.add_argument("--legacy-center-action-frame", action="store_true",
                        help="ablation: keep center/vertex action frame while retaining the same root protocol")
    parser.add_argument("--train-roots", type=int, default=48)
    parser.add_argument("--held-out-roots", type=int, default=24)
    parser.add_argument("--sequence-steps", type=int, default=6)
    parser.add_argument("--pulse-steps", type=int, default=4)
    parser.add_argument("--closure-steps", type=int, default=36)
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--contact-chart-alpha", type=float, default=10.0,
                        help="fixed Ridge regularization for the physical contact chart")
    parser.add_argument("--lookahead", type=int, choices=(1, 2), default=2)
    parser.add_argument("--phase-aware-cost", action="store_true",
                        help="use translation-first then rotation-within-tube imagined scoring")
    parser.add_argument("--contact-recovery", action="store_true",
                        help="use an explicit state-only reacquisition branch outside the contact tube")
    parser.add_argument("--recovery-threshold", type=float, default=24.0,
                        help="polygon-boundary gap above which state-only reacquisition is allowed")
    parser.add_argument("--recovery-magnitude", type=float, default=0.06)
    parser.add_argument("--recovery-pulse-steps", type=int, default=1,
                        help="short state-only contact reacquisition pulse; kept separate from learned pulse length")
    parser.add_argument("--goal-directed-recovery", action="store_true",
                        help="reacquire a contact point on the side opposite the desired block translation")
    parser.add_argument("--recovery-contact-radius", type=float, default=36.0)
    parser.add_argument("--validated-route-recovery", action="store_true",
                        help="use the existing disengage-and-route contact macro as a privileged recovery ablation")
    parser.add_argument("--route-clearance", type=float, default=70.0)
    parser.add_argument("--route-contact-distance", type=float, default=42.0)
    parser.add_argument("--route-settle-steps", type=int, default=2)
    parser.add_argument("--observed-sign-feedback", action="store_true",
                        help="correct torque sign from the last committed observed angle increment")
    parser.add_argument("--observed-sign-calibration", action="store_true",
                        help="run one short contact probe to estimate geometric-moment to angle-sign mapping")
    parser.add_argument("--contact-geometry-sign-feedback", action="store_true",
                        help="condition online moment-to-angle sign on polygon edge and edge-position bin")
    parser.add_argument("--recent-contact-sign-fallback", action="store_true",
                        help="use the most recent informative contact sign when the current edge bin has no support")
    parser.add_argument("--strict-contact-sign-gate", action="store_true",
                        help="ablation: after observed local sign calibration, rank only candidates with the desired torque sign")
    parser.add_argument("--direct-observed-sign-reversal", action="store_true",
                        help="ablation: after a wrong observed angle pulse, reverse that pulse's actual moment sign directly")
    parser.add_argument("--strict-direction-gate", action="store_true",
                        help="ablation: hard-gate contact candidates whose learned angle-direction probability disagrees with the goal")
    parser.add_argument("--online-effect-correction", action="store_true",
                        help="use observed same-region/action-bin transitions for subsequent runtime ranking")
    parser.add_argument("--calibration-pulse-steps", type=int, default=1)
    parser.add_argument("--calibration-angle-threshold", type=float, default=1.0e-4)
    parser.add_argument("--feedback-contact-threshold", type=float, default=24.0,
                        help="polygon-boundary gap within which observed torque-sign feedback is trusted")
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument("--free-recovery-magnitude", type=float, default=0.5)
    parser.add_argument("--terminal-xy-gate", type=float, default=20.0)
    parser.add_argument("--terminal-angle-gate", type=float, default=float(np.pi / 9.0))
    parser.add_argument("--terminal-handoff", action="store_true",
                        help="enable the terminal geometric sign handoff as an explicit ablation")
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+", default=[0.06, 0.10, 0.16])
    parser.add_argument("--goal-translation-magnitudes", type=float, nargs="+", default=[],
                        help="optional task-directed tangent candidates; empty preserves the legacy bank")
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The corrected protocol couples real-boundary roots to the real-boundary
    # action frame.  The legacy terminal protocol remains center/vertex based
    # unless the ablation is explicitly requested.
    args.boundary_action_frame = bool(
        not args.legacy_center_action_frame
        and (args.boundary_action_frame or args.root_protocol == "boundary_contact")
    )
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        root_factory = make_boundary_contact_roots if args.root_protocol == "boundary_contact" else make_terminal_roots
        train_roots = root_factory(env, args.train_roots, args.seed + 1000)
        held_roots = root_factory(env, args.held_out_roots, args.seed + 2000)
        train_rows = collect_mode_balanced_rows(
            env, train_roots, tuple(args.tangent_magnitudes), args.inward_magnitude,
            args.free_recovery_magnitude, args.pulse_steps, args.sequence_steps,
            args.seed + 3000, tuple(args.goal_translation_magnitudes),
            args.boundary_action_frame,
        )
        held_rows = collect_mode_balanced_rows(
            env, held_roots, tuple(args.tangent_magnitudes), args.inward_magnitude,
            args.free_recovery_magnitude, args.pulse_steps, args.sequence_steps,
            args.seed + 4000, tuple(args.goal_translation_magnitudes),
            args.boundary_action_frame,
        )
        model = fit_mode_aware(
            train_rows, args.seed, args.trees, env=env,
            boundary_action_frame=args.boundary_action_frame,
            contact_chart_alpha=args.contact_chart_alpha,
        )
        held_features = np.asarray([
            lifted_row_features(row, env=env, boundary_action_frame=args.boundary_action_frame)
            for row in held_rows
        ])
        held_labels = np.asarray([successor_mode(row["before"], row["after"], env=env) for row in held_rows])
        mode_prediction = model["classifier"].predict(held_features)
        mode_accuracy = float(np.mean(mode_prediction == held_labels))
        ranking = rank_rows(env, held_rows, model, args)
        global_loop = closed_loop(
            env, held_roots, model, args, mode_conditioned=False,
            contact_recovery=args.contact_recovery,
            observed_sign_feedback=args.observed_sign_feedback,
            observed_sign_calibration=args.observed_sign_calibration,
            contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
            recent_contact_sign_fallback=args.recent_contact_sign_fallback,
            strict_contact_sign_gate=args.strict_contact_sign_gate,
            direct_observed_sign_reversal=args.direct_observed_sign_reversal,
        )
        mode_loop = closed_loop(
            env, held_roots, model, args, mode_conditioned=True,
            contact_recovery=args.contact_recovery,
            observed_sign_feedback=args.observed_sign_feedback,
            observed_sign_calibration=args.observed_sign_calibration,
            contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
            recent_contact_sign_fallback=args.recent_contact_sign_fallback,
            strict_contact_sign_gate=args.strict_contact_sign_gate,
            direct_observed_sign_reversal=args.direct_observed_sign_reversal,
        )
        chart_loop = closed_loop(
            env, held_roots, model, args, mode_conditioned=True,
            contact_recovery=args.contact_recovery,
            observed_sign_feedback=args.observed_sign_feedback,
            observed_sign_calibration=args.observed_sign_calibration,
            contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
            recent_contact_sign_fallback=args.recent_contact_sign_fallback,
            strict_contact_sign_gate=args.strict_contact_sign_gate,
            direct_observed_sign_reversal=args.direct_observed_sign_reversal,
            predictor="contact_chart",
        )
        goal_ranker_loop = closed_loop(
            env, held_roots, model, args, mode_conditioned=True,
            contact_recovery=args.contact_recovery,
            observed_sign_feedback=args.observed_sign_feedback,
            observed_sign_calibration=args.observed_sign_calibration,
            contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
            recent_contact_sign_fallback=args.recent_contact_sign_fallback,
            strict_contact_sign_gate=args.strict_contact_sign_gate,
            direct_observed_sign_reversal=args.direct_observed_sign_reversal,
            predictor="goal_ranker",
        )
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "root_protocol": args.root_protocol,
            "boundary_action_frame": bool(args.boundary_action_frame),
            "legacy_center_action_frame": bool(args.legacy_center_action_frame),
            "train_roots": args.train_roots, "held_out_roots": args.held_out_roots,
            "train_seed": args.seed + 1000, "held_out_seed": args.seed + 2000,
            "candidate_bank": "shared per root; declared action-frame inward+tangent pulses plus one explicit outward separation probe in contact mode",
            "runtime_successor_access": False,
            "contact_recovery": bool(args.contact_recovery),
            "recovery_threshold": args.recovery_threshold,
            "recovery_magnitude": args.recovery_magnitude,
            "recovery_pulse_steps": args.recovery_pulse_steps,
            "goal_directed_recovery": bool(args.goal_directed_recovery),
            "recovery_contact_radius": args.recovery_contact_radius,
            "validated_route_recovery": bool(args.validated_route_recovery),
            "route_clearance": args.route_clearance,
            "route_contact_distance": args.route_contact_distance,
            "route_settle_steps": args.route_settle_steps,
            "observed_sign_feedback": bool(args.observed_sign_feedback),
            "observed_sign_calibration": bool(args.observed_sign_calibration),
            "contact_geometry_sign_feedback": bool(args.contact_geometry_sign_feedback),
            "recent_contact_sign_fallback": bool(args.recent_contact_sign_fallback),
            "strict_contact_sign_gate": bool(args.strict_contact_sign_gate),
            "direct_observed_sign_reversal": bool(args.direct_observed_sign_reversal),
            "strict_direction_gate": bool(args.strict_direction_gate),
            "online_effect_correction": bool(args.online_effect_correction),
            "temporal_event_features": True,
            "previous_observed_angle_delta": True,
            "goal_conditioned_ranker": "training-only direct one-step pose-cost ablation",
            "calibration_pulse_steps": args.calibration_pulse_steps,
            "calibration_angle_threshold": args.calibration_angle_threshold,
            "terminal_handoff": bool(args.terminal_handoff),
            "terminal_xy_gate": args.terminal_xy_gate,
            "terminal_angle_gate": args.terminal_angle_gate,
            "feedback_contact_threshold": args.feedback_contact_threshold,
            "mode_labels": SUCCESSOR_NAMES,
            "current_mode_thresholds_px": {"boundary_contact": 24.0, "boundary_approach": 70.0},
            "successor_thresholds": {"boundary_start_free": 70.0, "boundary_separation": 70.0, "block_motion": 1.0},
            "lookahead": args.lookahead,
            "phase_aware_cost": bool(args.phase_aware_cost),
            "pulse_steps": args.pulse_steps,
            "free_recovery_magnitude": args.free_recovery_magnitude,
            "goal_translation_magnitudes": list(args.goal_translation_magnitudes),
        },
        "training": {"rows": len(train_rows), "roots": len(train_roots), "mode_counts": model["counts"]},
        "contact_chart": {
            "feature_dim": model["contact_chart"]["feature_dim"],
            "alpha": model["contact_chart"]["alpha"],
            "mode_counts": model["contact_chart"]["counts"],
            "training_only": True,
            "geometry_source": model["contact_chart"]["geometry_source"],
        },
        "held_out": {
            "rows": len(held_rows), "mode_accuracy": mode_accuracy,
            "mode_label_counts": {name: int(np.sum(held_labels == index)) for index, name in enumerate(SUCCESSOR_NAMES)},
        },
        "ranking": ranking,
        "closed_loop": {
            "global_one_step": global_loop,
            "mode_aware_lookahead": mode_loop,
            "contact_chart_lookahead": chart_loop,
            "goal_conditioned_ranker": goal_ranker_loop,
        },
        "sanity_checks": {
            "finite": bool(np.isfinite(held_features).all()),
            "shared_candidate_bank": True,
            "root_disjoint": True,
            "runtime_successor_access": False,
        },
        "claim_boundary": (
            "Privileged simulator-state mode-aware adapter audit. A positive result is "
            "not evidence for frozen RGB planning, causal latent physics, or universal "
            "world-model transfer; it only establishes the mode-aware interface on PushT. "
            "When contact_recovery is enabled, recovery is a state-only interface repair "
            "and must not be attributed to the learned ranker."
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output / "mode_aware_shared_candidate.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# PushT mode-aware shared-candidate audit", "",
        "| selector | ranking regret | pose success | final XY | final angle | contact loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, loop in (
        ("global_one_step", global_loop),
        ("mode_aware_lookahead", mode_loop),
        ("contact_chart_lookahead", chart_loop),
    ):
        lines.append(
            "| %s | %.4f | %.3f | %.3f | %.3f | %.3f |" % (
                name, ranking["metrics"][name]["estimate"], loop["metrics"]["pose_success"]["estimate"],
                loop["metrics"]["final_xy"]["estimate"], loop["metrics"]["final_angle"]["estimate"],
                loop["metrics"]["contact_loss_rate"]["estimate"],
            )
        )
    lines.append(
        "| goal_conditioned_ranker | n/a | %.3f | %.3f | %.3f | %.3f |" % (
            goal_ranker_loop["metrics"]["pose_success"]["estimate"],
            goal_ranker_loop["metrics"]["final_xy"]["estimate"],
            goal_ranker_loop["metrics"]["final_angle"]["estimate"],
            goal_ranker_loop["metrics"]["contact_loss_rate"]["estimate"],
        )
    )
    lines += ["", "Mode accuracy: %.4f" % mode_accuracy, "", report["claim_boundary"], ""]
    (output / "mode_aware_shared_candidate.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "mode_accuracy": mode_accuracy, "ranking": ranking["metrics"], "closed_loop": {"global": global_loop["metrics"], "mode_aware": mode_loop["metrics"], "goal_ranker": goal_ranker_loop["metrics"]}}, indent=2))


if __name__ == "__main__":
    main()
