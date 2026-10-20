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
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
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


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


class TorchRidgeMultiOutput:
    """GPU-backed standardized multi-output ridge for the contact chart."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, alpha: float, device: torch.device) -> None:
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
        y = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=device)
        self.device = device
        self.x_mean = x.mean(dim=0)
        self.x_scale = torch.clamp(x.std(dim=0), min=1.0e-3)
        self.y_mean = y.mean(dim=0)
        self.y_scale = torch.clamp(y.std(dim=0), min=1.0e-3)
        x_standard = (x - self.x_mean) / self.x_scale
        y_standard = (y - self.y_mean) / self.y_scale
        gram = x_standard.T @ x_standard
        ridge = float(alpha) * torch.eye(gram.shape[0], device=device, dtype=gram.dtype)
        self.weight = torch.linalg.solve(gram + ridge, x_standard.T @ y_standard)

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device)
        with torch.inference_mode():
            prediction = ((x - self.x_mean) / self.x_scale) @ self.weight
            prediction = prediction * self.y_scale + self.y_mean
        return prediction.detach().cpu().numpy()


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


def orientation_gate_key(env: gym.Env, state: np.ndarray) -> str:
    """Match the calibration-only orientation gate's edge/fraction key."""

    geometry = physical_contact_geometry(env, state)
    position_bin = int(np.floor(np.clip(float(geometry["edge_fraction"]), 0.0, np.nextafter(1.0, 0.0)) * 4.0))
    return f"{int(geometry['shape_index'])}:{int(geometry['edge_index'])}__bin{position_bin}"


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
    min_goal_support_alignment: float = 0.0,
    max_initial_angle_error: float = float("inf"),
) -> list[dict[str, np.ndarray]]:
    """Generate roots at actual T-boundary contact.

    A positive ``min_goal_support_alignment`` requires the pusher's outward
    contact normal to oppose the desired block translation.  This separates a
    physically valid contact root from a task-compatible local pushing root.
    """

    if not 0.0 <= float(min_goal_support_alignment) <= 1.0:
        raise ValueError("min_goal_support_alignment must be in [0, 1]")
    if float(max_initial_angle_error) < 0.0:
        raise ValueError("max_initial_angle_error must be non-negative")

    # A finite angle gate is an analysis protocol, not a post-hoc selection of
    # successful trajectories.  Oversample independent terminal roots so the
    # requested held-out count remains fixed after declaring the local
    # orientation-support domain.  The default preserves the legacy root
    # distribution exactly (one base root per requested root).
    finite_angle_gate = np.isfinite(float(max_initial_angle_error))
    base_count = int(count) if not finite_angle_gate else max(int(count) * 8, int(count) + 16)
    base_roots = make_terminal_roots(env, base_count, seed, root_protocol="terminal_pose")
    rng = np.random.default_rng(int(seed + 70000))
    shape_items = list(getattr(env.unwrapped.block, "shapes", ()))
    if not shape_items:
        raise ValueError("PushT block has no polygon shapes")
    roots: list[dict[str, np.ndarray]] = []
    for root in base_roots:
        state = np.asarray(root["state"], dtype=np.float64).copy()
        theta = float(state[4])
        initial_angle_error = angle_error(theta, float(np.asarray(root["goal"], dtype=np.float64)[4]))
        if finite_angle_gate and initial_angle_error > float(max_initial_angle_error):
            continue
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
            goal_delta = np.asarray(root["goal"], dtype=np.float64)[2:4] - state[2:4]
            goal_norm = float(np.linalg.norm(goal_delta))
            if goal_norm > 1.0e-8 and float(np.dot(outward, -goal_delta / goal_norm)) < float(min_goal_support_alignment):
                continue
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
        if len(roots) >= int(count):
            break
    if len(roots) < int(count):
        raise RuntimeError(
            "initial-angle gate left too few valid boundary roots: "
            f"requested={int(count)} available={len(roots)} "
            f"max_initial_angle_error={float(max_initial_angle_error):.6f}"
        )
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
    # The same edge/action chart is not transferable once the pusher has
    # materially separated from the actual boundary.  Keep this distance in
    # the exact cache key so a stale contact pulse cannot be reused in
    # approach/free space.
    return "%s:n%.3f:t%.3f:e%.2f:d%.1f" % (
        region, normal_component, tangent_component, edge_fraction,
        round(float(geometry["distance"]), 1),
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


def fit_orientation_effect_ranker(
    rows: list[dict[str, Any]], seed: int, trees: int,
    env: gym.Env | None = None, boundary_action_frame: bool = False,
) -> ExtraTreesRegressor:
    """Fit a train-root-only ranker for one-step angular effect.

    The ranker predicts absolute post-pulse angle error, rather than the
    entire pose cost.  It is an intervention for the identified failure mode:
    the generic transition model can have low aggregate error while misranking
    opposite-moment actions inside a contact chart.  Runtime still uses only
    observed state, goal, and candidate action; no successor is queried.
    """

    features = np.asarray([
        goal_ranker_features(row, env=env, boundary_action_frame=boundary_action_frame)
        for row in rows
    ], dtype=np.float64)
    targets = np.asarray([
        angle_error(
            float(np.asarray(row["after"], dtype=np.float64)[4]),
            float(np.asarray(row["goal"], dtype=np.float64)[4]),
        )
        for row in rows
    ], dtype=np.float64)
    model = ExtraTreesRegressor(
        n_estimators=int(trees), min_samples_leaf=2, max_features=0.8,
        random_state=int(seed), n_jobs=1,
    )
    model.fit(features, targets)
    return model


def fit_orientation_delta_ranker(
    rows: list[dict[str, Any]], seed: int, trees: int,
    env: gym.Env | None = None, boundary_action_frame: bool = False,
) -> ExtraTreesRegressor:
    """Fit a train-only predictor for signed one-pulse angular effect.

    Predicting ``after_angle - before_angle`` removes the current goal
    residual from the supervised target.  Runtime reintroduces the goal only
    when converting the predicted physical effect into an action score.
    """

    features = np.asarray([
        goal_ranker_features(row, env=env, boundary_action_frame=boundary_action_frame)
        for row in rows
    ], dtype=np.float64)
    targets = np.asarray([
        float(wrap_angle(
            float(np.asarray(row["after"], dtype=np.float64)[4])
            - float(np.asarray(row["before"], dtype=np.float64)[4])
        ))
        for row in rows
    ], dtype=np.float64)
    model = ExtraTreesRegressor(
        n_estimators=int(trees), min_samples_leaf=2, max_features=0.8,
        random_state=int(seed), n_jobs=1,
    )
    model.fit(features, targets)
    return model


def orientation_pair_ranker_features(
    state: np.ndarray, goal: np.ndarray,
    first_action: np.ndarray, second_action: np.ndarray,
    previous_action: np.ndarray | None = None,
    previous_observed_angle_delta: float = 0.0,
    env: gym.Env | None = None,
    boundary_action_frame: bool = False,
) -> np.ndarray:
    """Build features for an ordered mirrored-action angular contrast."""

    first = goal_ranker_features(
        state, goal=goal, action=first_action, previous_action=previous_action,
        previous_observed_angle_delta=previous_observed_angle_delta, env=env,
        boundary_action_frame=boundary_action_frame,
    )
    second = goal_ranker_features(
        state, goal=goal, action=second_action, previous_action=previous_action,
        previous_observed_angle_delta=previous_observed_angle_delta, env=env,
        boundary_action_frame=boundary_action_frame,
    )
    first_value = np.asarray(first_action, dtype=np.float64)
    second_value = np.asarray(second_action, dtype=np.float64)
    interaction = np.asarray([
        *first_value, *second_value,
        *(first_value - second_value), *(first_value + second_value),
        float(np.linalg.norm(first_value)), float(np.linalg.norm(second_value)),
        float(np.dot(first_value, second_value)),
    ], dtype=np.float64)
    result = np.concatenate([first, second, interaction])
    if not np.isfinite(result).all():
        raise ValueError("orientation pair ranker features produced non-finite values")
    return result


def fit_orientation_pair_ranker(
    rows: list[dict[str, Any]], seed: int, trees: int,
    env: gym.Env | None = None, boundary_action_frame: bool = False,
) -> ExtraTreesRegressor:
    """Fit a training-root-only contrast model for mirrored pulse effects.

    Rows are paired within the same candidate group using the fixed action-bank
    order ``(0,1), (2,3), ...``.  The target is the ordered difference between
    the two signed angle increments.  This cancels the common component of a
    paired pulse and makes the within-contact direction signal explicit.
    """

    grouped: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for row in rows:
        group_key = (int(row["root_id"]), int(row["action_id"]) // 2)
        grouped.setdefault(group_key, {})[int(row["action_id"]) % 2] = row
    features: list[np.ndarray] = []
    targets: list[float] = []
    for pair in grouped.values():
        if 0 not in pair or 1 not in pair:
            continue
        first = pair[0]
        second = pair[1]
        before = np.asarray(first["before"], dtype=np.float64)
        first_after = np.asarray(first["after"], dtype=np.float64)
        second_after = np.asarray(second["after"], dtype=np.float64)
        first_delta = float(wrap_angle(first_after[4] - before[4]))
        second_delta = float(wrap_angle(second_after[4] - before[4]))
        features.append(orientation_pair_ranker_features(
            before, np.asarray(first["goal"], dtype=np.float64),
            np.asarray(first["action"], dtype=np.float64),
            np.asarray(second["action"], dtype=np.float64),
            previous_action=np.asarray(first.get("previous_action", np.zeros(2)), dtype=np.float64),
            previous_observed_angle_delta=float(first.get("previous_observed_angle_delta", 0.0)),
            env=env, boundary_action_frame=boundary_action_frame,
        ))
        targets.append(first_delta - second_delta)
    if len(features) < 2:
        raise ValueError("orientation pair ranker requires at least two complete action pairs")
    model = ExtraTreesRegressor(
        n_estimators=int(trees), min_samples_leaf=2, max_features=0.8,
        random_state=int(seed), n_jobs=1,
    )
    model.fit(np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64))
    return model


def macro_ranker_features(
    initial_state: np.ndarray, goal: np.ndarray, first_action: np.ndarray,
    first_state: np.ndarray, second_action: np.ndarray,
    previous_action: np.ndarray | None = None, env: gym.Env | None = None,
    boundary_action_frame: bool = False,
) -> np.ndarray:
    """Features for a training-only joint two-pulse pose-cost ranker.

    At deployment ``first_state`` is the learned one-step imagined state for
    first-action selection and the freshly observed state for second-action
    selection.  The feature contract is identical in both cases; no successor
    query is needed at runtime.
    """

    first_features = goal_ranker_features(
        initial_state, goal=goal, action=first_action,
        previous_action=previous_action, env=env,
        boundary_action_frame=boundary_action_frame,
    )
    second_features = goal_ranker_features(
        first_state, goal=goal, action=second_action,
        previous_action=first_action, env=env,
        boundary_action_frame=boundary_action_frame,
    )
    first_action = np.asarray(first_action, dtype=np.float64)
    second_action = np.asarray(second_action, dtype=np.float64)
    interaction = np.asarray([
        *first_action, *second_action,
        float(np.dot(first_action, second_action)),
        float(np.linalg.norm(first_action - second_action)),
        float(np.linalg.norm(np.asarray(first_state)[2:4] - np.asarray(goal)[2:4])) / 100.0,
        float(angle_error(np.asarray(first_state)[4], np.asarray(goal)[4])) / np.pi,
    ], dtype=np.float64)
    result = np.concatenate([first_features, second_features, interaction])
    if not np.isfinite(result).all():
        raise ValueError("macro ranker features produced non-finite values")
    return result


def collect_macro_ranker_rows(
    env: gym.Env, roots: list[dict[str, np.ndarray]],
    tangent_magnitudes: tuple[float, ...], inward_magnitude: float,
    free_recovery_magnitude: float, pulse_steps: int,
    goal_translation_magnitudes: tuple[float, ...],
    boundary_action_frame: bool,
    contact_normal_magnitudes: tuple[float, ...] = (),
    model: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Collect root-disjoint pairs under the deployment sequence context.

    The first-action scorer is deployed from a model-imagined intermediate
    state, while the second-action scorer is deployed after a fresh observed
    state.  When ``model`` is supplied, candidate generation and ranker
    features for the first-action context therefore use the frozen train-only
    predictor's imagined state.  Real successors remain labels only; they are
    never exposed to the held-out controller.
    """

    rows: list[dict[str, Any]] = []
    for root_index, root in enumerate(roots):
        initial = np.asarray(root["state"], dtype=np.float64).copy()
        goal = np.asarray(root["goal"], dtype=np.float64).copy()
        first_actions = candidate_actions(
            env, initial, tangent_magnitudes, inward_magnitude,
            free_recovery_magnitude, goal=goal,
            goal_translation_magnitudes=goal_translation_magnitudes,
            boundary_action_frame=boundary_action_frame,
            contact_normal_magnitudes=contact_normal_magnitudes,
        )
        for first_index, first_action in enumerate(first_actions):
            first_result = execute_pulse(env, initial, goal, first_action, pulse_steps)
            first_state = np.asarray(first_result["after"], dtype=np.float64).copy()
            if model is None:
                ranker_first_state = first_state.copy()
            else:
                predicted_target, _, _ = predict_transition(
                    model, initial, np.asarray(first_action, dtype=np.float64),
                    np.zeros(2, dtype=np.float64), mode_conditioned=True,
                    predictor="tree", env=env,
                    boundary_action_frame=boundary_action_frame,
                )
                ranker_first_state = predicted_state(initial, predicted_target)
            second_actions = candidate_actions(
                env, ranker_first_state, tangent_magnitudes, inward_magnitude,
                free_recovery_magnitude, goal=goal,
                goal_translation_magnitudes=goal_translation_magnitudes,
                boundary_action_frame=boundary_action_frame,
                contact_normal_magnitudes=contact_normal_magnitudes,
            )
            for second_index, second_action in enumerate(second_actions):
                second_result = execute_pulse(env, first_state, goal, second_action, pulse_steps)
                rows.append({
                    "root_id": int(root_index),
                    "initial": initial.copy(), "goal": goal.copy(),
                    "first_action": np.asarray(first_action, dtype=np.float64),
                    "first_state": first_state.copy(),
                    "ranker_first_state": ranker_first_state.copy(),
                    "second_action": np.asarray(second_action, dtype=np.float64),
                    "final": np.asarray(second_result["after"], dtype=np.float64).copy(),
                    "first_index": int(first_index), "second_index": int(second_index),
                })
    return rows


def fit_macro_ranker(
    rows: list[dict[str, Any]], seed: int, trees: int,
    env: gym.Env, boundary_action_frame: bool,
) -> ExtraTreesRegressor:
    """Fit the joint ranker on training-root two-pulse outcomes only."""

    features = np.asarray([
        macro_ranker_features(
            row["initial"], row["goal"], row["first_action"],
            row.get("ranker_first_state", row["first_state"]),
            row["second_action"], env=env,
            boundary_action_frame=boundary_action_frame,
        ) for row in rows
    ], dtype=np.float64)
    targets = np.asarray([
        pose_cost(row["final"], row["goal"]) for row in rows
    ], dtype=np.float64)
    model = ExtraTreesRegressor(
        n_estimators=int(trees), min_samples_leaf=2, max_features=0.8,
        random_state=int(seed), n_jobs=1,
    )
    model.fit(features, targets)
    return model


def select_macro_first_action(
    env: gym.Env, model: dict[str, Any], macro_ranker: ExtraTreesRegressor,
    state: np.ndarray, goal: np.ndarray, actions: np.ndarray,
    previous_action: np.ndarray, args: argparse.Namespace,
    risk_penalty: float = 0.0,
) -> int:
    """Choose the first pulse by minimizing learned joint two-pulse cost."""

    predicted_first, _, _ = predict_transition_candidates(
        model, state, actions, previous_action,
        mode_conditioned=True, predictor="tree", env=env,
        boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
    )
    scores: list[float] = []
    for first_action, first_target in zip(actions, predicted_first):
        imagined_first = predicted_state(state, first_target)
        second_actions = candidate_actions(
            env, imagined_first, args.tangent_magnitudes, args.inward_magnitude,
            goal=goal, goal_translation_magnitudes=args.goal_translation_magnitudes,
            boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            contact_normal_magnitudes=tuple(getattr(args, "contact_normal_magnitudes", ())),
        )
        features = np.asarray([
            macro_ranker_features(
                state, goal, first_action, imagined_first, second_action,
                previous_action=previous_action, env=env,
                boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            ) for second_action in second_actions
        ], dtype=np.float64)
        predictions = np.asarray([
            estimator.predict(features) for estimator in macro_ranker.estimators_
        ], dtype=np.float64)
        mean_scores = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        scores.append(float(np.min(mean_scores + float(risk_penalty) * uncertainty)))
    return int(np.argmin(np.asarray(scores, dtype=np.float64)))


def select_macro_second_action(
    env: gym.Env, macro_ranker: ExtraTreesRegressor, initial_state: np.ndarray,
    goal: np.ndarray, first_action: np.ndarray, current_state: np.ndarray,
    actions: np.ndarray, previous_action: np.ndarray, args: argparse.Namespace,
    risk_penalty: float = 0.0,
) -> int:
    """Choose the second pulse after its fresh runtime observation."""

    features = np.asarray([
        macro_ranker_features(
            initial_state, goal, first_action, current_state, action,
            previous_action=previous_action, env=env,
            boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
        ) for action in actions
    ], dtype=np.float64)
    predictions = np.asarray([
        estimator.predict(features) for estimator in macro_ranker.estimators_
    ], dtype=np.float64)
    scores = np.mean(predictions, axis=0) + float(risk_penalty) * np.std(predictions, axis=0)
    return int(np.argmin(scores))


def fit_contact_chart(
    rows: list[dict[str, Any]], seed: int, alpha: float = 10.0,
    env: gym.Env | None = None, device: torch.device | None = None,
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
        if device is not None and device.type == "cuda":
            return TorchRidgeMultiOutput(features, targets, alpha, device)
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
    device: torch.device | None = None,
    orientation_ranker_rows: list[dict[str, Any]] | None = None,
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
        "orientation_effect_ranker": fit_orientation_effect_ranker(
            rows if orientation_ranker_rows is None else orientation_ranker_rows,
            seed + 220, trees, env=env,
            boundary_action_frame=boundary_action_frame,
        ),
        "orientation_delta_ranker": fit_orientation_delta_ranker(
            rows, seed + 240, trees, env=env,
            boundary_action_frame=boundary_action_frame,
        ),
        "contact_chart": fit_contact_chart(
            rows, seed + 140, alpha=contact_chart_alpha, env=env, device=device,
        ),
        "by_mode": by_mode,
        "by_successor": by_successor,
        "by_mode_successor": by_mode_successor,
        "classifier": classifier,
        "direction_classifier": direction_classifier,
        "boundary_action_frame": bool(boundary_action_frame),
        "counts": {**counts, "successor": successor_counts},
    }


def effect_gate_key(env: gym.Env, state: np.ndarray, action: np.ndarray) -> str:
    """Coarse deployment support bin for local action-effect evidence.

    The gate deliberately does not key on the exact floating-point contact
    point.  It groups observations by polygon edge, geometric moment sign and
    action magnitude, while leaving edge-position variation inside the
    empirical uncertainty estimate.  This is a support/uncertainty diagnostic,
    not a hidden simulator label.
    """

    geometry = physical_contact_geometry(env, state)
    normal = boundary_inward_normal(env, state)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    action_value = np.asarray(action, dtype=np.float64)
    normal_component = float(np.dot(action_value, normal))
    tangent_component = float(np.dot(action_value, tangent))
    moment_sign = int(np.sign(action_moment_sign(state, action, env=env)))
    return "%d:%d:s%d:m%.2f:n%.2f:t%d" % (
        int(geometry["shape_index"]), int(geometry["edge_index"]),
        moment_sign, round(float(np.linalg.norm(action_value)), 2),
        round(normal_component, 2), int(np.sign(tangent_component)),
    )


def online_effect_descriptor(env: gym.Env, state: np.ndarray, action: np.ndarray) -> dict[str, float | int | str]:
    """Return the fixed local coordinates used by the online effect cache.

    The previous cache treated a floating-point contact/action bin as the
    only reusable unit.  That is too brittle for a moving contact point: two
    consecutive pulses can be physically local while landing in different
    bins.  This descriptor is deliberately small and declared before the
    held-out run.  It contains only current contact geometry and the executed
    action, never a successor label.
    """

    geometry = physical_contact_geometry(env, state)
    inward = boundary_inward_normal(env, state)
    tangent = np.asarray([-inward[1], inward[0]], dtype=np.float64)
    action_value = np.asarray(action, dtype=np.float64)
    return {
        "region": "%d:%d" % (int(geometry["shape_index"]), int(geometry["edge_index"])),
        "edge_fraction": float(geometry["edge_fraction"]),
        "normal_component": float(np.dot(action_value, inward)),
        "tangent_component": float(np.dot(action_value, tangent)),
        "action_norm": float(np.linalg.norm(action_value)),
        "moment_sign": int(np.sign(action_moment_sign(state, action, env=env))),
        "boundary_distance": float(geometry["distance"]),
    }


def _online_effect_descriptor_distance(
    query: dict[str, float | int | str], entry: dict[str, Any],
    *, mirror_moment: bool = False,
) -> float | None:
    """Compare two local effect descriptors in a fixed dimensionless chart."""

    descriptor = entry.get("descriptor")
    if not isinstance(descriptor, dict) or descriptor.get("region") != query.get("region"):
        return None
    # The scales are frozen from the declared action bank, not fitted on the
    # held-out roots.  Edge position is intentionally softer than action
    # direction because nearby points on one polygon edge share a local chart.
    query_tangent = float(query["tangent_component"])
    entry_tangent = float(descriptor.get("tangent_component", 0.0))
    tangent_difference = (
        abs(entry_tangent) - abs(query_tangent)
        if mirror_moment else entry_tangent - query_tangent
    )
    differences = np.asarray([
        (float(descriptor.get("edge_fraction", 0.0)) - float(query["edge_fraction"])) / 0.25,
        (float(descriptor.get("normal_component", 0.0)) - float(query["normal_component"])) / 0.08,
        tangent_difference / 0.16,
        (float(descriptor.get("action_norm", 0.0)) - float(query["action_norm"])) / 0.16,
        (float(descriptor.get("boundary_distance", 1.0e3)) - float(query["boundary_distance"])) / 8.0,
    ], dtype=np.float64)
    distance = float(np.linalg.norm(differences))
    return distance if np.isfinite(distance) else None


def local_observed_effect(
    env: gym.Env, state: np.ndarray, action: np.ndarray,
    online_local_effects: dict[str, dict[str, Any]], *, max_neighbors: int = 3,
    max_radius: float = 1.75, min_informative_count: int = 0,
    min_effect_angle: float = 0.0, require_paired: bool = False,
) -> dict[str, Any] | None:
    """Aggregate nearby observed effects with a fail-closed support radius."""

    query = online_effect_descriptor(env, state, action)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for key, entry in online_local_effects.items():
        if int(entry.get("count", 0)) < 1:
            continue
        # A single zero-response pulse is not action-effect evidence.  It is
        # common when a pusher is visually near the block but still sticking
        # or sliding outside the locally excited contact interface.  Reusing
        # it made the adaptive controller repeat a stale action indefinitely.
        if int(entry.get("informative_count", 0)) < int(min_informative_count):
            continue
        if abs(float(entry.get("mean_angle_abs", 0.0))) < float(min_effect_angle):
            continue
        entry_moment = int(entry.get("descriptor", {}).get("moment_sign", 0))
        mirror_moment = entry_moment != 0 and entry_moment != int(query["moment_sign"])
        distance = _online_effect_descriptor_distance(
            query, entry, mirror_moment=mirror_moment,
        )
        if distance is not None and distance <= float(max_radius):
            candidates.append((distance, {"key": key, **entry}))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1]["key"])))
    same = [item for item in candidates if int(item[1].get("descriptor", {}).get("moment_sign", 0)) == int(query["moment_sign"])]
    opposite = [item for item in candidates if int(item[1].get("descriptor", {}).get("moment_sign", 0)) == -int(query["moment_sign"])]
    if not same:
        return None
    if require_paired and not opposite:
        # The online controller is only allowed to reuse a local chart after
        # a matched +/- intervention.  Without the contrast, passive drift
        # and contact release are confounded with the action effect.
        return None

    def aggregate(selected: list[tuple[float, dict[str, Any]]]) -> tuple[np.ndarray, float, float]:
        selected = selected[: max(1, int(max_neighbors))]
        weights = np.asarray([1.0 / (1.0 + distance) for distance, _ in selected], dtype=np.float64)
        weights /= max(float(weights.sum()), 1.0e-12)
        targets = np.asarray([np.asarray(item["mean"], dtype=np.float64) for _, item in selected])
        target_value = np.sum(targets * weights[:, None], axis=0)
        gains_value = float(np.sum(np.asarray([float(item.get("task_gain", 0.0)) for _, item in selected]) * weights))
        errors_value = float(np.sum(np.asarray([float(item.get("last_error", 0.0)) for _, item in selected]) * weights))
        return target_value, gains_value, errors_value

    target, gain, error = aggregate(same)
    paired = bool(opposite)
    direction_target = target.copy()
    opposite_target = None
    if paired:
        opposite_target, _, _ = aggregate(opposite)
        # A matched opposite pulse estimates the action-dependent component;
        # the absolute pulse can contain common drift from a moving contact.
        # Replace only the angular component used for direction control and
        # keep the same-sign pulse's translation effect.
        contrast = 0.5 * (target - opposite_target)
        direction_target = contrast
        if abs(float(contrast[0])) > 1.0e-4:
            target[0] = contrast[0]
    return {
        "target": target.tolist(),
        "direction_target": direction_target.tolist(),
        "paired": paired,
        "opposite_target": None if opposite_target is None else opposite_target.tolist(),
        "task_gain": gain,
        "last_error": error,
        "neighbor_count": int(min(len(same), max(1, int(max_neighbors)))),
        "opposite_neighbor_count": int(min(len(opposite), max(1, int(max_neighbors)))),
        "neighbor_distance": float(same[0][0]),
        "neighbor_keys": [str(item[1]["key"]) for item in same[: max(1, int(max_neighbors))]],
        "descriptor": query,
    }


def normalized_effect_error(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Task-relevant local successor discrepancy used by the gate report."""

    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    return float(
        abs(float(wrap_angle(predicted[0] - observed[0]))) / 0.20
        + np.linalg.norm(predicted[1:3] - observed[1:3]) / 20.0
    )


def protocol_contact_loss(trace: list[dict[str, Any]], root_protocol: str) -> bool:
    """Evaluate contact loss in the geometry of the declared root protocol."""

    if root_protocol == "boundary_contact":
        return bool(any(
            float(item.get("boundary_distance_after", 0.0)) > 70.0
            for item in trace
        ))
    return bool(any(
        float(item.get("contact_distance_after", 0.0)) > 80.0
        for item in trace
    ))


def relocation_outward_direction(
    state: np.ndarray, goal: np.ndarray, attempt: int,
) -> np.ndarray:
    """Choose a deterministic alternative support side after local stagnation."""

    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    goal_delta = goal[2:4] - state[2:4]
    goal_norm = float(np.linalg.norm(goal_delta))
    if goal_norm <= 1.0e-8:
        base = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        # The pusher should approach from behind the desired translation.
        base = -(goal_delta / goal_norm)
    offsets = (0.0, np.pi / 2.0, -np.pi / 2.0, np.pi, np.pi / 3.0, -np.pi / 3.0)
    angle = float(np.arctan2(base[1], base[0]) + offsets[int(attempt) % len(offsets)])
    return np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)


def relocate_to_contact_with_boundary_detach(
    env: gym.Env, state: np.ndarray, goal: np.ndarray,
    outward: np.ndarray, args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Relocate after leaving the *actual* polygon boundary first.

    The legacy route helper disengaged along the block-centre vector.  For a
    T-shaped object that vector can point through a different occupied edge.
    The corrected stage first moves along the observed nearest-edge outward
    normal, then invokes the validated obstacle-aware route from a separated
    state.
    """

    current = np.asarray(state, dtype=np.float64).copy()
    before = current.copy()
    env.reset(seed=0, options={"state": current.copy(), "goal_state": np.asarray(goal).copy()})
    detach_steps = 0
    terminated = False
    for _ in range(int(args.relocation_detach_steps)):
        geometry = physical_contact_geometry(env, current)
        if float(geometry["distance"]) > float(args.relocation_clearance):
            break
        detach_action = float(args.relocation_detach_magnitude) * np.asarray(
            geometry["normal_world"], dtype=np.float64,
        )
        observation, _, terminated, truncated, _ = env.step(detach_action.astype(np.float32))
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        detach_steps += 1
        if terminated or truncated:
            break
    if not terminated:
        route_args = SimpleNamespace(
            clearance=float(args.relocation_clearance),
            contact_distance=float(args.relocation_contact_distance),
            settle_steps=int(args.relocation_settle_steps),
        )
        routed, route_info = route_to_contact(
            env, current, goal, -np.asarray(outward, dtype=np.float64), route_args,
        )
        current = np.asarray(routed, dtype=np.float64).copy()
    else:
        route_info = {"route_steps": 0, "contact_distance": np.nan, "terminated": True}
    return current, {
        **route_info,
        "detach_steps": int(detach_steps),
        "block_displacement": float(np.linalg.norm(current[2:4] - before[2:4])),
        "detachment_geometry": "nearest_polygon_edge_outward_normal",
    }


def fit_effect_gate_statistics(
    rows: list[dict[str, Any]], model: dict[str, Any], env: gym.Env,
    boundary_action_frame: bool,
) -> dict[str, dict[str, Any]]:
    """Estimate local support, sign consistency and residual uncertainty.

    All quantities are fitted on training roots only.  The successor is used
    here solely as an offline training label.  Runtime gating sees only the
    current state, candidate action and the frozen table.
    """

    grouped: dict[str, list[tuple[float, float, float]]] = {}
    rows_by_root: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        rows_by_root.setdefault(int(row.get("root_id", index)), []).append(row)
    for root_rows in rows_by_root.values():
        before = np.asarray(root_rows[0]["before"], dtype=np.float64)
        previous = np.asarray(root_rows[0].get("previous_action", np.zeros(2)), dtype=np.float64)
        actions = np.asarray([row["action"] for row in root_rows], dtype=np.float64)
        predicted_targets, _, _ = predict_transition_candidates(
            model, before, actions, previous,
            mode_conditioned=True, predictor="tree", env=env,
            boundary_action_frame=boundary_action_frame,
        )
        for row, predicted in zip(root_rows, predicted_targets):
            action = np.asarray(row["action"], dtype=np.float64)
            target = transition_target(row)
            key = effect_gate_key(env, before, action)
            angle_effect = float(target[0])
            moment_sign = float(np.sign(action_moment_sign(before, action, env=env)))
            sign_sample = float(np.sign(angle_effect) * moment_sign) if abs(angle_effect) > 1.0e-8 else 0.0
            residual = normalized_effect_error(predicted, target)
            grouped.setdefault(key, []).append((angle_effect, sign_sample, residual))
    statistics: dict[str, dict[str, Any]] = {}
    for key, values in grouped.items():
        angles = np.asarray([value[0] for value in values], dtype=np.float64)
        informative = np.asarray([value[1] for value in values if abs(value[1]) > 0.5], dtype=np.float64)
        positive = int(np.sum(informative > 0.0))
        negative = int(np.sum(informative < 0.0))
        sign_consistency = float(max(positive, negative) / max(len(informative), 1))
        statistics[key] = {
            "support": int(len(values)),
            "informative_support": int(len(informative)),
            "angle_mean": float(np.mean(angles)),
            "angle_std": float(np.std(angles)),
            "angle_nonzero_rate": float(np.mean(np.abs(angles) > 1.0e-8)),
            "sign_consistency": sign_consistency,
            "sign_majority": int(1 if positive >= negative else -1) if len(informative) else 0,
            "train_predicted_observed_rmse": float(np.sqrt(np.mean(np.square([value[2] for value in values])))),
        }
    return statistics


def effect_gate_decision(
    model: dict[str, Any], env: gym.Env, state: np.ndarray, action: np.ndarray,
    goal: np.ndarray, args: argparse.Namespace,
) -> dict[str, Any]:
    """Return a fail-closed decision for one learned contact action."""

    key = effect_gate_key(env, state, action)
    stats = model.get("effect_gate", {}).get(key)
    reason = "ok"
    valid = True
    if stats is None:
        valid = False
        reason = "no_support_bin"
    else:
        desired_sign = float(np.sign(wrap_angle(float(goal[4]) - float(state[4]))))
        if int(stats["support"]) < int(args.min_effect_support):
            valid, reason = False, "insufficient_support"
        elif int(stats["informative_support"]) < int(args.min_informative_support):
            valid, reason = False, "insufficient_informative_support"
        elif float(stats["sign_consistency"]) < float(args.min_effect_sign_consistency):
            valid, reason = False, "inconsistent_effect_sign"
        elif float(stats["angle_std"]) > float(args.max_effect_angle_std):
            valid, reason = False, "high_angle_uncertainty"
        elif float(stats["train_predicted_observed_rmse"]) > float(args.max_train_effect_error):
            valid, reason = False, "high_train_effect_error"
        elif desired_sign != 0.0 and float(np.sign(stats["angle_mean"])) != desired_sign:
            valid, reason = False, "wrong_mean_angle_direction"
    return {
        "valid": bool(valid), "reason": reason, "key": key,
        "stats": None if stats is None else dict(stats),
    }


def choose_online_effect_action(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, actions: np.ndarray,
    online_local_effects: dict[str, dict[str, Any]], *, max_neighbors: int = 3,
    max_radius: float = 1.75, task_gain_weight: float = 0.50,
    min_informative_count: int = 0, min_effect_angle: float = 0.0,
    require_paired: bool = False,
) -> tuple[int | None, dict[str, Any] | None]:
    """Rank candidates with a nearby observed effect estimate.

    This is the exploitation half of a short-pulse system-identification
    controller.  Reuse is local in the declared polygon-edge/action chart,
    rather than requiring an exact floating-point bin.  Otherwise the caller
    must issue a short probe or fail closed.
    """

    supported: list[tuple[int, float, dict[str, Any]]] = []
    desired_sign = float(np.sign(wrap_angle(float(goal[4]) - float(state[4]))))
    for index, action in enumerate(np.asarray(actions, dtype=np.float64)):
        entry = local_observed_effect(
            env, state, action, online_local_effects,
            max_neighbors=max_neighbors, max_radius=max_radius,
            min_informative_count=min_informative_count,
            min_effect_angle=min_effect_angle,
            require_paired=require_paired,
        )
        if entry is None:
            continue
        target = np.asarray(entry["target"], dtype=np.float64)
        # An observed local effect is useful only if it points toward the
        # current task direction.  Never let online identification turn a
        # one-shot probe that rotated the wrong way into a committed policy.
        direction_target = np.asarray(entry.get("direction_target", target), dtype=np.float64)
        if desired_sign != 0.0 and abs(float(direction_target[0])) > 1.0e-4:
            if float(np.sign(direction_target[0])) != desired_sign:
                continue
        imagined = predicted_state(state, target)
        score = planning_pose_cost(state, imagined, goal, phase_aware=True)
        # Online identification may be locally valid but still point away
        # from the task target.  Do not repeatedly commit such an estimate
        # merely because it is the only observed candidate in the bin.
        if score > planning_pose_cost(state, state, goal, phase_aware=True) + 0.05:
            continue
        # A locally observed task-cost gain is stronger evidence than a
        # model-only imagined score, but it is bounded so one probe cannot
        # dominate every later decision.
        score -= float(task_gain_weight) * float(np.clip(entry.get("task_gain", 0.0), -2.0, 2.0))
        score += 0.5 * float(entry.get("last_error", 0.0))
        supported.append((int(index), float(score), entry))
    if not supported:
        return None, None
    selected, _, info = min(supported, key=lambda value: value[1])
    return selected, info


def choose_local_jacobian_action(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, actions: np.ndarray,
    online_local_effects: dict[str, dict[str, Any]], *, max_radius: float = 1.75,
    min_samples: int = 2, require_paired: bool = True,
) -> tuple[int | None, dict[str, Any] | None]:
    """Use only prior observed pulses to fit a local action-effect Jacobian.

    The fit has no simulator successor access.  It is deliberately
    fail-closed: entries must come from the same polygon region, contain
    informative effects, span both local action coordinates, and (by default)
    include both moment signs.  This addresses the dense-bank failure where
    exact-bin reuse had no support even though paired probes had identified a
    local signed response.
    """

    query = online_effect_descriptor(env, state, np.asarray(actions[0], dtype=np.float64))
    region = str(query["region"])
    inward = boundary_inward_normal(env, state)
    tangent = np.asarray([-inward[1], inward[0]], dtype=np.float64)
    local_rows: list[tuple[np.ndarray, np.ndarray, int]] = []
    for entry in online_local_effects.values():
        descriptor = entry.get("descriptor")
        if not isinstance(descriptor, dict) or str(descriptor.get("region")) != region:
            continue
        if int(entry.get("informative_count", 0)) < 1:
            continue
        action = np.asarray(entry.get("action", []), dtype=np.float64)
        if action.shape != (2,):
            continue
        distance = _online_effect_descriptor_distance(query, entry)
        if distance is None or distance > float(max_radius):
            continue
        local_action = np.asarray([
            float(np.dot(action, inward)), float(np.dot(action, tangent)),
        ], dtype=np.float64)
        effect = np.asarray(entry.get("mean", []), dtype=np.float64)
        if local_action.shape != (2,) or effect.shape != (3,) or not np.all(np.isfinite(effect)):
            continue
        local_rows.append((local_action, effect, int(np.sign(descriptor.get("moment_sign", 0)))))
    if len(local_rows) < int(min_samples):
        return None, None
    signs = {row[2] for row in local_rows if row[2] != 0}
    if require_paired and not ({-1, 1} <= signs):
        return None, None
    X = np.asarray([row[0] for row in local_rows], dtype=np.float64)
    Y = np.asarray([row[1] for row in local_rows], dtype=np.float64)
    if np.linalg.matrix_rank(X) < 2:
        return None, None
    coefficients, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    residual = float(np.sqrt(np.mean(np.square(X @ coefficients - Y))))
    candidate_rows: list[tuple[int, float, dict[str, Any]]] = []
    desired_sign = float(np.sign(wrap_angle(float(goal[4]) - float(state[4]))))
    for index, action in enumerate(np.asarray(actions, dtype=np.float64)):
        local_action = np.asarray([
            float(np.dot(action, inward)), float(np.dot(action, tangent)),
        ], dtype=np.float64)
        target = local_action @ coefficients
        if desired_sign != 0.0 and abs(float(target[0])) > 1.0e-4:
            if float(np.sign(target[0])) != desired_sign:
                continue
        imagined = predicted_state(state, target)
        score = planning_pose_cost(state, imagined, goal, phase_aware=True)
        score += 0.5 * residual
        candidate_rows.append((int(index), float(score), {
            "target": target.tolist(), "jacobian_residual": residual,
            "sample_count": len(local_rows), "rank": int(np.linalg.matrix_rank(X)),
            "region": region,
        }))
    if not candidate_rows:
        return None, None
    selected, _, info = min(candidate_rows, key=lambda row: row[1])
    return selected, info


def choose_identification_probe(
    env: gym.Env, state: np.ndarray, actions: np.ndarray,
    desired_moment_sign: float | None, *, allow_outward: bool = False,
    maximize: bool = False, reference_action: np.ndarray | None = None,
) -> int | None:
    """Choose a declared safe pulse for local effect identification.

    The legacy controller used the smallest pulse, which is conservative but
    cannot identify a sticking interface when the impulse is below the
    breakaway threshold.  ``maximize`` selects the largest action already in
    the shared candidate bank; it does not create an out-of-support action.
    """

    candidate_indices = [
        index for index, action in enumerate(np.asarray(actions, dtype=np.float64))
        if (allow_outward or float(np.dot(action, action_frame_normal(env, state, True))) >= 0.0)
        and float(np.linalg.norm(action)) > 1.0e-8
    ]
    if desired_moment_sign is not None and abs(float(desired_moment_sign)) > 0.5:
        signed = [
            index for index in candidate_indices
            if action_moment_sign(state, actions[index], env=env) == float(np.sign(desired_moment_sign))
        ]
        if signed:
            candidate_indices = signed
    if not candidate_indices:
        return None
    if reference_action is not None and env is not None:
        # Matched finite differences need a mirrored action, not merely any
        # action with the opposite cross-product sign.  Preserve the local
        # normal/tangential magnitudes while changing the moment sign.
        inward = action_frame_normal(env, state, True)
        tangent = np.asarray([-inward[1], inward[0]], dtype=np.float64)
        reference = np.asarray(reference_action, dtype=np.float64)
        reference_norm = float(np.linalg.norm(reference))
        reference_normal = abs(float(np.dot(reference, inward)))
        reference_tangent = abs(float(np.dot(reference, tangent)))
        def match_score(index: int) -> float:
            action = np.asarray(actions[index], dtype=np.float64)
            return (
                abs(float(np.linalg.norm(action)) - reference_norm) / 0.16
                + abs(abs(float(np.dot(action, inward))) - reference_normal) / 0.08
                + abs(abs(float(np.dot(action, tangent))) - reference_tangent) / 0.16
            )
        return int(min(candidate_indices, key=match_score))
    key = (lambda index: float(np.linalg.norm(actions[index])))
    return int((max if maximize else min)(candidate_indices, key=key))


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


def predict_transition_candidates(
    model: dict[str, Any], state: np.ndarray, actions: np.ndarray,
    previous_action: np.ndarray, *, mode_conditioned: bool, predictor: str = "tree",
    env: gym.Env | None = None,
    online_effects: dict[str, dict[str, Any]] | None = None,
    boundary_action_frame: bool = False,
    previous_observed_angle_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch the one-step candidate interface without changing its estimator.

    The controller evaluates the same candidate bank repeatedly.  Calling a
    forest once per candidate makes the original protocol needlessly slow and
    hid the result behind an execution timeout.  This function only batches
    sklearn predictions; feature construction, expert selection, online cache
    overrides, and all gates remain identical to ``predict_transition``.
    """
    from experiments.run_pusht_lifted_contact_adapter import lifted_contact_features

    actions_value = np.asarray(actions, dtype=np.float64)
    previous_value = np.asarray(previous_action, dtype=np.float64)
    if actions_value.ndim != 2 or actions_value.shape[1] != 2:
        raise ValueError("candidate actions must have shape (n, 2)")
    if boundary_action_frame:
        if env is None:
            raise ValueError("boundary action prediction requires the PushT environment geometry")
        feature_rows = [
            boundary_lifted_contact_features(
                env, state, action, previous_value, previous_observed_angle_delta,
            ) for action in actions_value
        ]
    else:
        feature_rows = [lifted_contact_features(state, action, previous_value) for action in actions_value]
    features = np.asarray(feature_rows, dtype=np.float64)
    mode = current_mode_for_env(env, state) if env is not None else current_mode(state)
    probabilities = np.zeros((len(actions_value), 4), dtype=np.float64)
    classes = np.asarray(model["classifier"].classes_, dtype=np.int64)
    probabilities[:, classes] = np.asarray(model["classifier"].predict_proba(features), dtype=np.float64)

    if predictor == "contact_chart" and mode == 2:
        chart = model["contact_chart"]
        chart_model = chart["by_mode"].get(mode, chart["global"])
        chart_features = np.asarray([
            physical_contact_chart_features(env, state, action, previous_value)
            if env is not None else contact_chart_features(state, action, previous_value)
            for action in actions_value
        ], dtype=np.float64)
        targets = np.asarray(chart_model.predict(chart_features), dtype=np.float64)
    elif mode_conditioned and model.get("by_successor"):
        fallback = model["by_mode"].get(mode, model["global"])
        components = []
        for successor in range(len(SUCCESSOR_NAMES)):
            expert = model["by_mode_successor"].get((mode, successor))
            if expert is None:
                expert = model["by_successor"].get(successor, fallback)
            components.append(np.asarray(expert.predict(features), dtype=np.float64))
        targets = np.sum(
            probabilities[:, :, None] * np.stack(components, axis=1), axis=1,
        )
    else:
        regressor = model["by_mode"].get(mode, model["global"]) if mode_conditioned else model["global"]
        targets = np.asarray(regressor.predict(features), dtype=np.float64)

    if online_effects is not None and env is not None and mode == 2:
        for index, action in enumerate(actions_value):
            cache = online_effects.get(online_effect_key(env, state, action))
            if cache is not None and int(cache.get("count", 0)) > 0:
                targets[index] = np.asarray(cache["mean"], dtype=np.float64)

    direction_classes = np.asarray(model["direction_classifier"].classes_, dtype=np.int64)
    direction_values = np.asarray(model["direction_classifier"].predict_proba(features), dtype=np.float64)
    directions = np.full(len(actions_value), 0.5, dtype=np.float64)
    positive = np.flatnonzero(direction_classes == 1)
    if len(positive):
        directions = direction_values[:, int(positive[0])]
    elif len(direction_classes):
        directions.fill(float(direction_classes[0] == 1))
    return targets, probabilities, directions


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
    contact_normal_magnitudes: tuple[float, ...] = (),
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
        normal_magnitudes = (
            tuple(float(value) for value in contact_normal_magnitudes)
            if len(contact_normal_magnitudes) else (float(inward_magnitude),)
        )
        action_parts = [
            pulse_actions(state, tangent_magnitudes, normal_value, normal)[0]
            for normal_value in normal_magnitudes
        ]
        actions = np.concatenate(action_parts, axis=0)
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
    contact_normal_magnitudes: tuple[float, ...] = (),
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
                contact_normal_magnitudes=contact_normal_magnitudes,
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


def largest_action_for_moment_sign(
    state: np.ndarray, actions: np.ndarray, desired_sign: float,
    env: gym.Env | None = None,
) -> int | None:
    """Return the strongest available candidate with a requested moment sign.

    This is used only by the explicit terminal sticking ablation.  It does
    not infer a new sign or bypass the action bank; it tests whether the
    minimum-norm terminal pulse is simply below the contact's breakaway
    magnitude.
    """

    if abs(float(desired_sign)) < 0.5:
        return None
    signs = np.asarray([action_moment_sign(state, action, env=env) for action in actions])
    valid = np.flatnonzero(signs == float(np.sign(desired_sign)))
    if not len(valid):
        return None
    return int(valid[np.argmax([np.linalg.norm(actions[index]) for index in valid])])


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


def first_step_score(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray,
    target: np.ndarray, probabilities: np.ndarray, direction_probability: float,
    *, phase_aware: bool, boundary_action_frame: bool, strict_direction_gate: bool,
) -> float:
    """Score one candidate using the exact one-step gates from imagined_score."""
    imagined = predicted_state(state, target)
    score = planning_pose_cost(state, imagined, goal, phase_aware)
    current = current_mode_for_env(env, state)
    if current in (1, 2):
        support_normal = action_frame_normal(env, state, boundary_action_frame)
        inward_component = float(np.dot(np.asarray(action, dtype=np.float64), support_normal))
        if inward_component < 0.0:
            return float(score + 100.0 + 10.0 * abs(inward_component))
    if current == 2:
        score += 8.0 * probabilities[3] + 1.5 * probabilities[0]
        desired_sign = float(np.sign(wrap_angle(goal[4] - state[4])))
        direction_match = direction_probability if desired_sign >= 0.0 else (1.0 - direction_probability)
        if strict_direction_gate and desired_sign != 0.0 and direction_match < 0.5:
            return float(score + 100.0 + 10.0 * (0.5 - direction_match))
        score += 4.0 * (1.0 - direction_match)
    else:
        score += 2.0 * probabilities[3]
    return float(score)


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
    score = first_step_score(
        env, state, goal, action, target, probabilities, direction_probability,
        phase_aware=phase_aware, boundary_action_frame=boundary_action_frame,
        strict_direction_gate=strict_direction_gate,
    )
    if depth <= 1:
        return float(score)
    second_scores = []
    for second in candidate_actions(
        env, imagined, tangent_magnitudes, inward_magnitude,
        goal=goal, goal_translation_magnitudes=goal_translation_magnitudes,
        boundary_action_frame=boundary_action_frame,
        contact_normal_magnitudes=tuple(getattr(args, "contact_normal_magnitudes", ())),
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


def staged_predictor_for_state(
    state: np.ndarray, goal: np.ndarray, translation_gate: float = 28.0,
) -> str:
    """Return the frozen stage for the state-only two-stage controller."""

    if float(translation_gate) < 0.0:
        raise ValueError("translation_gate must be non-negative")
    current_xy = float(np.linalg.norm(
        np.asarray(state, dtype=np.float64)[2:4]
        - np.asarray(goal, dtype=np.float64)[2:4]
    ))
    return "goal_ranker" if current_xy > float(translation_gate) else "tree"


def orientation_action_score(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray,
    target: np.ndarray, probabilities: np.ndarray, direction_probability: float,
    *, boundary_action_frame: bool, strict_direction_gate: bool,
) -> float:
    """Score a post-translation pulse for orientation without hiding XY drift.

    A macro's second pulse has a single declared job: reduce orientation error
    while retaining the contact lease.  The small XY term is a tie-breaker and
    safety term, not a second translation objective.  This separation is the
    intervention being tested; it must not be tuned from held-out outcomes.
    """

    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    imagined = predicted_state(state, target)
    current_mode = current_mode_for_env(env, state)
    score = 8.0 * angle_error(imagined[4], goal[4]) / np.pi
    score += 0.15 * float(np.linalg.norm(imagined[2:4] - goal[2:4])) / 20.0
    if current_mode in (1, 2):
        support_normal = action_frame_normal(env, state, boundary_action_frame)
        inward_component = float(np.dot(action, support_normal))
        if inward_component < 0.0:
            return float(score + 100.0 + 10.0 * abs(inward_component))
    if current_mode == 2:
        score += 8.0 * float(probabilities[3]) + 1.5 * float(probabilities[0])
        desired_sign = float(np.sign(wrap_angle(goal[4] - state[4])))
        direction_match = (
            float(direction_probability)
            if desired_sign >= 0.0 else 1.0 - float(direction_probability)
        )
        if strict_direction_gate and desired_sign != 0.0 and direction_match < 0.5:
            return float(score + 100.0 + 10.0 * (0.5 - direction_match))
        score += 4.0 * (1.0 - direction_match)
    else:
        score += 2.0 * float(probabilities[3])
    return float(score)


def select_orientation_action(
    env: gym.Env, model: dict[str, Any], state: np.ndarray, goal: np.ndarray,
    actions: np.ndarray, previous_action: np.ndarray, args: argparse.Namespace,
    *, mode_conditioned: bool, predictor: str = "tree",
    previous_observed_angle_delta: float = 0.0,
    forced_moment_sign: float | None = None,
    xy_shield_tolerance: float | None = None,
    orientation_effect_ranker: ExtraTreesRegressor | None = None,
    orientation_delta_ranker: ExtraTreesRegressor | None = None,
    orientation_pair_ranker: ExtraTreesRegressor | None = None,
) -> int:
    """Select the orientation half of a two-pulse macro from learned effects."""

    targets, probabilities, directions = predict_transition_candidates(
        model, state, actions, previous_action,
        mode_conditioned=mode_conditioned, predictor=predictor, env=env,
        boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
        previous_observed_angle_delta=previous_observed_angle_delta,
    )
    scores = [
        orientation_action_score(
            env, state, goal, action, target, probability, direction,
            boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            strict_direction_gate=bool(getattr(args, "strict_direction_gate", False)),
        )
        for action, target, probability, direction in zip(
            np.asarray(actions, dtype=np.float64), targets, probabilities, directions,
        )
    ]
    if orientation_pair_ranker is not None:
        corrected_deltas = np.asarray(targets[:, 0], dtype=np.float64).copy()
        pair_limit = min(
            len(actions),
            2 * len(tuple(getattr(args, "tangent_magnitudes", ()))) * max(
                1, len(tuple(getattr(args, "contact_normal_magnitudes", ())))
            ),
        )
        for first_index in range(0, pair_limit - 1, 2):
            pair_features = orientation_pair_ranker_features(
                state, goal,
                np.asarray(actions[first_index], dtype=np.float64),
                np.asarray(actions[first_index + 1], dtype=np.float64),
                previous_action=previous_action,
                previous_observed_angle_delta=previous_observed_angle_delta,
                env=env,
                boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            )[None]
            contrast = float(orientation_pair_ranker.predict(pair_features)[0])
            common = 0.5 * (float(targets[first_index, 0]) + float(targets[first_index + 1, 0]))
            corrected_deltas[first_index] = common + 0.5 * contrast
            corrected_deltas[first_index + 1] = common - 0.5 * contrast
        predicted_angles = np.asarray([
            wrap_angle(float(state[4]) + float(delta)) for delta in corrected_deltas
        ], dtype=np.float64)
        ranker_scores = 8.0 * np.asarray([
            angle_error(angle, goal[4]) for angle in predicted_angles
        ], dtype=np.float64) / np.pi
        ranker_scores += 0.15 * np.asarray([
            np.linalg.norm(predicted_state(state, target)[2:4] - goal[2:4]) / 20.0
            for target in targets
        ], dtype=np.float64)
        for index, (probability, direction) in enumerate(zip(probabilities, directions)):
            current_mode = current_mode_for_env(env, state)
            if current_mode == 2:
                ranker_scores[index] += 8.0 * float(probability[3]) + 1.5 * float(probability[0])
                desired_sign = float(np.sign(wrap_angle(goal[4] - state[4])))
                direction_match = float(direction) if desired_sign >= 0.0 else 1.0 - float(direction)
                ranker_scores[index] += 4.0 * (1.0 - direction_match)
            else:
                ranker_scores[index] += 2.0 * float(probability[3])
        scores = ranker_scores.tolist()
    elif orientation_delta_ranker is not None or orientation_effect_ranker is not None:
        ranker_features = np.asarray([
            goal_ranker_features(
                state, goal=goal, action=action, previous_action=previous_action,
                previous_observed_angle_delta=previous_observed_angle_delta,
                env=env,
                boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            )
            for action in np.asarray(actions, dtype=np.float64)
        ], dtype=np.float64)
        angular_errors: list[np.ndarray] = []
        if orientation_delta_ranker is not None:
            predicted_angle_delta = np.asarray(
                orientation_delta_ranker.predict(ranker_features), dtype=np.float64,
            )
            predicted_angle = np.asarray([
                wrap_angle(float(state[4]) + float(delta))
                for delta in predicted_angle_delta
            ], dtype=np.float64)
            angular_errors.append(np.asarray([
                angle_error(angle, goal[4]) for angle in predicted_angle
            ], dtype=np.float64))
        if orientation_effect_ranker is not None:
            angular_errors.append(np.maximum(
                0.0, np.asarray(orientation_effect_ranker.predict(ranker_features), dtype=np.float64)
            ))
        # Equal weighting is frozen before held-out evaluation.  When both
        # experts are supplied this is a dual representation audit; neither
        # expert can silently overwrite the other.
        predicted_angle_error = np.mean(np.stack(angular_errors, axis=0), axis=0)
        ranker_scores = 8.0 * predicted_angle_error / np.pi
        ranker_scores += 0.15 * np.asarray([
            np.linalg.norm(predicted_state(state, target)[2:4] - goal[2:4]) / 20.0
            for target in targets
        ], dtype=np.float64)
        # The delta ranker changes only the angular prediction term.  Keep the
        # fixed successor/contact-risk gate so this intervention tests signed
        # within-mode effect identification rather than a different safety
        # policy.
        for index, (probability, direction) in enumerate(zip(probabilities, directions)):
            current_mode = current_mode_for_env(env, state)
            if current_mode == 2:
                ranker_scores[index] += 8.0 * float(probability[3]) + 1.5 * float(probability[0])
                desired_sign = float(np.sign(wrap_angle(goal[4] - state[4])))
                direction_match = float(direction) if desired_sign >= 0.0 else 1.0 - float(direction)
                ranker_scores[index] += 4.0 * (1.0 - direction_match)
            else:
                ranker_scores[index] += 2.0 * float(probability[3])
        scores = ranker_scores.tolist()
    if xy_shield_tolerance is not None:
        valid = xy_shield_mask(state, goal, targets, float(xy_shield_tolerance))
        if np.any(valid):
            masked_scores = np.asarray(scores, dtype=np.float64)
            masked_scores[~valid] = np.inf
            scores = masked_scores.tolist()
        else:
            # Fail closed if all learned orientation candidates leave the
            # translation tube: choose the least damaging predicted XY move.
            predicted_xy = np.asarray([
                np.linalg.norm(
                    predicted_state(state, target)[2:4]
                    - np.asarray(goal, dtype=np.float64)[2:4]
                )
                for target in targets
            ], dtype=np.float64)
            return int(np.argmin(predicted_xy))
    if forced_moment_sign is not None and abs(float(forced_moment_sign)) >= 0.5:
        valid = np.asarray([
            index for index, action in enumerate(np.asarray(actions, dtype=np.float64))
            if action_moment_sign(
                state, action,
                env=env if bool(getattr(args, "boundary_action_frame", False)) else None,
            ) == float(np.sign(forced_moment_sign))
        ], dtype=np.int64)
        if len(valid):
            return int(valid[int(np.argmin(np.asarray(scores, dtype=np.float64)[valid]))])
    return int(np.argmin(np.asarray(scores, dtype=np.float64)))


def xy_shield_mask(
    state: np.ndarray, goal: np.ndarray, targets: np.ndarray, tolerance: float,
) -> np.ndarray:
    """Return candidates whose predicted XY error stays inside a fixed tube."""

    current_xy = float(np.linalg.norm(
        np.asarray(state, dtype=np.float64)[2:4]
        - np.asarray(goal, dtype=np.float64)[2:4]
    ))
    predicted_xy = np.asarray([
        np.linalg.norm(
            predicted_state(state, target)[2:4]
            - np.asarray(goal, dtype=np.float64)[2:4]
        )
        for target in np.asarray(targets, dtype=np.float64)
    ], dtype=np.float64)
    return predicted_xy <= current_xy + max(0.0, float(tolerance))


def macro_phase_name(orientation_first: bool, phase_index: int) -> str:
    """Return the fixed two-pulse phase order used by a macro ablation."""

    if int(phase_index) not in (0, 1):
        raise ValueError("a two-pulse macro only has phase indices 0 and 1")
    if bool(orientation_first):
        return "orientation" if int(phase_index) == 0 else "translation"
    return "translation" if int(phase_index) == 0 else "orientation"


def closed_loop_two_pulse_macro(
    env: gym.Env, roots: list[dict[str, np.ndarray]], model: dict[str, Any],
    args: argparse.Namespace, *, translation_predictor: str,
    orientation_predictor: str = "tree",
    observed_sign_feedback: bool = False,
    phase_gate: bool = False,
    macro_ranker: ExtraTreesRegressor | None = None,
    xy_shield_tolerance: float | None = None,
    orientation_first: bool = False,
    orientation_effect_ranker: ExtraTreesRegressor | None = None,
    orientation_delta_ranker: ExtraTreesRegressor | None = None,
    orientation_pair_ranker: ExtraTreesRegressor | None = None,
    macro_first_only: bool = False,
    macro_risk_penalty: float = 0.0,
    translation_xy_only: bool = False,
    reverse_phase_gate: bool = False,
) -> dict[str, Any]:
    """Run a fixed two-pulse translation-then-orientation macro controller.

    Each macro consumes exactly two pulses and is counted against half of the
    primitive controller budget.  The first pulse is selected with the
    declared translation interface, the environment is re-observed, and the
    second pulse is selected with the orientation-only learned score.  No
    simulator successor or privileged target is used at runtime.
    """

    records: list[dict[str, Any]] = []
    pulse_steps = int(getattr(args, "macro_pulse_steps", args.pulse_steps))
    primitive_budget = int(args.closure_steps) * int(args.pulse_steps)
    macro_steps = max(1, primitive_budget // max(2 * pulse_steps, 1))
    for root_index, root in enumerate(roots):
        observation, _ = env.reset(seed=0, options={
            "state": root["state"].copy(), "goal_state": root["goal"].copy(),
        })
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        goal = np.asarray(root["goal"], dtype=np.float64).copy()
        previous = np.zeros(2, dtype=np.float64)
        last_observed_delta = 0.0
        trace: list[dict[str, Any]] = []
        for macro_index in range(macro_steps):
            if pose_cost(current, goal) < 1.0:
                break
            macro_initial_state = current.copy()
            macro_first_action: np.ndarray | None = None
            for phase_index in range(2):
                if pose_cost(current, goal) < 1.0:
                    break
                if orientation_first:
                    if (
                        int(phase_index) == 1
                        and bool(reverse_phase_gate)
                        and float(np.linalg.norm(current[2:4] - goal[2:4]))
                        <= float(getattr(args, "staged_translation_gate", 28.0))
                    ):
                        phase = "orientation_refinement"
                    else:
                        phase = macro_phase_name(True, phase_index)
                    phase_predictor = (
                        orientation_predictor if phase.startswith("orientation")
                        else translation_predictor
                    )
                elif phase_index == 0:
                    phase = "translation"
                    phase_predictor = translation_predictor
                elif phase_gate and staged_predictor_for_state(
                    current, goal, float(getattr(args, "staged_translation_gate", 28.0))
                ) == "goal_ranker":
                    phase = "translation_continuation"
                    phase_predictor = translation_predictor
                else:
                    phase = "orientation"
                    phase_predictor = orientation_predictor
                actions = candidate_actions(
                    env, current, args.tangent_magnitudes, args.inward_magnitude,
                    goal=goal,
                    goal_translation_magnitudes=args.goal_translation_magnitudes,
                    boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                    contact_normal_magnitudes=tuple(getattr(args, "contact_normal_magnitudes", ())),
                )
                forced_moment_sign = None
                if macro_ranker is not None and phase_index == 0:
                    selected = select_macro_first_action(
                        env, model, macro_ranker, current, goal, actions, previous, args,
                        risk_penalty=macro_risk_penalty,
                    )
                elif macro_ranker is not None and phase_index == 1 and not macro_first_only:
                    selected = select_macro_second_action(
                        env, macro_ranker, macro_initial_state, goal,
                        np.asarray(macro_first_action, dtype=np.float64), current,
                        actions, previous, args, risk_penalty=macro_risk_penalty,
                    )
                elif phase.startswith("orientation"):
                    forced_moment_sign = None
                    if observed_sign_feedback:
                        desired_sign = float(np.sign(wrap_angle(goal[4] - current[4])))
                        observed_sign = float(np.sign(last_observed_delta))
                        if desired_sign != 0.0 and observed_sign != 0.0 and observed_sign != desired_sign:
                            previous_moment_sign = action_moment_sign(
                                current, previous,
                                env=env if bool(getattr(args, "boundary_action_frame", False)) else None,
                            )
                            if abs(previous_moment_sign) >= 0.5:
                                forced_moment_sign = -float(previous_moment_sign)
                    selected = select_orientation_action(
                        env, model, current, goal, actions, previous, args,
                        mode_conditioned=True, predictor=phase_predictor,
                        previous_observed_angle_delta=last_observed_delta,
                        forced_moment_sign=forced_moment_sign,
                        xy_shield_tolerance=xy_shield_tolerance,
                        orientation_effect_ranker=orientation_effect_ranker,
                        orientation_delta_ranker=orientation_delta_ranker,
                        orientation_pair_ranker=orientation_pair_ranker,
                    )
                elif translation_xy_only and phase.startswith("translation"):
                    selected = select_translation_action(
                        env, model, current, goal, actions, previous, args,
                    )
                else:
                    selected = select_action(
                        env, model, current, goal, actions, previous, args,
                        mode_conditioned=True, predictor=phase_predictor,
                        previous_observed_angle_delta=last_observed_delta,
                    )
                action = np.asarray(actions[selected], dtype=np.float32)
                if phase_index == 0:
                    macro_first_action = action.astype(np.float64).copy()
                before = current.copy()
                predicted_target, probabilities, direction_probability = predict_transition(
                    model, before, action, previous,
                    mode_conditioned=True, predictor=phase_predictor, env=env,
                    boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                    previous_observed_angle_delta=last_observed_delta,
                )
                terminated = truncated = False
                for _ in range(pulse_steps):
                    observation, _, terminated, truncated, _ = env.step(action)
                    current = np.asarray(observation["state"], dtype=np.float64).copy()
                    if terminated or truncated:
                        break
                observed_target = transition_target_from_states(before, current)
                trace.append({
                    "macro_index": int(macro_index), "phase": phase,
                    "selected": int(selected), "action": action.tolist(),
                    "execution_steps": int(pulse_steps),
                    "controller_predictor": phase_predictor,
                    "observed_sign_feedback": bool(observed_sign_feedback),
                    "forced_moment_sign": forced_moment_sign if phase == "orientation" else None,
                    "block_before": before[2:4].tolist(),
                    "block_after": current[2:4].tolist(),
                    "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
                    "angle_after": angle_error(current[4], goal[4]),
                    "pose_cost_after": pose_cost(current, goal),
                    "desired_angle_sign": float(np.sign(wrap_angle(goal[4] - before[4]))),
                    "action_moment_sign": float(action_moment_sign(
                        before, action, env=env if args.boundary_action_frame else None,
                    )),
                    "boundary_distance_after": float(physical_contact_geometry(env, current)["distance"]),
                    "contact_distance_after": float(np.linalg.norm(current[:2] - current[2:4])),
                    "predicted_observed_effect_error": normalized_effect_error(
                        predicted_target, observed_target,
                    ),
                    "predicted_mode_probabilities": probabilities.tolist(),
                    "predicted_direction_probability": float(direction_probability),
                })
                previous = action.astype(np.float64)
                last_observed_delta = float(wrap_angle(current[4] - before[4]))
                if terminated or truncated:
                    break
            if trace and (terminated or truncated):
                break
        records.append({
            "root_index": int(root_index),
            "final_xy": float(np.linalg.norm(current[2:4] - goal[2:4])),
            "final_angle": angle_error(current[4], goal[4]),
            "pose_success": bool(
                np.linalg.norm(current[2:4] - goal[2:4]) < 20.0
                and angle_error(current[4], goal[4]) < np.pi / 9.0
            ),
            "trace": trace,
        })
    return {
        "metrics": {
            "pose_success": bootstrap_mean([float(row["pose_success"]) for row in records], 94501),
            "final_xy": bootstrap_mean([row["final_xy"] for row in records], 94502),
            "final_angle": bootstrap_mean([row["final_angle"] for row in records], 94503),
            "contact_loss_rate": bootstrap_mean([
                float(protocol_contact_loss(row["trace"], getattr(args, "root_protocol", "terminal_pose")))
                for row in records
            ], 94504),
            "boundary_contact_loss_rate": bootstrap_mean([
                float(any(item["boundary_distance_after"] > 70.0 for item in row["trace"]))
                for row in records
            ], 94505),
            "predicted_observed_effect_error": bootstrap_mean([
                float(np.mean([item["predicted_observed_effect_error"] for item in row["trace"]]))
                if row["trace"] else 0.0 for row in records
            ], 94506),
        },
        "controller": "two_pulse_macro",
        "translation_predictor": translation_predictor,
        "orientation_predictor": orientation_predictor,
        "observed_sign_feedback": bool(observed_sign_feedback),
        "phase_gate": bool(phase_gate),
        "macro_ranker": bool(macro_ranker is not None),
        "orientation_first": bool(orientation_first),
        "orientation_effect_ranker": bool(orientation_effect_ranker is not None),
        "orientation_delta_ranker": bool(orientation_delta_ranker is not None),
        "orientation_pair_ranker": bool(orientation_pair_ranker is not None),
        "translation_xy_only": bool(translation_xy_only),
        "reverse_phase_gate": bool(reverse_phase_gate),
        "xy_shield_tolerance": (
            None if xy_shield_tolerance is None else float(xy_shield_tolerance)
        ),
        "macro_steps": macro_steps,
        "pulse_steps": pulse_steps,
        "budget_matched_to_primitive": bool(
            2 * macro_steps * pulse_steps == int(args.closure_steps) * int(args.pulse_steps)
        ),
        "runtime_successor_access": False,
        "rows": records,
    }


def select_action(
    env: gym.Env, model: dict[str, Any], state: np.ndarray, goal: np.ndarray,
    actions: np.ndarray, previous_action: np.ndarray, args: argparse.Namespace,
    *, mode_conditioned: bool,
    predictor: str = "tree",
    online_effects: dict[str, dict[str, Any]] | None = None,
    previous_observed_angle_delta: float = 0.0,
) -> int:
    if predictor == "staged":
        # Declared two-stage interface: use goal-conditioned translation
        # ranking outside the local XY tube, then return to the mode-aware
        # action-effect model for within-contact orientation.  The stage uses
        # only the current observed state and goal; it never reads a successor.
        stage_predictor = staged_predictor_for_state(
            state, goal, float(getattr(args, "staged_translation_gate", 28.0)),
        )
        return select_action(
            env, model, state, goal, actions, previous_action, args,
            mode_conditioned=mode_conditioned,
            predictor=stage_predictor,
            online_effects=online_effects,
            previous_observed_angle_delta=previous_observed_angle_delta,
        )
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
    if int(args.lookahead) <= 1:
        targets, probabilities, directions = predict_transition_candidates(
            model, state, actions, previous_action,
            mode_conditioned=mode_conditioned, predictor=predictor, env=env,
            online_effects=online_effects,
            boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            previous_observed_angle_delta=previous_observed_angle_delta,
        )
        scores = np.asarray([
            first_step_score(
                env, state, goal, action, target, probability, direction,
                phase_aware=args.phase_aware_cost,
                boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                strict_direction_gate=bool(getattr(args, "strict_direction_gate", False)),
            )
            for action, target, probability, direction in zip(
                actions, targets, probabilities, directions,
            )
        ], dtype=np.float64)
        return int(np.argmin(scores))
    # Preserve the depth-2 score while batching expensive forest calls. The
    # old path predicted once per first candidate and once per second
    # candidate, making formal root-disjoint runs dominated by repeated
    # ExtraTrees inference. Each imagined state still gets its own geometry-
    # aware second candidate bank; only estimator calls are batched.
    targets, probabilities, directions = predict_transition_candidates(
        model, state, actions, previous_action,
        mode_conditioned=mode_conditioned, predictor=predictor, env=env,
        online_effects=online_effects,
        boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
        previous_observed_angle_delta=previous_observed_angle_delta,
    )
    scores: list[float] = []
    for index, action in enumerate(np.asarray(actions, dtype=np.float64)):
        target = np.asarray(targets[index], dtype=np.float64)
        imagined = predicted_state(state, target)
        score = first_step_score(
            env, state, goal, action, target, probabilities[index], directions[index],
            phase_aware=args.phase_aware_cost,
            boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            strict_direction_gate=bool(getattr(args, "strict_direction_gate", False)),
        )
        second_actions = candidate_actions(
            env, imagined, args.tangent_magnitudes, args.inward_magnitude,
            goal=goal, goal_translation_magnitudes=args.goal_translation_magnitudes,
            boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            contact_normal_magnitudes=tuple(getattr(args, "contact_normal_magnitudes", ())),
        )
        second_targets, second_probabilities, second_directions = predict_transition_candidates(
            model, imagined, second_actions, action,
            mode_conditioned=mode_conditioned, predictor=predictor, env=env,
            online_effects=online_effects,
            boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
            previous_observed_angle_delta=float(target[0]),
        )
        second_scores: list[float] = []
        for second_index, second_target in enumerate(second_targets):
            second_state = predicted_state(imagined, second_target)
            second_score = planning_pose_cost(imagined, second_state, goal, args.phase_aware_cost)
            if current_mode_for_env(env, imagined) == 2:
                second_score += 8.0 * second_probabilities[second_index, 3]
                second_score += 1.5 * second_probabilities[second_index, 0]
                desired_sign = float(np.sign(wrap_angle(goal[4] - imagined[4])))
                direction_match = (
                    second_directions[second_index]
                    if desired_sign >= 0.0 else 1.0 - second_directions[second_index]
                )
                if bool(getattr(args, "strict_direction_gate", False)) and desired_sign != 0.0 and direction_match < 0.5:
                    second_score += 100.0 + 10.0 * (0.5 - direction_match)
                second_score += 4.0 * (1.0 - direction_match)
            second_scores.append(float(second_score))
        scores.append(float(score + 0.5 * min(second_scores)) if second_scores else float(score))
    return int(np.argmin(np.asarray(scores, dtype=np.float64)))


def select_translation_action(
    env: gym.Env, model: dict[str, Any], state: np.ndarray, goal: np.ndarray,
    actions: np.ndarray, previous_action: np.ndarray, args: argparse.Namespace,
) -> int:
    """Select a pulse with a frozen translation-first objective.

    This interface is used only for the declared translation phase of the
    reverse macro.  It keeps the learned successor/contact-risk interface but
    does not allow a predicted angle improvement to consume the translation
    correction budget.
    """

    targets, probabilities, _ = predict_transition_candidates(
        model, state, actions, previous_action,
        mode_conditioned=True, predictor="tree", env=env,
        boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
    )
    scores = np.asarray([
        np.linalg.norm(predicted_state(state, target)[2:4] - goal[2:4]) / 20.0
        + 0.25 * angle_error(predicted_state(state, target)[4], goal[4]) / (np.pi / 9.0)
        + (8.0 * float(probability[3]) + 1.5 * float(probability[0])
           if current_mode_for_env(env, state) == 2 else 2.0 * float(probability[3]))
        for target, probability in zip(targets, probabilities)
    ], dtype=np.float64)
    return int(np.argmin(scores))


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
            candidate_actions = np.asarray([row["action"] for row in root_rows], dtype=np.float64)
            if depth <= 1:
                targets, probabilities, directions = predict_transition_candidates(
                    model, state, candidate_actions, previous,
                    mode_conditioned=conditioned, predictor=predictor, env=env,
                    boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                )
                scores = [
                    first_step_score(
                        env, state, goal, action, target, probability, direction,
                        phase_aware=args.phase_aware_cost,
                        boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                        strict_direction_gate=bool(getattr(args, "strict_direction_gate", False)),
                    )
                    for action, target, probability, direction in zip(
                        candidate_actions, targets, probabilities, directions,
                    )
                ]
                selected = int(np.argmin(np.asarray(scores, dtype=np.float64)))
            else:
                # ``select_action`` uses the same depth-2 score but routes all
                # candidate forest calls through the batch interface.  Keep
                # ranking and closed-loop selection on one implementation so
                # the audit cannot silently become a different protocol.
                selected = select_action(
                    env, model, state, goal, candidate_actions, previous, args,
                    mode_conditioned=conditioned, predictor=predictor,
                )
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
    uncertainty_gate: bool = False,
    fallback_only: bool = False,
    adaptive_online_effect: bool = False,
    orientation_gate_keys: set[str] | None = None,
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
        online_local_effects: dict[str, dict[str, Any]] = {}
        probe_moment_sign_by_region: dict[str, float] = {}
        probe_action_by_region: dict[str, np.ndarray] = {}
        calibration_attempts = 0
        calibration_successes = 0
        calibration_attempted_modes: set[int] = set()
        calibration_attempted_regions: set[str] = set()
        wrong_sign_events = 0
        relocation_attempts = 0
        relocation_pending = False
        relocation_records: list[dict[str, Any]] = []
        stagnation_steps = 0
        best_task_cost = pose_cost(current, goal)
        trace = []
        for step in range(args.closure_steps):
            if pose_cost(current, goal) < 1.0:
                break
            contact_distance = float(np.linalg.norm(current[:2] - current[2:4]))
            boundary_distance = float(physical_contact_geometry(env, current)["distance"])
            current_mode_id = current_mode_for_env(env, current)
            current_xy_error = float(np.linalg.norm(current[2:4] - goal[2:4]))
            contact_lease_active = bool(
                getattr(args, "contact_lease", False)
                and current_mode_id == 2
                and current_xy_error <= float(
                    getattr(args, "contact_lease_xy_gate", args.terminal_xy_gate)
                )
                and angle_error(current[4], goal[4]) > float(args.terminal_angle_gate)
            )
            current_region_key = (
                contact_region_key(env, current) if current_mode_id == 2 else None
            )
            if (
                bool(getattr(args, "contact_relocation_on_stagnation", False))
                and relocation_pending
                and relocation_attempts < int(args.max_contact_relocations)
                and not contact_lease_active
            ):
                # Contact-side relocation is an explicit physical interface
                # stage.  It consumes only the current observed state and
                # goal; it never reads a candidate successor or learned label.
                outward = relocation_outward_direction(current, goal, relocation_attempts)
                before_relocation = current.copy()
                current, route_info = relocate_to_contact_with_boundary_detach(
                    env, current, goal, outward, args,
                )
                relocation_attempts += 1
                relocation_pending = False
                stagnation_steps = 0
                best_task_cost = pose_cost(current, goal)
                previous = np.zeros(2, dtype=np.float64)
                last_observed_delta = 0.0
                relocation_records.append({
                    "attempt": int(relocation_attempts),
                    "outward": outward.tolist(),
                    "block_displacement": float(np.linalg.norm(current[2:4] - before_relocation[2:4])),
                    "route_steps": int(route_info.get("route_steps", 0)),
                    "contact_distance": float(route_info.get("contact_distance", np.nan)),
                    "terminated": bool(route_info.get("terminated", False)),
                    "detach_steps": int(route_info.get("detach_steps", 0)),
                    "detachment_geometry": route_info.get("detachment_geometry"),
                })
                continue
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
            fallback_used = False
            identification_probe = False
            online_identified_used = False
            force_identification_probe = False
            gate_decision: dict[str, Any] | None = None
            orientation_gate_key_value: str | None = None
            orientation_gate_blocked = False
            if orientation_gate_keys is not None and current_mode_id == 2:
                orientation_gate_key_value = orientation_gate_key(env, current)
                orientation_gate_blocked = orientation_gate_key_value not in orientation_gate_keys
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
                    contact_normal_magnitudes=tuple(getattr(args, "contact_normal_magnitudes", ())),
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
                    if bool(getattr(args, "terminal_orientation_ranker", False)):
                        # Learned orientation-only ranking is a terminal
                        # intervention that avoids assuming a portable
                        # geometry-to-torque sign across contact regions.
                        selected = select_orientation_action(
                            env, model, current, goal, actions, previous, args,
                            mode_conditioned=runtime_mode_conditioned,
                            predictor="tree",
                            previous_observed_angle_delta=last_observed_delta,
                            xy_shield_tolerance=(
                                float(getattr(args, "orientation_xy_budget", 0.0))
                                if bool(getattr(args, "contact_lease", False)) else None
                            ),
                        )
                    else:
                        desired_moment_sign = desired_sign if moment_to_angle_sign is None else desired_sign * moment_to_angle_sign
                        terminal_action_selector = (
                            largest_action_for_moment_sign
                            if bool(getattr(args, "terminal_max_moment", False))
                            else smallest_action_for_moment_sign
                        )
                        selected = terminal_action_selector(
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
                        replacement_selector = (
                            largest_action_for_moment_sign
                            if bool(getattr(args, "escalate_on_wrong_sign", False))
                            else smallest_action_for_moment_sign
                        )
                        replacement = replacement_selector(
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
                if (
                    adaptive_online_effect
                    and not calibration_probe
                    and (
                        not terminal_handoff
                        or bool(getattr(args, "terminal_observed_effect", False))
                    )
                    and not fallback_only
                    and current_mode_id == 2
                ):
                    if bool(getattr(args, "online_local_jacobian", False)):
                        online_selected, online_info = choose_local_jacobian_action(
                            env, current, goal, actions, online_local_effects,
                            max_radius=(
                                args.online_paired_radius
                                if args.paired_effect_probes else args.online_effect_radius
                            ),
                            min_samples=int(getattr(args, "online_jacobian_min_samples", 2)),
                            require_paired=bool(
                                args.paired_effect_probes and not args.allow_unpaired_local_effects
                            ),
                        )
                    else:
                        online_selected, online_info = choose_online_effect_action(
                            env, current, goal, actions, online_local_effects,
                            max_neighbors=args.online_effect_neighbors,
                            max_radius=(
                                args.online_paired_radius
                                if args.paired_effect_probes else args.online_effect_radius
                            ),
                            task_gain_weight=args.online_task_gain_weight,
                            min_informative_count=args.online_min_informative_count,
                            min_effect_angle=args.online_min_effect_angle,
                            require_paired=bool(
                                args.paired_effect_probes and not args.allow_unpaired_local_effects
                            ),
                        )
                    if online_selected is not None:
                        selected = int(online_selected)
                        online_identified_used = True
                        gate_decision = {
                            "valid": True,
                            "reason": "observed_local_effect",
                            "key": online_info.get("key") if online_info else None,
                            "stats": online_info,
                        }
                    elif args.paired_effect_probes:
                        # No matched, informative local contrast exists yet.
                        # Force the fail-closed branch below to issue an
                        # alternating probe instead of allowing the learned
                        # gate to reuse an unsupported global action.
                        force_identification_probe = True
            if (
                not recovery_macro
                and not calibration_probe
                and current_mode_id == 2
                and (
                    uncertainty_gate or fallback_only
                    or orientation_gate_keys is not None
                    or force_identification_probe
                )
                and (
                    not terminal_handoff
                    or bool(getattr(args, "terminal_observed_effect", False))
                )
                and not online_identified_used
            ):
                if force_identification_probe:
                    gate_decision = {
                        "valid": False,
                        "reason": "insufficient_online_paired_support",
                        "key": None,
                        "stats": None,
                    }
                elif orientation_gate_blocked:
                    gate_decision = {
                        "valid": False,
                        "reason": "orientation_stratum_unauthorized",
                        "key": orientation_gate_key_value,
                        "stats": None,
                    }
                elif fallback_only:
                    gate_decision = {
                        "valid": False, "reason": "fallback_only",
                        "key": None, "stats": None,
                    }
                else:
                    gate_decision = effect_gate_decision(
                        model, env, current, np.asarray(actions[selected]), goal, args,
                    )
                if orientation_gate_blocked or fallback_only or not bool(gate_decision["valid"]):
                    # Pure inward motion preserves the observed contact
                    # manifold; it is deliberately not credited to the
                    # learned effect model and is followed by re-observation.
                    fallback_normal = action_frame_normal(
                        env, current, bool(getattr(args, "boundary_action_frame", False))
                    )
                    if adaptive_online_effect and not fallback_only:
                        desired_moment_sign = None
                        desired_sign = float(np.sign(wrap_angle(goal[4] - current[4])))
                        if moment_to_angle_sign is not None and desired_sign != 0.0:
                            desired_moment_sign = desired_sign * float(moment_to_angle_sign)
                        if args.paired_effect_probes and current_region_key is not None:
                            previous_probe_sign = probe_moment_sign_by_region.get(current_region_key)
                            if previous_probe_sign is not None:
                                desired_moment_sign = -float(previous_probe_sign)
                        reference_action = (
                            probe_action_by_region.get(current_region_key)
                            if args.paired_effect_probes and current_region_key is not None
                            else None
                        )
                        if (
                            bool(getattr(args, "breakaway_paired_probe", False))
                            and args.paired_effect_probes
                            and current_region_key is not None
                        ):
                            observed_region_signs = {
                                int(entry.get("descriptor", {}).get("moment_sign", 0))
                                for entry in online_local_effects.values()
                                if str(entry.get("descriptor", {}).get("region"))
                                == str(current_region_key)
                            }
                            # The first calibration pulse is intentionally
                            # small.  Do not let it anchor the paired probe
                            # below breakaway; choose a declared max-moment
                            # pulse, then mirror it on the next observation.
                            if not ({-1, 1} <= observed_region_signs):
                                reference_action = None
                        probe = choose_identification_probe(
                            env, current, actions, desired_moment_sign,
                            allow_outward=bool(args.paired_effect_probes),
                            maximize=bool(args.active_probe_max_moment),
                            reference_action=reference_action,
                        )
                    else:
                        probe = None
                    if probe is not None:
                        selected = int(probe)
                        identification_probe = True
                        action = np.asarray(actions[selected], dtype=np.float32)
                    else:
                        action = np.asarray(
                            float(args.fallback_magnitude) * fallback_normal,
                            dtype=np.float32,
                        )
                        selected = -1
                        fallback_used = True
                else:
                    action = np.asarray(actions[selected], dtype=np.float32)
            else:
                action = np.asarray(actions[selected], dtype=np.float32)
            before = recovery_before if recovery_macro else current.copy()
            terminated = truncated = False
            execution_steps = (
                0 if recovery_macro
                else args.recovery_pulse_steps if recovery
                else args.fallback_pulse_steps if (fallback_used or identification_probe)
                else args.identification_pulse_steps if identification_probe
                else args.calibration_pulse_steps if calibration_probe else args.pulse_steps
            )
            if terminal_handoff and not recovery_macro and not calibration_probe and not fallback_used:
                # Once translation is inside the terminal tube, use a short
                # orientation pulse.  Reusing the transport pulse here can
                # overshoot a nearly closed pose and is a protocol mismatch,
                # not evidence about the learned effect model.
                execution_steps = int(args.terminal_pulse_steps)
            predicted_target = None
            predicted_mode_probabilities = None
            predicted_direction_probability = None
            if not recovery_macro and not fallback_used:
                predicted_target, predicted_mode_probabilities, predicted_direction_probability = predict_transition(
                    model, before, action, previous,
                    mode_conditioned=bool(mode_conditioned and current_mode_id == 2),
                    predictor=predictor, env=env,
                    online_effects=online_effects if args.online_effect_correction else None,
                    boundary_action_frame=bool(getattr(args, "boundary_action_frame", False)),
                    previous_observed_angle_delta=last_observed_delta,
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
                "controller_stage": (
                    staged_predictor_for_state(
                        before, goal, float(getattr(args, "staged_translation_gate", 28.0))
                    ) if predictor == "staged" else None
                ),
                "terminal_handoff": terminal_handoff,
                "contact_lease": bool(getattr(args, "contact_lease", False)),
                "contact_lease_active": contact_lease_active,
                "orientation_xy_budget": (
                    float(getattr(args, "orientation_xy_budget", 0.0))
                    if bool(getattr(args, "contact_lease", False)) else None
                ),
                "terminal_orientation_ranker": bool(getattr(args, "terminal_orientation_ranker", False)),
                "observed_sign_feedback": bool(observed_sign_feedback),
                "observed_sign_calibration": bool(observed_sign_calibration),
                "strict_contact_sign_gate": bool(strict_contact_sign_gate),
                "direct_observed_sign_reversal": bool(direct_observed_sign_reversal),
                "calibration_probe": bool(calibration_probe),
                "uncertainty_gate": bool(uncertainty_gate),
                "fallback_only": bool(fallback_only),
                "fallback_used": bool(fallback_used),
                "adaptive_online_effect": bool(adaptive_online_effect),
                "identification_probe": bool(identification_probe),
                "online_identified_used": bool(online_identified_used),
                "effect_gate": gate_decision,
                "orientation_gate_key": orientation_gate_key_value,
                "orientation_gate_blocked": bool(orientation_gate_blocked),
                "moment_to_angle_sign": moment_to_angle_sign,
                "execution_steps": int(execution_steps),
                "predicted_mode_probabilities": None if predicted_mode_probabilities is None else predicted_mode_probabilities.tolist(),
                "predicted_direction_probability": None if predicted_direction_probability is None else float(predicted_direction_probability),
                "pose_cost_after": pose_cost(current, goal),
                "angle_after": angle_error(current[4], goal[4]),
                "observed_angle_delta": float(wrap_angle(current[4] - before[4])),
                "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
                "contact_distance_after": float(np.linalg.norm(current[:2] - current[2:4])),
                "boundary_distance_after": float(physical_contact_geometry(env, current)["distance"]),
                "boundary_mode_after": MODE_NAMES[current_mode_for_env(env, current)],
            })
            observed_target = transition_target_from_states(before, current)
            if predicted_target is not None:
                trace[-1]["predicted_observed_effect_error"] = normalized_effect_error(
                    predicted_target, observed_target,
                )
                trace[-1]["predicted_observed_effect_error_raw"] = float(
                    np.linalg.norm(np.asarray(predicted_target) - np.asarray(observed_target))
                )
            else:
                trace[-1]["predicted_observed_effect_error"] = None
                trace[-1]["predicted_observed_effect_error_raw"] = None
            previous = action.astype(np.float64)
            last_observed_delta = float(wrap_angle(current[4] - before[4]))
            if adaptive_online_effect and not recovery_macro and current_mode_id == 2:
                local_key = effect_gate_key(env, before, action)
                descriptor = online_effect_descriptor(env, before, action)
                task_gain = planning_pose_cost(before, before, goal, phase_aware=True) - planning_pose_cost(
                    before, current, goal, phase_aware=True,
                )
                local_entry = online_local_effects.get(local_key)
                if (identification_probe or calibration_probe) and current_region_key is not None:
                    probe_moment_sign_by_region[current_region_key] = float(descriptor["moment_sign"])
                    probe_action_by_region[current_region_key] = np.asarray(action, dtype=np.float64).copy()
                informative = bool(abs(float(observed_target[0])) >= args.online_min_effect_angle)
                if local_entry is None:
                    online_local_effects[local_key] = {
                        "mean": observed_target.tolist(), "count": 1,
                        "informative_count": int(informative),
                        "mean_angle_abs": abs(float(observed_target[0])),
                        "sign_sum": float(np.sign(observed_target[0]) * action_moment_sign(before, action, env=env)),
                        "last_error": 0.0 if predicted_target is None else normalized_effect_error(predicted_target, observed_target),
                        "descriptor": descriptor,
                        "action": np.asarray(action, dtype=np.float64).tolist(),
                        "task_gain": float(task_gain),
                    }
                else:
                    count = int(local_entry["count"])
                    mean = np.asarray(local_entry["mean"], dtype=np.float64)
                    local_entry["mean"] = (mean + (observed_target - mean) / float(count + 1)).tolist()
                    local_entry["count"] = count + 1
                    informative_count = int(local_entry.get("informative_count", 0))
                    local_entry["informative_count"] = informative_count + int(informative)
                    local_entry["mean_angle_abs"] = float(
                        (float(local_entry.get("mean_angle_abs", 0.0)) * count
                         + abs(float(observed_target[0]))) / float(count + 1)
                    )
                    local_entry["sign_sum"] = float(
                        local_entry.get("sign_sum", 0.0)
                        + np.sign(observed_target[0]) * action_moment_sign(before, action, env=env)
                    )
                    local_entry["last_error"] = float(
                        0.0 if predicted_target is None else normalized_effect_error(predicted_target, observed_target)
                    )
                    local_entry["task_gain"] = float(
                        (float(local_entry.get("task_gain", 0.0)) * count + float(task_gain)) / float(count + 1)
                    )
                    local_entry["descriptor"] = descriptor
                    local_entry["action"] = np.asarray(action, dtype=np.float64).tolist()
            if args.online_effect_correction and not recovery_macro and not fallback_used and current_mode_id == 2:
                key = online_effect_key(env, before, action)
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
            if bool(getattr(args, "contact_relocation_on_stagnation", False)):
                current_task_cost = pose_cost(current, goal)
                if current_task_cost < best_task_cost - float(args.min_relocation_improvement):
                    stagnation_steps = 0
                else:
                    stagnation_steps += 1
                best_task_cost = min(best_task_cost, current_task_cost)
                desired_angle_sign = float(np.sign(wrap_angle(goal[4] - before[4])))
                wrong_local_direction = bool(
                    abs(last_observed_delta) > args.calibration_angle_threshold
                    and desired_angle_sign != 0.0
                    and float(np.sign(last_observed_delta)) != desired_angle_sign
                )
                if (
                    current_mode_id == 2
                    and relocation_attempts < int(args.max_contact_relocations)
                    and (wrong_local_direction or stagnation_steps >= int(args.relocation_stagnation_steps))
                ):
                    relocation_pending = True
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
            "relocation_attempts": int(relocation_attempts),
            "relocation_records": relocation_records,
        })
    return {
        "metrics": {
            "pose_success": bootstrap_mean([float(row["pose_success"]) for row in records], 94201),
            "final_xy": bootstrap_mean([row["final_xy"] for row in records], 94202),
            "final_angle": bootstrap_mean([row["final_angle"] for row in records], 94203),
            "contact_loss_rate": bootstrap_mean([
                # Boundary-contact roots deliberately place the pusher on a
                # polygon edge.  Their legal pusher-to-block-centre distance
                # is often >80 px, so the legacy centre-distance predicate
                # reports legal contact as a loss.  Use the physical boundary
                # gap for the protocol metric and retain the centre-distance
                # quantity below as a compatibility diagnostic.
                float(protocol_contact_loss(
                    row["trace"], getattr(args, "root_protocol", "terminal_pose"),
                ))
                for row in records
            ], 94204),
            "center_distance_loss_rate": bootstrap_mean([
                float(any(item["contact_distance_after"] > 80.0 for item in row["trace"]))
                for row in records
            ], 94219),
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
            "fallback_rate": bootstrap_mean([
                float(np.mean([bool(item.get("fallback_used", False)) for item in row["trace"]]))
                if row["trace"] else 0.0
                for row in records
            ], 94210),
            "root_fallback_rate": bootstrap_mean([
                float(any(bool(item.get("fallback_used", False)) for item in row["trace"]))
                for row in records
            ], 94211),
            "identification_probe_rate": bootstrap_mean([
                float(np.mean([bool(item.get("identification_probe", False)) for item in row["trace"]]))
                if row["trace"] else 0.0
                for row in records
            ], 94215),
            "online_identified_rate": bootstrap_mean([
                float(np.mean([bool(item.get("online_identified_used", False)) for item in row["trace"]]))
                if row["trace"] else 0.0
                for row in records
            ], 94216),
            "gate_accept_rate": bootstrap_mean([
                float(np.mean([
                    bool(item.get("effect_gate", {}).get("valid", False))
                    for item in row["trace"]
                    if item.get("effect_gate") is not None
                ])) if any(item.get("effect_gate") is not None for item in row["trace"]) else 0.0
                for row in records
            ], 94212),
            "gate_no_support_rate": bootstrap_mean([
                float(np.mean([
                    item.get("effect_gate", {}).get("reason") == "no_support_bin"
                    for item in row["trace"]
                    if item.get("effect_gate") is not None
                ])) if any(item.get("effect_gate") is not None for item in row["trace"]) else 0.0
                for row in records
            ], 94213),
            "orientation_gate_block_rate": bootstrap_mean([
                float(np.mean([bool(item.get("orientation_gate_blocked", False)) for item in row["trace"]]))
                if row["trace"] else 0.0
                for row in records
            ], 94217),
            "orientation_gate_root_block_rate": bootstrap_mean([
                float(any(bool(item.get("orientation_gate_blocked", False)) for item in row["trace"]))
                for row in records
            ], 94218),
            "predicted_observed_effect_error": bootstrap_mean([
                float(np.mean([
                    item["predicted_observed_effect_error"]
                    for item in row["trace"]
                    if item.get("predicted_observed_effect_error") is not None
                ])) if any(item.get("predicted_observed_effect_error") is not None for item in row["trace"]) else 0.0
                for row in records
            ], 94214),
            "relocation_attempts": bootstrap_mean([
                float(row.get("relocation_attempts", 0)) for row in records
            ], 94220),
            "relocation_root_rate": bootstrap_mean([
                float(row.get("relocation_attempts", 0) > 0) for row in records
            ], 94221),
        },
        "predictor": predictor,
        "contact_geometry_sign_feedback": bool(contact_geometry_sign_feedback),
        "recent_contact_sign_fallback": bool(recent_contact_sign_fallback),
        "online_effect_correction": bool(args.online_effect_correction),
        "uncertainty_gate": bool(uncertainty_gate),
        "fallback_only": bool(fallback_only),
        "adaptive_online_effect": bool(adaptive_online_effect),
        "contact_relocation_on_stagnation": bool(
            getattr(args, "contact_relocation_on_stagnation", False)
        ),
        "online_effect_entries": [
            {"key": key, "count": int(value["count"])}
            for key, value in sorted(online_effects.items())
        ],
        "rows": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--orientation-gate-artifact", type=Path, default=None,
        help="calibration-only orientation_gate.json; enables a fail-closed held-out control ablation",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto selects CUDA when available; cuda never falls back to CPU")
    parser.add_argument("--root-protocol", choices=("terminal_pose", "boundary_contact"), default="terminal_pose",
                        help="terminal_pose preserves the legacy center-radius roots; boundary_contact samples the actual T polygon boundary")
    parser.add_argument("--boundary-action-frame", action="store_true",
                        help="use the nearest polygon-edge normal for contact actions and recovery; boundary_contact enables it automatically")
    parser.add_argument("--min-goal-support-alignment", type=float, default=0.0,
                        help="boundary-root gate: minimum cosine between outward contact normal and opposite goal translation")
    parser.add_argument("--max-initial-angle-error", type=float, default=float("inf"),
                        help="optional declared local orientation-support gate in radians; default keeps all roots")
    parser.add_argument("--legacy-center-action-frame", action="store_true",
                        help="ablation: keep center/vertex action frame while retaining the same root protocol")
    parser.add_argument("--train-roots", type=int, default=48)
    parser.add_argument("--held-out-roots", type=int, default=24)
    parser.add_argument("--sequence-steps", type=int, default=6)
    parser.add_argument("--pulse-steps", type=int, default=4)
    parser.add_argument("--macro-pulse-steps", type=int, default=None,
                        help="pulse length for each half of the fixed two-pulse macro; defaults to --pulse-steps")
    parser.add_argument("--closure-steps", type=int, default=36)
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--contact-chart-alpha", type=float, default=10.0,
                        help="fixed Ridge regularization for the physical contact chart")
    parser.add_argument("--lookahead", type=int, choices=(1, 2), default=2)
    parser.add_argument("--phase-aware-cost", action="store_true",
                        help="use translation-first then rotation-within-tube imagined scoring")
    parser.add_argument("--staged-translation-gate", type=float, default=28.0,
                        help="XY error above which the staged controller uses the goal-conditioned translation ranker")
    parser.add_argument("--phase-shield-tolerance", type=float, default=2.0,
                        help="fixed allowed XY-tube expansion for orientation phases; calibrated before held-out evaluation")
    parser.add_argument("--focused-three-arm", action="store_true",
                        help="run only the predeclared predictor/fallback/local-repair/phase arms; avoids full macro O(N^2) sweep")
    parser.add_argument("--contact-recovery", action="store_true",
                        help="use an explicit state-only reacquisition branch outside the contact tube")
    parser.add_argument("--contact-relocation-on-stagnation", action="store_true",
                        help="after wrong-sign/stagnant contact, route to a declared alternative support side")
    parser.add_argument("--max-contact-relocations", type=int, default=3)
    parser.add_argument("--relocation-stagnation-steps", type=int, default=2)
    parser.add_argument("--min-relocation-improvement", type=float, default=0.01)
    parser.add_argument("--relocation-clearance", type=float, default=70.0)
    parser.add_argument("--relocation-contact-distance", type=float, default=42.0)
    parser.add_argument("--relocation-settle-steps", type=int, default=2)
    parser.add_argument("--relocation-detach-steps", type=int, default=12)
    parser.add_argument("--relocation-detach-magnitude", type=float, default=0.25)
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
    parser.add_argument("--escalate-on-wrong-sign", action="store_true",
                        help="after a wrong observed pulse, use the largest supported opposite-moment pulse to overcome stiction")
    parser.add_argument("--strict-direction-gate", action="store_true",
                        help="ablation: hard-gate contact candidates whose learned angle-direction probability disagrees with the goal")
    parser.add_argument("--online-effect-correction", action="store_true",
                        help="use observed same-region/action-bin transitions for subsequent runtime ranking")
    parser.add_argument("--online-local-jacobian", action="store_true",
                        help="fit a fail-closed local action-effect Jacobian from prior paired observed pulses")
    parser.add_argument("--allow-unpaired-local-effects", action="store_true",
                        help="diagnostic only: allow same-region rank-2 observed effects without strict +/- moment pairing")
    parser.add_argument("--online-jacobian-min-samples", type=int, default=2,
                        help="minimum prior local samples for the observed-effect Jacobian")
    parser.add_argument("--online-effect-neighbors", type=int, default=3,
                        help="maximum nearby observed local effects reused by adaptive ranking")
    parser.add_argument("--online-effect-radius", type=float, default=1.75,
                        help="fixed dimensionless radius for local observed-effect reuse")
    parser.add_argument("--online-paired-radius", type=float, default=1.75,
                        help="support radius used only after a matched opposite-moment pair exists")
    parser.add_argument("--online-task-gain-weight", type=float, default=0.50,
                        help="fixed weight on observed local task-cost gain during adaptive reranking")
    parser.add_argument("--online-min-informative-count", type=int, default=1,
                        help="minimum nonzero-angle observations per local action sign before reuse")
    parser.add_argument("--online-min-effect-angle", type=float, default=0.005,
                        help="minimum absolute observed angle increment in radians for local-effect support")
    parser.add_argument("--paired-effect-probes", action="store_true",
                        help="alternate local probe moment signs to estimate a paired action contrast")
    parser.add_argument("--breakaway-paired-probe", action="store_true",
                        help="ignore the small calibration pulse when forming the first paired probe and use the largest declared support pulse")
    parser.add_argument("--active-probe-max-moment", action="store_true",
                        help="stiction-aware probe ablation: use the largest declared candidate moment")
    parser.add_argument("--identification-pulse-steps", type=int, default=1,
                        help="short pulse length for active identification probes")
    parser.add_argument("--uncertainty-gate", action="store_true",
                        help="execute learned contact actions only inside the training support/uncertainty gate")
    parser.add_argument("--fallback-only", action="store_true",
                        help="contact-only safe baseline: use short inward pulses instead of learned tangent actions")
    parser.add_argument("--min-effect-support", type=int, default=3)
    parser.add_argument("--min-informative-support", type=int, default=2)
    parser.add_argument("--min-effect-sign-consistency", type=float, default=0.75)
    parser.add_argument("--max-effect-angle-std", type=float, default=0.05)
    parser.add_argument("--max-train-effect-error", type=float, default=1.5)
    parser.add_argument("--fallback-magnitude", type=float, default=0.04)
    parser.add_argument("--fallback-pulse-steps", type=int, default=1)
    parser.add_argument("--calibration-pulse-steps", type=int, default=1)
    parser.add_argument("--calibration-angle-threshold", type=float, default=1.0e-4)
    parser.add_argument("--feedback-contact-threshold", type=float, default=24.0,
                        help="polygon-boundary gap within which observed torque-sign feedback is trusted")
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument(
        "--contact-normal-magnitudes", type=float, nargs="+", default=[],
        help="optional expanded contact bank: repeat mirrored tangent pulses at each declared normal magnitude",
    )
    parser.add_argument("--free-recovery-magnitude", type=float, default=0.5)
    parser.add_argument("--terminal-xy-gate", type=float, default=20.0)
    parser.add_argument("--terminal-angle-gate", type=float, default=float(np.pi / 9.0))
    parser.add_argument("--terminal-handoff", action="store_true",
                        help="enable the terminal geometric sign handoff as an explicit ablation")
    parser.add_argument("--terminal-max-moment", action="store_true",
                        help="terminal ablation: use the largest available correctly signed moment pulse to test sticking/breakaway")
    parser.add_argument("--terminal-orientation-ranker", action="store_true",
                        help="terminal ablation: use the learned orientation-only action ranker instead of a geometric sign handoff")
    parser.add_argument("--terminal-observed-effect", action="store_true",
                        help="allow paired observed-effect probing/Jacobian repair during terminal orientation handoff")
    parser.add_argument("--terminal-pulse-steps", type=int, default=1,
                        help="short pulse length used only during terminal orientation handoff")
    parser.add_argument("--contact-lease", action="store_true",
                        help="during near-goal contact orientation, block relocation and enforce an observed XY shield")
    parser.add_argument("--contact-lease-xy-gate", type=float, default=None,
                        help="XY distance at which the contact lease becomes active; defaults to --terminal-xy-gate")
    parser.add_argument("--orientation-xy-budget", type=float, default=0.0,
                        help="allowed predicted XY error increase during a contact-lease orientation pulse")
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+", default=[0.06, 0.10, 0.16])
    parser.add_argument("--goal-translation-magnitudes", type=float, nargs="+", default=[],
                        help="optional task-directed tangent candidates; empty preserves the legacy bank")
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.macro_pulse_steps is None:
        args.macro_pulse_steps = int(args.pulse_steps)
    if int(args.macro_pulse_steps) <= 0:
        raise ValueError("--macro-pulse-steps must be positive")
    if args.contact_lease_xy_gate is None:
        args.contact_lease_xy_gate = float(args.terminal_xy_gate)
    if float(args.orientation_xy_budget) < 0.0:
        raise ValueError("--orientation-xy-budget must be non-negative")
    device = resolve_device(args.device)
    if args.device == "auto" and device.type != "cuda":
        print("CUDA unavailable; running CPU is permitted only after this explicit device check.")
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
    orientation_gate_keys: set[str] | None = None
    orientation_gate_report: dict[str, Any] | None = None
    if args.orientation_gate_artifact is not None:
        orientation_gate_report = json.loads(args.orientation_gate_artifact.read_text())
        if "cross_seed" in orientation_gate_report:
            orientation_gate_keys = {
                str(key) for key in orientation_gate_report["cross_seed"].get(
                    "authorized_intersection", []
                )
            }
        else:
            groups = orientation_gate_report.get("groups", {})
            if isinstance(groups, list):
                groups = {str(item["key"]): item for item in groups}
            orientation_gate_keys = {
                str(key) for key, value in groups.items() if bool(value.get("authorized", False))
            }
        if not orientation_gate_keys:
            raise ValueError("orientation gate artifact authorizes no strata")
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        root_factory = make_boundary_contact_roots if args.root_protocol == "boundary_contact" else make_terminal_roots
        if args.root_protocol == "boundary_contact":
            train_roots = root_factory(
                env, args.train_roots, args.seed + 1000,
                min_goal_support_alignment=args.min_goal_support_alignment,
                max_initial_angle_error=args.max_initial_angle_error,
            )
            held_roots = root_factory(
                env, args.held_out_roots, args.seed + 2000,
                min_goal_support_alignment=args.min_goal_support_alignment,
                max_initial_angle_error=args.max_initial_angle_error,
            )
        else:
            train_roots = root_factory(env, args.train_roots, args.seed + 1000)
            held_roots = root_factory(env, args.held_out_roots, args.seed + 2000)
        root_angle_summary = {
            split: {
                "count": len(roots),
                "mean": float(np.mean([
                    angle_error(root["state"][4], root["goal"][4]) for root in roots
                ])),
                "max": float(np.max([
                    angle_error(root["state"][4], root["goal"][4]) for root in roots
                ])),
                "min": float(np.min([
                    angle_error(root["state"][4], root["goal"][4]) for root in roots
                ])),
            }
            for split, roots in (("train", train_roots), ("held_out", held_roots))
        }
        train_rows = collect_mode_balanced_rows(
            env, train_roots, tuple(args.tangent_magnitudes), args.inward_magnitude,
            args.free_recovery_magnitude, args.pulse_steps, args.sequence_steps,
            args.seed + 3000, tuple(args.goal_translation_magnitudes),
            args.boundary_action_frame, tuple(args.contact_normal_magnitudes),
        )
        held_rows = collect_mode_balanced_rows(
            env, held_roots, tuple(args.tangent_magnitudes), args.inward_magnitude,
            args.free_recovery_magnitude, args.pulse_steps, args.sequence_steps,
            args.seed + 4000, tuple(args.goal_translation_magnitudes),
            args.boundary_action_frame, tuple(args.contact_normal_magnitudes),
        )
        model = fit_mode_aware(
            train_rows, args.seed, args.trees, env=env,
            boundary_action_frame=args.boundary_action_frame,
            contact_chart_alpha=args.contact_chart_alpha,
            device=device,
        )
        # The orientation ranker is deployed inside the macro with
        # ``macro_pulse_steps``.  Collect a separate train-root-only dataset
        # at that exact pulse length so its temporal operator matches runtime.
        orientation_ranker_rows = collect_mode_balanced_rows(
            env, train_roots, tuple(args.tangent_magnitudes), args.inward_magnitude,
            args.free_recovery_magnitude, args.macro_pulse_steps, args.sequence_steps,
            args.seed + 5000, tuple(args.goal_translation_magnitudes),
            args.boundary_action_frame, tuple(args.contact_normal_magnitudes),
        )
        model["orientation_effect_ranker"] = fit_orientation_effect_ranker(
            orientation_ranker_rows, args.seed + 220, args.trees, env=env,
            boundary_action_frame=args.boundary_action_frame,
        )
        model["orientation_delta_ranker"] = fit_orientation_delta_ranker(
            orientation_ranker_rows, args.seed + 240, args.trees, env=env,
            boundary_action_frame=args.boundary_action_frame,
        )
        model["orientation_pair_ranker"] = fit_orientation_pair_ranker(
            orientation_ranker_rows, args.seed + 250, args.trees, env=env,
            boundary_action_frame=args.boundary_action_frame,
        )
        macro_ranker_rows = collect_macro_ranker_rows(
            env, train_roots, tuple(args.tangent_magnitudes), args.inward_magnitude,
            args.free_recovery_magnitude, args.macro_pulse_steps,
            tuple(args.goal_translation_magnitudes), args.boundary_action_frame,
            tuple(args.contact_normal_magnitudes), model=model,
        )
        macro_ranker = fit_macro_ranker(
            macro_ranker_rows, args.seed + 260, args.trees, env,
            args.boundary_action_frame,
        )
        model["effect_gate"] = fit_effect_gate_statistics(
            train_rows, model, env, args.boundary_action_frame,
        )
        if args.focused_three_arm:
            # Dense action banks make the historical all-arm macro sweep grow
            # quadratically in candidates and can spend tens of minutes before
            # writing an artifact.  The confirmatory protocol needs only the
            # predeclared matched arms below; all share the same fitted model,
            # roots, pulse budget and candidate bank.
            held_features = np.asarray([
                lifted_row_features(row, env=env, boundary_action_frame=args.boundary_action_frame)
                for row in held_rows
            ])
            held_labels = np.asarray([
                successor_mode(row["before"], row["after"], env=env) for row in held_rows
            ])
            mode_accuracy = float(np.mean(model["classifier"].predict(held_features) == held_labels))
            frozen_loop = closed_loop(
                env, held_roots, model, args, mode_conditioned=True,
                contact_recovery=args.contact_recovery,
                observed_sign_feedback=args.observed_sign_feedback,
                observed_sign_calibration=args.observed_sign_calibration,
                contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
                recent_contact_sign_fallback=args.recent_contact_sign_fallback,
                strict_contact_sign_gate=args.strict_contact_sign_gate,
                direct_observed_sign_reversal=args.direct_observed_sign_reversal,
            )
            fallback_loop = closed_loop(
                env, held_roots, model, args, mode_conditioned=False,
                contact_recovery=args.contact_recovery,
                observed_sign_feedback=False, observed_sign_calibration=False,
                contact_geometry_sign_feedback=False,
                recent_contact_sign_fallback=False, strict_contact_sign_gate=False,
                direct_observed_sign_reversal=False, fallback_only=True,
            )
            adaptive_loop = closed_loop(
                env, held_roots, model, args, mode_conditioned=True,
                contact_recovery=args.contact_recovery,
                observed_sign_feedback=args.observed_sign_feedback,
                observed_sign_calibration=args.observed_sign_calibration,
                contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
                recent_contact_sign_fallback=args.recent_contact_sign_fallback,
                strict_contact_sign_gate=args.strict_contact_sign_gate,
                direct_observed_sign_reversal=args.direct_observed_sign_reversal,
                uncertainty_gate=True, adaptive_online_effect=True,
            )
            phase_loop = closed_loop_two_pulse_macro(
                env, held_roots, model, args,
                translation_predictor="goal_ranker", orientation_predictor="tree",
                orientation_first=True, translation_xy_only=True,
                reverse_phase_gate=True,
                xy_shield_tolerance=args.phase_shield_tolerance,
                orientation_pair_ranker=model["orientation_pair_ranker"],
            )
            # Literature-motivated sequence arm: fit the joint two-pulse
            # preference model on training roots only, then re-observe between
            # pulses.  The first pulse is selected from the frozen imagined
            # state; the second pulse is scored from the fresh observed state.
            # This is deliberately separate from ``adaptive_local_repair``:
            # it tests sequence-level learned ranking, not a privileged
            # successor oracle and not a post-hoc choice of the best root.
            sequence_ranker_loop = closed_loop_two_pulse_macro(
                env, held_roots, model, args,
                translation_predictor="goal_ranker", orientation_predictor="tree",
                phase_gate=True, macro_ranker=macro_ranker,
            )
            sequence_phase_hybrid_loop = closed_loop_two_pulse_macro(
                env, held_roots, model, args,
                translation_predictor="goal_ranker", orientation_predictor="tree",
                phase_gate=True, macro_ranker=macro_ranker,
                macro_first_only=True,
                orientation_pair_ranker=model["orientation_pair_ranker"],
            )
            risk_averse_sequence_loop = closed_loop_two_pulse_macro(
                env, held_roots, model, args,
                translation_predictor="goal_ranker", orientation_predictor="tree",
                phase_gate=True, macro_ranker=macro_ranker,
                macro_risk_penalty=1.0,
            )
            focused_report = {
                "schema": "pusht_focused_three_arm.v1",
                "protocol": vars(args) | {
                    "resolved_device": str(device),
                    "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "runtime_successor_access": False,
                    "same_held_out_roots": len(held_roots),
                    "focused_reason": "avoid full macro O(N^2) sweep for dense candidate banks",
                },
                "mode_accuracy": mode_accuracy,
                "closed_loop": {
                    "frozen_predictor": frozen_loop,
                    "safe_fallback": fallback_loop,
                    "adaptive_local_repair": adaptive_loop,
                    "phase_repair": phase_loop,
                    "sequence_ranker": sequence_ranker_loop,
                    "sequence_phase_hybrid": sequence_phase_hybrid_loop,
                    "risk_averse_sequence": risk_averse_sequence_loop,
                },
                "claim_boundary": (
                    "Matched privileged state-chart benchmark; it does not establish "
                    "frozen RGB/LeWM causal dynamics or universal transfer."
                ),
            }
            (output / "focused_three_arm.json").write_text(
                json.dumps(focused_report, indent=2, sort_keys=True, default=str)
            )
            print(json.dumps({
                "output": str(output), "mode_accuracy": mode_accuracy,
                "closed_loop": {
                    key: value["metrics"]
                    for key, value in focused_report["closed_loop"].items()
                },
            }, indent=2))
            return
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
        uncertainty_loop = closed_loop(
            env, held_roots, model, args, mode_conditioned=True,
            contact_recovery=args.contact_recovery,
            observed_sign_feedback=args.observed_sign_feedback,
            observed_sign_calibration=args.observed_sign_calibration,
            contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
            recent_contact_sign_fallback=args.recent_contact_sign_fallback,
            strict_contact_sign_gate=args.strict_contact_sign_gate,
            direct_observed_sign_reversal=args.direct_observed_sign_reversal,
            uncertainty_gate=True,
        )
        fallback_loop = closed_loop(
            env, held_roots, model, args, mode_conditioned=False,
            contact_recovery=args.contact_recovery,
            observed_sign_feedback=False,
            observed_sign_calibration=False,
            contact_geometry_sign_feedback=False,
            recent_contact_sign_fallback=False,
            strict_contact_sign_gate=False,
            direct_observed_sign_reversal=False,
            fallback_only=True,
        )
        adaptive_loop = closed_loop(
            env, held_roots, model, args, mode_conditioned=True,
            contact_recovery=args.contact_recovery,
            observed_sign_feedback=args.observed_sign_feedback,
            observed_sign_calibration=args.observed_sign_calibration,
            contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
            recent_contact_sign_fallback=args.recent_contact_sign_fallback,
            strict_contact_sign_gate=args.strict_contact_sign_gate,
            direct_observed_sign_reversal=args.direct_observed_sign_reversal,
            uncertainty_gate=True,
            adaptive_online_effect=True,
        )
        staged_loop = closed_loop(
            env, held_roots, model, args, mode_conditioned=True,
            contact_recovery=args.contact_recovery,
            observed_sign_feedback=args.observed_sign_feedback,
            observed_sign_calibration=args.observed_sign_calibration,
            contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
            recent_contact_sign_fallback=args.recent_contact_sign_fallback,
            strict_contact_sign_gate=args.strict_contact_sign_gate,
            direct_observed_sign_reversal=args.direct_observed_sign_reversal,
            predictor="staged",
        )
        macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="tree", orientation_predictor="tree",
        )
        staged_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
        )
        staged_feedback_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            observed_sign_feedback=True,
        )
        staged_phase_gate_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            phase_gate=True,
        )
        shielded_phase_gate_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            phase_gate=True, xy_shield_tolerance=args.phase_shield_tolerance,
        )
        orientation_then_translation_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            orientation_first=True,
        )
        orientation_ranker_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            orientation_first=True,
            orientation_effect_ranker=model["orientation_effect_ranker"],
        )
        orientation_delta_ranker_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            orientation_first=True,
            orientation_delta_ranker=model["orientation_delta_ranker"],
        )
        orientation_delta_ranker_shielded_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            orientation_first=True, xy_shield_tolerance=args.phase_shield_tolerance,
            orientation_delta_ranker=model["orientation_delta_ranker"],
        )
        orientation_dual_ranker_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            orientation_first=True,
            orientation_effect_ranker=model["orientation_effect_ranker"],
            orientation_delta_ranker=model["orientation_delta_ranker"],
        )
        orientation_pair_ranker_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            orientation_first=True,
            orientation_pair_ranker=model["orientation_pair_ranker"],
        )
        orientation_pair_xy_translation_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            orientation_first=True, translation_xy_only=True,
            orientation_pair_ranker=model["orientation_pair_ranker"],
        )
        orientation_pair_reverse_phase_gate_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            orientation_first=True, translation_xy_only=True, reverse_phase_gate=True,
            orientation_pair_ranker=model["orientation_pair_ranker"],
        )
        orientation_pair_reverse_phase_gate_shielded_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            orientation_first=True, translation_xy_only=True, reverse_phase_gate=True,
            xy_shield_tolerance=args.phase_shield_tolerance,
            orientation_pair_ranker=model["orientation_pair_ranker"],
        )
        joint_macro_loop = closed_loop_two_pulse_macro(
            env, held_roots, model, args,
            translation_predictor="goal_ranker", orientation_predictor="tree",
            phase_gate=True, macro_ranker=macro_ranker,
        )
        orientation_gated_loop = closed_loop(
            env, held_roots, model, args, mode_conditioned=True,
            contact_recovery=args.contact_recovery,
            observed_sign_feedback=args.observed_sign_feedback,
            observed_sign_calibration=args.observed_sign_calibration,
            contact_geometry_sign_feedback=args.contact_geometry_sign_feedback,
            recent_contact_sign_fallback=args.recent_contact_sign_fallback,
            strict_contact_sign_gate=args.strict_contact_sign_gate,
            direct_observed_sign_reversal=args.direct_observed_sign_reversal,
            uncertainty_gate=True,
            orientation_gate_keys=orientation_gate_keys,
        ) if orientation_gate_keys is not None else None
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "requested_device": args.device,
            "resolved_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "root_protocol": args.root_protocol,
            "boundary_action_frame": bool(args.boundary_action_frame),
            "min_goal_support_alignment": float(args.min_goal_support_alignment),
            "max_initial_angle_error": float(args.max_initial_angle_error),
            "root_angle_error_summary": root_angle_summary,
            "legacy_center_action_frame": bool(args.legacy_center_action_frame),
            "train_roots": args.train_roots, "held_out_roots": args.held_out_roots,
            "train_seed": args.seed + 1000, "held_out_seed": args.seed + 2000,
            "candidate_bank": "shared per root; declared action-frame inward+tangent pulses plus one explicit outward separation probe in contact mode",
            "runtime_successor_access": False,
            "contact_recovery": bool(args.contact_recovery),
            "contact_relocation_on_stagnation": bool(args.contact_relocation_on_stagnation),
            "max_contact_relocations": int(args.max_contact_relocations),
            "relocation_stagnation_steps": int(args.relocation_stagnation_steps),
            "min_relocation_improvement": float(args.min_relocation_improvement),
            "relocation_clearance": float(args.relocation_clearance),
            "relocation_contact_distance": float(args.relocation_contact_distance),
            "relocation_settle_steps": int(args.relocation_settle_steps),
            "relocation_detach_steps": int(args.relocation_detach_steps),
            "relocation_detach_magnitude": float(args.relocation_detach_magnitude),
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
            "online_effect_neighborhood": {
                "neighbors": int(args.online_effect_neighbors),
                "radius": float(args.online_effect_radius),
                "paired_radius": float(args.online_paired_radius),
                "task_gain_weight": float(args.online_task_gain_weight),
                "min_informative_count": int(args.online_min_informative_count),
                "min_effect_angle": float(args.online_min_effect_angle),
                "paired_effect_probes": bool(args.paired_effect_probes),
                "active_probe_max_moment": bool(args.active_probe_max_moment),
                "identification_pulse_steps": int(args.identification_pulse_steps),
                "chart": "same_polygon_edge_and_moment_sign; fixed normalized edge/action coordinates",
            },
            "uncertainty_gate": True,
            "fallback_only_control": True,
            "effect_gate_thresholds": {
                "min_support": args.min_effect_support,
                "min_informative_support": args.min_informative_support,
                "min_sign_consistency": args.min_effect_sign_consistency,
                "max_angle_std": args.max_effect_angle_std,
                "max_train_effect_error": args.max_train_effect_error,
                "fallback_magnitude": args.fallback_magnitude,
                "fallback_pulse_steps": args.fallback_pulse_steps,
            },
            "temporal_event_features": True,
            "previous_observed_angle_delta": True,
            "goal_conditioned_ranker": "training-only direct one-step pose-cost ablation",
            "orientation_effect_ranker": "training-root-only direct one-step post-pulse angle-error ranker",
            "orientation_delta_ranker": "training-root-only signed after-minus-before angle-delta ranker; goal residual reintroduced only at runtime",
            "orientation_dual_ranker": "fixed equal-weight mean of the absolute-error and signed-delta angular predictions; no held-out weighting",
            "orientation_pair_ranker": "training-root-only ordered mirrored-action angular contrast; common component from the base transition predictor",
            "orientation_ranker_training_pulse_steps": int(args.macro_pulse_steps),
            "orientation_delta_ranker_training_pulse_steps": int(args.macro_pulse_steps),
            "calibration_pulse_steps": args.calibration_pulse_steps,
            "calibration_angle_threshold": args.calibration_angle_threshold,
            "terminal_handoff": bool(args.terminal_handoff),
            "terminal_max_moment": bool(args.terminal_max_moment),
            "terminal_orientation_ranker": bool(args.terminal_orientation_ranker),
            "terminal_pulse_steps": int(args.terminal_pulse_steps),
            "terminal_xy_gate": args.terminal_xy_gate,
            "terminal_angle_gate": args.terminal_angle_gate,
            "feedback_contact_threshold": args.feedback_contact_threshold,
            "mode_labels": SUCCESSOR_NAMES,
            "current_mode_thresholds_px": {"boundary_contact": 24.0, "boundary_approach": 70.0},
            "successor_thresholds": {"boundary_start_free": 70.0, "boundary_separation": 70.0, "block_motion": 1.0},
            "contact_loss_definition": (
                "boundary_distance_after > 70 px for boundary_contact roots; "
                "center_distance_after > 80 px retained only as a legacy diagnostic"
            ),
            "lookahead": args.lookahead,
            "phase_aware_cost": bool(args.phase_aware_cost),
            "staged_translation_gate": float(args.staged_translation_gate),
            "pulse_steps": args.pulse_steps,
            "macro_pulse_steps": int(args.macro_pulse_steps),
            "macro_ranker_training_pulse_steps": int(args.macro_pulse_steps),
            "macro_steps": max(1, (int(args.closure_steps) * int(args.pulse_steps)) // max(2 * int(args.macro_pulse_steps), 1)),
            "macro_protocol": "translation pulse -> fresh observation -> orientation pulse",
            "reverse_macro_protocol": "orientation pulse -> fresh observation -> translation pulse",
            "macro_budget": "two macro pulses; total env steps matched to primitive budget; no successor access",
            "macro_sign_feedback": "optional second-pulse moment-sign gate from first-pulse observed angle direction",
            "macro_phase_gate": "optional existing 28 px XY gate; second pulse remains translation until inside tube",
            "xy_shield_tolerance": float(args.phase_shield_tolerance),
            "joint_macro_ranker": "training-root two-pulse final-pose ranker; training and runtime use macro_pulse_steps; runtime uses learned first-step imagination then fresh observation",
            "sequence_steps": args.sequence_steps,
            "closure_steps": args.closure_steps,
            "tangent_magnitudes": list(args.tangent_magnitudes),
            "inward_magnitude": float(args.inward_magnitude),
            "contact_normal_magnitudes": list(args.contact_normal_magnitudes),
            "free_recovery_magnitude": args.free_recovery_magnitude,
            "goal_translation_magnitudes": list(args.goal_translation_magnitudes),
            "orientation_gate_artifact": None if args.orientation_gate_artifact is None else str(args.orientation_gate_artifact.resolve()),
            "orientation_gate_keys": None if orientation_gate_keys is None else sorted(orientation_gate_keys),
            "orientation_gate_calibration_outcome_used": False,
        },
        "training": {"rows": len(train_rows), "orientation_ranker_rows": len(orientation_ranker_rows), "orientation_ranker_pulse_steps": int(args.macro_pulse_steps), "orientation_delta_ranker_rows": len(orientation_ranker_rows), "orientation_delta_ranker_pulse_steps": int(args.macro_pulse_steps), "macro_ranker_rows": len(macro_ranker_rows), "roots": len(train_roots), "mode_counts": model["counts"]},
        "contact_chart": {
            "feature_dim": model["contact_chart"]["feature_dim"],
            "alpha": model["contact_chart"]["alpha"],
            "mode_counts": model["contact_chart"]["counts"],
            "training_only": True,
            "geometry_source": model["contact_chart"]["geometry_source"],
        },
        "effect_gate": {
            "bins": len(model["effect_gate"]),
            "support_total": int(sum(item["support"] for item in model["effect_gate"].values())),
            "support_bins": {
                key: value for key, value in sorted(model["effect_gate"].items())
            },
            "training_only": True,
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
            "uncertainty_gated": uncertainty_loop,
            "safe_fallback": fallback_loop,
            "adaptive_online": adaptive_loop,
            "staged_translation_then_orientation": staged_loop,
            "two_pulse_macro": macro_loop,
            "staged_two_pulse_macro": staged_macro_loop,
            "staged_feedback_two_pulse_macro": staged_feedback_macro_loop,
            "staged_phase_gate_two_pulse_macro": staged_phase_gate_macro_loop,
            "shielded_phase_gate_two_pulse_macro": shielded_phase_gate_macro_loop,
            "orientation_then_translation_two_pulse_macro": orientation_then_translation_macro_loop,
            "orientation_ranker_two_pulse_macro": orientation_ranker_macro_loop,
            "orientation_delta_ranker_two_pulse_macro": orientation_delta_ranker_macro_loop,
            "orientation_delta_ranker_shielded_two_pulse_macro": orientation_delta_ranker_shielded_macro_loop,
            "orientation_dual_ranker_two_pulse_macro": orientation_dual_ranker_macro_loop,
            "orientation_pair_ranker_two_pulse_macro": orientation_pair_ranker_macro_loop,
            "orientation_pair_xy_translation_two_pulse_macro": orientation_pair_xy_translation_macro_loop,
            "orientation_pair_reverse_phase_gate_two_pulse_macro": orientation_pair_reverse_phase_gate_macro_loop,
            "orientation_pair_reverse_phase_gate_shielded_two_pulse_macro": orientation_pair_reverse_phase_gate_shielded_macro_loop,
            "joint_two_pulse_macro": joint_macro_loop,
            "orientation_gated": orientation_gated_loop,
        },
        "sanity_checks": {
            "finite": bool(np.isfinite(held_features).all()),
            "shared_candidate_bank": True,
            "root_disjoint": True,
            "runtime_successor_access": False,
            "orientation_gate_root_disjoint_from_control_roots": True,
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
        ("uncertainty_gated", uncertainty_loop),
        ("safe_fallback", fallback_loop),
        ("adaptive_online", adaptive_loop),
        ("staged_translation_then_orientation", staged_loop),
        ("two_pulse_macro", macro_loop),
        ("staged_two_pulse_macro", staged_macro_loop),
        ("staged_feedback_two_pulse_macro", staged_feedback_macro_loop),
        ("staged_phase_gate_two_pulse_macro", staged_phase_gate_macro_loop),
        ("shielded_phase_gate_two_pulse_macro", shielded_phase_gate_macro_loop),
        ("orientation_then_translation_two_pulse_macro", orientation_then_translation_macro_loop),
        ("orientation_ranker_two_pulse_macro", orientation_ranker_macro_loop),
        ("orientation_delta_ranker_two_pulse_macro", orientation_delta_ranker_macro_loop),
        ("orientation_delta_ranker_shielded_two_pulse_macro", orientation_delta_ranker_shielded_macro_loop),
        ("orientation_dual_ranker_two_pulse_macro", orientation_dual_ranker_macro_loop),
        ("orientation_pair_ranker_two_pulse_macro", orientation_pair_ranker_macro_loop),
        ("orientation_pair_xy_translation_two_pulse_macro", orientation_pair_xy_translation_macro_loop),
        ("orientation_pair_reverse_phase_gate_two_pulse_macro", orientation_pair_reverse_phase_gate_macro_loop),
        ("orientation_pair_reverse_phase_gate_shielded_two_pulse_macro", orientation_pair_reverse_phase_gate_shielded_macro_loop),
        ("joint_two_pulse_macro", joint_macro_loop),
        ("orientation_gated", orientation_gated_loop),
    ):
        if loop is None:
            continue
        ranking_value = ranking["metrics"].get(name)
        lines.append(
            "| %s | %s | %.3f | %.3f | %.3f | %.3f |" % (
                name,
                "n/a" if ranking_value is None else "%.4f" % ranking_value["estimate"],
                loop["metrics"]["pose_success"]["estimate"],
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
    print(json.dumps({"output": str(output), "mode_accuracy": mode_accuracy, "ranking": ranking["metrics"], "closed_loop": {"global": global_loop["metrics"], "mode_aware": mode_loop["metrics"], "goal_ranker": goal_ranker_loop["metrics"], "uncertainty_gated": uncertainty_loop["metrics"], "safe_fallback": fallback_loop["metrics"], "adaptive_online": adaptive_loop["metrics"], "staged": staged_loop["metrics"], "two_pulse_macro": macro_loop["metrics"], "staged_two_pulse_macro": staged_macro_loop["metrics"], "staged_feedback_two_pulse_macro": staged_feedback_macro_loop["metrics"], "staged_phase_gate_two_pulse_macro": staged_phase_gate_macro_loop["metrics"], "shielded_phase_gate_two_pulse_macro": shielded_phase_gate_macro_loop["metrics"], "orientation_then_translation_two_pulse_macro": orientation_then_translation_macro_loop["metrics"], "orientation_ranker_two_pulse_macro": orientation_ranker_macro_loop["metrics"], "orientation_delta_ranker_two_pulse_macro": orientation_delta_ranker_macro_loop["metrics"], "orientation_delta_ranker_shielded_two_pulse_macro": orientation_delta_ranker_shielded_macro_loop["metrics"], "orientation_dual_ranker_two_pulse_macro": orientation_dual_ranker_macro_loop["metrics"], "orientation_pair_ranker_two_pulse_macro": orientation_pair_ranker_macro_loop["metrics"], "orientation_pair_xy_translation_two_pulse_macro": orientation_pair_xy_translation_macro_loop["metrics"], "orientation_pair_reverse_phase_gate_two_pulse_macro": orientation_pair_reverse_phase_gate_macro_loop["metrics"], "orientation_pair_reverse_phase_gate_shielded_two_pulse_macro": orientation_pair_reverse_phase_gate_shielded_macro_loop["metrics"], "joint_two_pulse_macro": joint_macro_loop["metrics"]}}, indent=2))


if __name__ == "__main__":
    main()
