"""Audit local PushT orientation response under matched moment probes.

This is a privileged simulator-state mechanism audit, not a visual planner
evaluation.  Each root is a genuine contact state from a behavior trajectory.
The same root and the same nearest polygon edge are probed with a zero pulse
and two equal-norm pulses whose tangential components have opposite moment
signs.  The zero pulse estimates common angular drift; subtracting it gives an
action-dependent response.  The paired contrast and its antisymmetry residual
are the primary orientation-identifiability quantities.

The Gym/Pymunk simulator is CPU-bound.  All aggregate response statistics and
bootstrap intervals are nevertheless computed on an explicitly validated CUDA
device and the artifact records both devices.  This prevents a CPU fallback
from being silently presented as a GPU run while keeping the physical probe
semantics unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
    mode_for_state,
    unit,
)
from experiments.run_pusht_receding_block_pose_planner import make_state_only_env  # noqa: E402
from experiments.run_pusht_signed_torque_audit import (  # noqa: E402
    physical_contact_geometry,
    wrap_angle,
)

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--pool-transitions", type=int, default=2048)
    parser.add_argument(
        "--root-protocol",
        choices=("trajectory", "random_boundary", "edge_boundary"),
        default="trajectory",
        help="trajectory roots preserve the legacy audit; random_boundary creates independent seed-conditioned roots; edge_boundary explicitly covers polygon edge/fraction strata",
    )
    parser.add_argument("--states-per-bin", type=int, default=2)
    parser.add_argument("--fraction-bins", type=int, default=4)
    parser.add_argument("--max-roots", type=int, default=32)
    parser.add_argument("--max-contact-gap", type=float, default=30.0)
    parser.add_argument("--inward-magnitude", type=float, default=0.18)
    parser.add_argument("--tangent-magnitude", type=float, default=0.55)
    parser.add_argument("--pulse-steps", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261014)
    return parser.parse_args()


def require_cuda() -> torch.device:
    """Fail closed unless the actual tensor path can initialize on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("orientation audit requires CUDA; torch.cuda.is_available() is false")
    torch.cuda.init()
    device = torch.device("cuda")
    probe = torch.ones((4, 4), device=device)
    torch.cuda.synchronize()
    if float(probe.sum().item()) != 16.0:
        raise RuntimeError("CUDA allocation probe returned an unexpected value")
    return device


def edge_chart(env: gym.Env, state: np.ndarray) -> dict[str, Any]:
    """Return the actual nearest transformed edge and a matched pulse chart."""
    state = np.asarray(state, dtype=np.float64)
    geometry = physical_contact_geometry(env, state)
    shape_index = int(geometry["shape_index"])
    edge_index = int(geometry["edge_index"])
    shape = list(getattr(env.unwrapped.block, "shapes", ()))[shape_index]
    vertices = [
        np.asarray([float(vertex.x), float(vertex.y)], dtype=np.float64)
        for vertex in shape.get_vertices()
    ]
    theta = float(state[4])
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    block = state[2:4]
    point_a = block + rotation @ vertices[edge_index]
    point_b = block + rotation @ vertices[(edge_index + 1) % len(vertices)]
    edge = point_b - point_a
    edge_sq = max(float(np.dot(edge, edge)), 1.0e-12)
    fraction = float(np.clip(np.dot(np.asarray(geometry["contact_point_world"]) - point_a, edge) / edge_sq, 0.0, 1.0))
    contact_radius = np.asarray(geometry["contact_point_world"], dtype=np.float64) - block
    radial_tangent = unit(np.asarray([-contact_radius[1], contact_radius[0]], dtype=np.float64))
    inward = unit(np.asarray(geometry["contact_point_world"], dtype=np.float64) - state[:2])
    return {
        "shape_index": shape_index,
        "edge_index": edge_index,
        "edge_id": f"{shape_index}:{edge_index}",
        "edge_fraction": fraction,
        "contact_gap": float(geometry["distance"]),
        "contact_point_world": np.asarray(geometry["contact_point_world"], dtype=np.float64),
        "contact_radius": contact_radius,
        "inward": inward,
        "radial_tangent": radial_tangent,
    }


def matched_actions(chart: dict[str, Any], inward_magnitude: float, tangent_magnitude: float) -> dict[str, Any]:
    inward = np.asarray(chart["inward"], dtype=np.float64)
    # Build the pair in the block-centered radial frame.  The common component
    # is directed from the contact point toward the block center; the +/-
    # component is exactly orthogonal to the lever arm.  Therefore the two
    # pulses have equal norm and exactly opposite geometric moments even on an
    # asymmetric polygon edge.  The physical contact-normal chart remains in
    # the record so mode/contact retention is audited separately.
    radial_inward = -unit(np.asarray(chart["contact_radius"], dtype=np.float64))
    tangent = np.asarray([-radial_inward[1], radial_inward[0]], dtype=np.float64)
    plus = inward_magnitude * radial_inward + tangent_magnitude * tangent
    minus = inward_magnitude * radial_inward - tangent_magnitude * tangent
    zero = np.zeros(2, dtype=np.float64)
    block_to_contact = np.asarray(chart["contact_radius"], dtype=np.float64)
    # The action moment is evaluated about the block center; the sign is the
    # physical label used by the audit, not an inferred goal direction.
    plus_moment = float(block_to_contact[0] * plus[1] - block_to_contact[1] * plus[0])
    minus_moment = float(block_to_contact[0] * minus[1] - block_to_contact[1] * minus[0])
    return {
        "zero": zero,
        "plus": plus,
        "minus": minus,
        "moment_plus": plus_moment,
        "moment_minus": minus_moment,
        "pair_valid": bool(
            abs(plus_moment) > 1.0e-8
            and abs(minus_moment) > 1.0e-8
            and np.sign(plus_moment) == -np.sign(minus_moment)
        ),
    }


def probe(env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray, seed: int, pulse_steps: int) -> dict[str, Any]:
    observation, _ = env.reset(
        seed=int(seed),
        options={"state": np.asarray(state, dtype=np.float64).copy(), "goal_state": np.asarray(goal, dtype=np.float64).copy()},
    )
    before = np.asarray(observation["state"], dtype=np.float64).copy()
    terminated = truncated = False
    for _ in range(int(pulse_steps)):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            break
    after = np.asarray(observation["state"], dtype=np.float64).copy()
    return {
        "before_angle": float(before[4]),
        "after_angle": float(after[4]),
        "signed_angle_delta": float(wrap_angle(after[4] - before[4])),
        "before_xy": [float(value) for value in before[2:4]],
        "after_xy": [float(value) for value in after[2:4]],
        "mode_before": str(mode_for_state(before)),
        "mode_after": str(mode_for_state(after)),
        "contact_gap_after": float(np.linalg.norm(after[:2] - after[2:4])),
        "terminated": bool(terminated or truncated),
    }


def independent_boundary_candidates(env: gym.Env, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Generate valid contact roots directly from seed-conditioned geometry.

    This protocol does not claim that roots are on-policy trajectory states.
    It exists to test cross-seed independence without relying on the highly
    repetitive behavior-trajectory collector.
    """
    candidates: list[dict[str, Any]] = []
    target = max(int(args.max_roots) * 3, int(args.states_per_bin) * int(args.fraction_bins) * 8)
    rng = np.random.default_rng(int(args.seed) + 910000)
    # The geometry helper reads the initialized polygon shapes.  Initialize
    # once before constructing synthetic-but-valid contact roots.
    env.reset(seed=int(args.seed))
    for source_index in range(target * 4):
        state = np.zeros(7, dtype=np.float64)
        state[2:4] = rng.uniform(90.0, 410.0, size=2)
        state[4] = float(rng.uniform(-np.pi, np.pi))
        normal_angle = float(rng.uniform(-np.pi, np.pi))
        normal = np.asarray([np.cos(normal_angle), np.sin(normal_angle)], dtype=np.float64)
        state[:2] = np.clip(state[2:4] + 64.0 * normal, 24.0, 488.0)
        geometry = physical_contact_geometry(env, state)
        projected = state.copy()
        projected[:2] = np.asarray(geometry["contact_point_world"], dtype=np.float64) + 16.0 * np.asarray(geometry["normal_world"], dtype=np.float64)
        if np.any(projected[:2] < 16.0) or np.any(projected[:2] > 496.0):
            continue
        goal = projected.copy()
        goal[2:4] = np.clip(projected[2:4] + rng.uniform(-24.0, 24.0, size=2), 70.0, 442.0)
        goal[4] = float(wrap_angle(projected[4] + rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.2)))
        goal[5:] = 0.0
        observation, _ = env.reset(
            seed=int(args.seed + 500000 + source_index),
            options={"state": projected.copy(), "goal_state": goal.copy()},
        )
        accepted = np.asarray(observation["state"], dtype=np.float64).copy()
        chart = edge_chart(env, accepted)
        if str(mode_for_state(accepted)) != "contact" or chart["contact_gap"] > float(args.max_contact_gap):
            continue
        candidates.append({
            "source_index": int(source_index),
            "state": accepted,
            "goal": goal,
            "chart": chart,
        })
        if len(candidates) >= target:
            break
    if len(candidates) < 2:
        raise RuntimeError(f"insufficient independent boundary roots: {len(candidates)}")
    return candidates


def edge_boundary_candidates(env: gym.Env, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Generate seed-conditioned roots by targeting every polygon edge.

    ``random_boundary`` samples a pusher direction and then asks the nearest
    edge to define the stratum.  For the asymmetric T this is a biased
    rejection sampler: many valid states collapse onto the same arm edge.
    Here the edge and contact fraction are sampled first, then the pusher is
    placed along the selected edge's outward normal.  The accepted state is
    still checked after the environment reset, so the record contains the
    actual nearest edge rather than the requested one.
    """
    candidates: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed) + 920000)
    env.reset(seed=int(args.seed))
    block = env.unwrapped.block
    targets: list[tuple[int, int, int, int]] = []
    for shape_index, shape in enumerate(list(getattr(block, "shapes", ()))):
        vertices = [
            np.asarray([float(vertex.x), float(vertex.y)], dtype=np.float64)
            for vertex in shape.get_vertices()
        ]
        for edge_index in range(len(vertices)):
            for fraction_bin in range(int(args.fraction_bins)):
                for replicate in range(int(args.states_per_bin)):
                    targets.append((shape_index, edge_index, fraction_bin, replicate))
    rng.shuffle(targets)
    target_count = min(len(targets), max(int(args.max_roots) * 2, len(targets)))
    max_attempts = max(64, target_count * 40)
    retry_counts: dict[tuple[int, int, int], int] = {}
    for source_index in range(max_attempts):
        if not targets:
            break
        shape_index, edge_index, fraction_bin, replicate = targets.pop(0)
        shape = list(getattr(block, "shapes", ()))[shape_index]
        vertices = [
            np.asarray([float(vertex.x), float(vertex.y)], dtype=np.float64)
            for vertex in shape.get_vertices()
        ]
        local_a = vertices[edge_index]
        local_b = vertices[(edge_index + 1) % len(vertices)]
        edge = local_b - local_a
        edge_length = max(float(np.linalg.norm(edge)), 1.0e-8)
        tangent = edge / edge_length
        midpoint = local_a + (float(fraction_bin) + 0.5) / float(args.fraction_bins) * edge
        # Pick the normal pointing away from the shape centroid.  This makes
        # the requested point a true exterior contact rather than an interior
        # point, independent of polygon winding.
        centroid = np.mean(np.asarray(vertices), axis=0)
        normal_a = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        normal_b = -normal_a
        outward = normal_a if np.dot(midpoint + normal_a - centroid, normal_a) > np.dot(midpoint + normal_b - centroid, normal_b) else normal_b
        block_xy = rng.uniform(96.0, 416.0, size=2)
        theta = float(rng.uniform(-np.pi, np.pi))
        rotation = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        gap = float(rng.uniform(12.0, 20.0))
        state = np.zeros(7, dtype=np.float64)
        state[2:4] = block_xy
        state[4] = theta
        state[:2] = block_xy + rotation @ (midpoint + gap * outward)
        if np.any(state[:2] < 24.0) or np.any(state[:2] > 488.0):
            continue
        goal = state.copy()
        goal[2:4] = np.clip(block_xy + rng.uniform(-24.0, 24.0, size=2), 70.0, 442.0)
        goal[4] = float(wrap_angle(theta + rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.2)))
        goal[5:] = 0.0
        observation, _ = env.reset(
            seed=int(args.seed + 700000 + source_index),
            options={"state": state.copy(), "goal_state": goal.copy()},
        )
        accepted = np.asarray(observation["state"], dtype=np.float64).copy()
        chart = edge_chart(env, accepted)
        actual_bin = min(int(args.fraction_bins) - 1, int(chart["edge_fraction"] * int(args.fraction_bins)))
        requested_key = f"{shape_index}:{edge_index}__bin{fraction_bin}"
        actual_key = f"{chart['edge_id']}__bin{actual_bin}"
        # Center-distance mode labels are not a valid contact test for the
        # long T stem.  The selected edge and continuous geometric gap are
        # the protocol's acceptance criteria; the coarse mode is reported,
        # never used to censor a requested edge.
        if chart["contact_gap"] > float(args.max_contact_gap):
            target_key = (shape_index, edge_index, fraction_bin)
            retry_counts[target_key] = retry_counts.get(target_key, 0) + 1
            if retry_counts[target_key] < 20:
                targets.append((shape_index, edge_index, fraction_bin, replicate))
            continue
        if actual_key != requested_key:
            target_key = (shape_index, edge_index, fraction_bin)
            retry_counts[target_key] = retry_counts.get(target_key, 0) + 1
            if retry_counts[target_key] < 20:
                targets.append((shape_index, edge_index, fraction_bin, replicate))
            continue
        candidates.append({"source_index": int(source_index), "state": accepted, "goal": goal, "chart": chart})
        if len(candidates) >= int(args.max_roots):
            break
    if len(candidates) < 2:
        raise RuntimeError(f"insufficient edge-conditioned boundary roots: {len(candidates)}")
    return candidates


def collect_roots(env: gym.Env, args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if args.root_protocol == "random_boundary":
        candidates = independent_boundary_candidates(env, args)
    elif args.root_protocol == "edge_boundary":
        candidates = edge_boundary_candidates(env, args)
    else:
        pool = collect_trajectory_data(env, int(args.pool_transitions), int(args.seed))
        candidates = [
            {"source_index": int(index), "state": np.asarray(state, dtype=np.float64).copy(), "goal": None}
            for index, (state, mode) in enumerate(zip(pool["states"], pool["modes"]))
            if str(mode) == "contact"
        ]
    bins: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        source_index = int(candidate["source_index"])
        state = np.asarray(candidate["state"], dtype=np.float64)
        chart = edge_chart(env, state)
        if chart["contact_gap"] > float(args.max_contact_gap):
            continue
        fraction_bin = min(args.fraction_bins - 1, int(chart["edge_fraction"] * args.fraction_bins))
        key = f"{chart['edge_id']}__bin{fraction_bin}"
        values = bins.setdefault(key, [])
        if len(values) < int(args.states_per_bin):
            goal = candidate.get("goal")
            if goal is None:
                goal = np.asarray(state, dtype=np.float64).copy()
                goal[4] = float(wrap_angle(goal[4] + 0.8))
                goal[5:] = 0.0
            values.append({
                "source_index": source_index,
                "state": np.asarray(state, dtype=np.float64).copy(),
                "goal": np.asarray(goal, dtype=np.float64).copy(),
                "chart": chart,
            })
    roots: list[dict[str, Any]] = []
    for key in sorted(bins):
        roots.extend(bins[key])
    if len(roots) > int(args.max_roots):
        rng = np.random.default_rng(int(args.seed) + 1)
        selected = rng.choice(len(roots), size=int(args.max_roots), replace=False)
        roots = [roots[int(index)] for index in sorted(selected)]
    if len(roots) < 2:
        raise RuntimeError(f"insufficient contact roots for orientation audit: {len(roots)}")
    return roots


def root_record(env: gym.Env, root: dict[str, Any], args: argparse.Namespace, root_id: int) -> dict[str, Any]:
    state = np.asarray(root["state"], dtype=np.float64)
    chart = edge_chart(env, state)
    actions = matched_actions(chart, float(args.inward_magnitude), float(args.tangent_magnitude))
    probes = {
        name: probe(env, state, root["goal"], action, int(args.seed + root_id * 100 + offset), int(args.pulse_steps))
        for offset, (name, action) in enumerate((
            ("zero", actions["zero"]), ("plus", actions["plus"]), ("minus", actions["minus"]),
        ))
    }
    delta_zero = float(probes["zero"]["signed_angle_delta"])
    effect_plus = float(wrap_angle(probes["plus"]["signed_angle_delta"] - delta_zero))
    effect_minus = float(wrap_angle(probes["minus"]["signed_angle_delta"] - delta_zero))
    contrast = float(wrap_angle(effect_plus - effect_minus) / 2.0)
    odd_rmse = float(math.sqrt(((effect_plus - contrast) ** 2 + (effect_minus + contrast) ** 2) / 2.0))
    moment_difference = float(actions["moment_plus"] - actions["moment_minus"])
    sign_consistent = bool(
        actions["pair_valid"]
        and abs(contrast) > 1.0e-8
        and np.sign(contrast) == np.sign(moment_difference)
    )
    return {
        "root_id": int(root_id),
        "source_index": int(root["source_index"]),
        "root_state_digest": hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest(),
        "edge_id": str(chart["edge_id"]),
        "edge_fraction": float(chart["edge_fraction"]),
        "fraction_bin": int(min(args.fraction_bins - 1, int(chart["edge_fraction"] * args.fraction_bins))),
        "contact_gap": float(chart["contact_gap"]),
        "mode_before": str(mode_for_state(state)),
        "action_norms": {name: float(np.linalg.norm(value)) for name, value in actions.items() if name in {"zero", "plus", "minus"}},
        "moment_plus": float(actions["moment_plus"]),
        "moment_minus": float(actions["moment_minus"]),
        "pair_valid": bool(actions["pair_valid"]),
        "probe": probes,
        "common_drift": delta_zero,
        "effect_plus": effect_plus,
        "effect_minus": effect_minus,
        "paired_contrast": contrast,
        "paired_separation": float(abs(effect_plus - effect_minus)),
        "paired_antisymmetry_rmse": odd_rmse,
        "common_drift_abs": float(abs(delta_zero)),
        "effect_to_drift_ratio": float(abs(contrast) / (abs(delta_zero) + 1.0e-6)),
        "sign_consistent": sign_consistent,
        "successor_mode_agreement_pm": bool(probes["plus"]["mode_after"] == probes["minus"]["mode_after"]),
        "successor_mode_agreement_with_zero": bool(probes["plus"]["mode_after"] == probes["zero"]["mode_after"] == probes["minus"]["mode_after"]),
        "mode_after_plus": probes["plus"]["mode_after"],
        "mode_after_minus": probes["minus"]["mode_after"],
        "mode_after_zero": probes["zero"]["mode_after"],
    }


def gpu_bootstrap(values: np.ndarray, device: torch.device, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n": 0}
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    indices = torch.randint(values.size, (int(draws), values.size), generator=generator, device=device)
    means = tensor[indices].mean(dim=1)
    result = {
        "estimate": float(tensor.mean().item()),
        "ci95_low": float(torch.quantile(means, 0.025).item()),
        "ci95_high": float(torch.quantile(means, 0.975).item()),
        "n": int(values.size),
    }
    del indices, means, tensor
    torch.cuda.synchronize()
    return result


def summarize(records: list[dict[str, Any]], device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    metrics = ("pair_valid", "sign_consistent", "paired_contrast", "paired_separation", "paired_antisymmetry_rmse", "common_drift_abs", "effect_to_drift_ratio", "successor_mode_agreement_pm", "successor_mode_agreement_with_zero")
    groups: dict[str, list[dict[str, Any]]] = {"all": records}
    for row in records:
        groups.setdefault(str(row["edge_id"]), []).append(row)
        groups.setdefault(f"{row['edge_id']}__bin{row['fraction_bin']}", []).append(row)
    output: dict[str, Any] = {}
    for group_index, (key, rows) in enumerate(sorted(groups.items())):
        output[key] = {
            metric: gpu_bootstrap(
                np.asarray([float(row[metric]) for row in rows], dtype=np.float64),
                device,
                int(args.bootstrap),
                int(args.seed + 101 * (group_index + 1) + metric_index),
            )
            for metric_index, metric in enumerate(metrics)
        }
        output[key]["n_roots"] = len(rows)
    return output


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int, bool, str)) or value is None:
        return value
    return value


def main() -> None:
    args = parse_args()
    if args.fraction_bins < 1 or args.states_per_bin < 1 or args.max_roots < 2:
        raise ValueError("fraction-bins, states-per-bin and max-roots must be positive; max-roots >= 2")
    device = require_cuda()
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    try:
        roots = collect_roots(env, args)
        records = [root_record(env, root, args, root_id) for root_id, root in enumerate(roots)]
    finally:
        env.close()
    report = {
        "artifact": "pusht_orientation_response_audit",
        "claim_boundary": (
            "Privileged matched-pulse mechanism audit on genuine simulator contact states. "
            "It diagnoses local orientation-effect identifiability and does not establish "
            "RGB planning, learned-adapter pose closure, or a universal world model."
        ),
        "protocol": {
            "environment": "swm/PushT-v1",
            "seed": int(args.seed),
            "pool_transitions": int(args.pool_transitions),
            "states_per_bin": int(args.states_per_bin),
            "fraction_bins": int(args.fraction_bins),
            "max_roots": int(args.max_roots),
            "max_contact_gap": float(args.max_contact_gap),
            "inward_magnitude": float(args.inward_magnitude),
            "tangent_magnitude": float(args.tangent_magnitude),
            "pulse_steps": int(args.pulse_steps),
            "root_protocol": str(args.root_protocol),
            "matched_action_norms": {
                "zero": 0.0,
                "plus": float(np.hypot(args.inward_magnitude, args.tangent_magnitude)),
                "minus": float(np.hypot(args.inward_magnitude, args.tangent_magnitude)),
            },
            "root_selection": (
                "explicit polygon-edge/fraction roots; accepted nearest transformed edge; no successor-based selection"
                if args.root_protocol == "edge_boundary"
                else "seed-conditioned valid contact roots; nearest transformed polygon edge; edge/fraction strata; no successor-based selection"
                if args.root_protocol == "random_boundary"
                else "behavior-trajectory contact states; nearest transformed polygon edge; edge/fraction strata; no successor-based selection"
            ),
            "runtime_successor_verification": False,
            "simulator_device": "cpu (Gymnasium/Pymunk)",
            "analysis_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "wall_seconds": time.perf_counter() - started,
        },
        "summary": summarize(records, device, args),
        "records": records,
        "sanity_checks": {
            "all_finite": bool(all(np.isfinite(float(row[metric])) for row in records for metric in ("contact_gap", "paired_contrast", "paired_antisymmetry_rmse", "common_drift_abs"))),
            "root_count": len(records),
            "valid_pair_count": int(sum(bool(row["pair_valid"]) for row in records)),
            "equal_norm_plus_minus": bool(all(abs(row["action_norms"]["plus"] - row["action_norms"]["minus"]) < 1.0e-6 for row in records)),
            "same_edge_chart_used_for_pair": True,
            "cuda_analysis_completed": True,
        },
    }
    report = json_safe(report)
    (output / "orientation_response_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# PushT orientation response audit", "",
        f"- Analysis device: `{report['protocol']['analysis_device']}` ({report['protocol']['cuda_device_name']}).",
        f"- Genuine contact roots: `{report['sanity_checks']['root_count']}`; valid opposite-moment pairs: `{report['sanity_checks']['valid_pair_count']}`.", "",
        "| group | roots | valid pair | sign consistency | paired contrast | antisymmetry RMSE | mode agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(report["summary"].items()):
        lines.append(
            f"| {key} | {item['n_roots']} | {item['pair_valid']['estimate']:.3f} | "
            f"{item['sign_consistent']['estimate']:.3f} | {item['paired_contrast']['estimate']:.5f} | "
            f"{item['paired_antisymmetry_rmse']['estimate']:.5f} | {item['successor_mode_agreement_pm']['estimate']:.3f} |"
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "orientation_response_audit.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "sanity_checks": report["sanity_checks"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
