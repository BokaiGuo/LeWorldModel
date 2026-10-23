"""Privileged PushT reachability and planner-horizon routing audit.

Every arm starts from exactly the same boundary-contact root and uses the
same declared action-bank construction.  It compares a one-pulse greedy
real-successor oracle, an exhaustive two-pulse real-successor oracle, and a
beam-bounded three-pulse real-successor oracle in receding-horizon control.

This is deliberately an *upper-bound routing audit*.  Runtime access to the
simulator successor and state is privileged, so neither an oracle success nor
the ``ranking_limited`` routing label establishes a learned, RGB, frozen-LeWM,
or deployable-planning result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

import stable_worldmodel  # noqa: F401 -- registers the PushT environment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_mode_aware_shared_candidate import (
    angle_error,
    candidate_actions,
    make_boundary_contact_roots,
    pose_cost,
)
from experiments.pusht_protocol_contracts import action_bank_fingerprint, root_fingerprint

gym.logger.min_level = 40

State = np.ndarray
Action = np.ndarray
SuccessorFn = Callable[[State, Action], State]
ActionBankFn = Callable[[State], np.ndarray]
CostFn = Callable[[State], float]


@dataclass(frozen=True)
class SearchResult:
    """First action and best terminal real-successor path for one decision."""

    first_index: int
    first_action: np.ndarray
    sequence_indices: tuple[int, ...]
    terminal_state: np.ndarray
    terminal_cost: float
    expanded_nodes: int
    searched_horizon: int
    bounded: bool


def resolve_device(requested: str) -> tuple[torch.device, dict[str, Any]]:
    """Require the requested CUDA context; never silently turn a GPU run into CPU."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of auto, cpu, cuda")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(requested)
    probe = torch.ones(32, device=device)
    probe.sum().item()
    return device, {
        "resolved_device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_allocation_verified": bool(device.type == "cuda"),
    }


def collect_nonterminal_boundary_contact_roots(
    env: gym.Env, *, count: int, seed: int, oversample_factor: int = 4,
) -> list[dict[str, np.ndarray]]:
    """Return a deterministic, unique, nonterminal contact-root cohort.

    Counting a root that already satisfies the strict terminal pose predicate
    as an oracle success manufactures reachability.  The old smoke artifact
    exhibited exactly this failure mode.  Oversampling is deterministic and
    records only valid roots; it never reuses a physical source/goal pair.
    """

    if count < 1 or oversample_factor < 1:
        raise ValueError("count and oversample_factor must be positive")
    candidates = make_boundary_contact_roots(
        env, int(count) * int(oversample_factor), int(seed)
    )
    selected: list[dict[str, np.ndarray]] = []
    seen: set[str] = set()
    for root in candidates:
        identity = root_fingerprint(root)
        if identity in seen:
            continue
        seen.add(identity)
        if is_pose_success(root["state"], root["goal"]):
            continue
        selected.append({
            "state": np.asarray(root["state"], dtype=np.float64).copy(),
            "goal": np.asarray(root["goal"], dtype=np.float64).copy(),
        })
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RuntimeError(
            "insufficient unique nonterminal boundary-contact roots: "
            f"requested={int(count)}, obtained={len(selected)}"
        )
    return selected


def select_horizon_oracle(
    state: State,
    *,
    horizon: int,
    action_bank_fn: ActionBankFn,
    successor_fn: SuccessorFn,
    cost_fn: CostFn,
    beam_width: int,
) -> SearchResult:
    """Select a first action from real successors under a bounded tree search.

    Horizon 1 and 2 are exhaustive.  Horizon 3 is deliberately bounded after
    each expansion by ``beam_width``.  The returned action is only the first
    primitive; callers replan from the actually reached successor, making all
    arms equal-budget receding-horizon oracle policies.
    """

    if horizon not in {1, 2, 3}:
        raise ValueError("horizon must be 1, 2, or 3")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    source = np.asarray(state, dtype=np.float64).copy()
    root_bank = np.asarray(action_bank_fn(source.copy()), dtype=np.float64)
    if root_bank.ndim != 2 or root_bank.shape[0] == 0:
        raise ValueError("action_bank_fn must return a non-empty [candidate, action_dim] array")

    # (state, root action index, sequence indices).  Keeping the root index
    # makes the selected primitive unambiguous even though child banks may be
    # expressed in their own current contact frame.
    frontier: list[tuple[State, int, tuple[int, ...]]] = [
        (source, int(index), (int(index),)) for index in range(len(root_bank))
    ]
    expanded_nodes = 0
    for depth in range(horizon):
        if depth == 0:
            expanded: list[tuple[State, int, tuple[int, ...]]] = []
            for _source, root_index, sequence in frontier:
                next_state = np.asarray(successor_fn(source.copy(), root_bank[root_index]), dtype=np.float64)
                expanded.append((next_state, root_index, sequence))
                expanded_nodes += 1
        else:
            expanded = []
            for node_state, root_index, sequence in frontier:
                node_bank = np.asarray(action_bank_fn(node_state.copy()), dtype=np.float64)
                if node_bank.ndim != 2 or node_bank.shape[0] == 0:
                    raise ValueError("action_bank_fn returned an invalid child action bank")
                for child_index, action in enumerate(node_bank):
                    next_state = np.asarray(successor_fn(node_state.copy(), action), dtype=np.float64)
                    expanded.append((next_state, root_index, sequence + (int(child_index),)))
                    expanded_nodes += 1
        if not expanded:
            raise RuntimeError("real-successor search produced no candidates")
        # H=1/H=2 are exact.  At h=3 this is explicitly a bounded beam, not
        # a claim that all three-pulse sequences were searched.
        if horizon == 3:
            expanded.sort(key=lambda item: float(cost_fn(item[0])))
            frontier = expanded[:beam_width]
        else:
            frontier = expanded

    terminal_state, root_index, sequence = min(frontier, key=lambda item: float(cost_fn(item[0])))
    return SearchResult(
        first_index=int(root_index),
        first_action=np.asarray(root_bank[root_index], dtype=np.float64).copy(),
        sequence_indices=tuple(int(value) for value in sequence),
        terminal_state=np.asarray(terminal_state, dtype=np.float64).copy(),
        terminal_cost=float(cost_fn(terminal_state)),
        expanded_nodes=int(expanded_nodes),
        searched_horizon=int(horizon),
        bounded=bool(horizon == 3),
    )


def is_pose_success(state: State, goal: State) -> bool:
    return bool(
        np.linalg.norm(np.asarray(state)[2:4] - np.asarray(goal)[2:4]) < 20.0
        and angle_error(float(np.asarray(state)[4]), float(np.asarray(goal)[4])) < np.pi / 9.0
    )


def route_root(arm_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Route from privileged ceilings without attributing failure to any model.

    ``ranking_limited`` means that at least one privileged oracle policy can
    close the root within this *declared* bank and replanning budget.  It says
    a correct ranking exists under the protocol, not that a learned ranker was
    evaluated or is the source of a future policy failure.
    """

    required = ("one_pulse", "two_pulse", "three_pulse_bounded")
    missing = [name for name in required if name not in arm_rows]
    if missing:
        raise ValueError(f"missing horizon arms: {missing}")
    successes = {name: bool(arm_rows[name]["pose_success"]) for name in required}
    first_successful = next((name for name in required if successes[name]), None)
    if first_successful is None:
        return {
            "route": "candidate_or_horizon_limited",
            "reason": "No declared privileged successor oracle reached strict pose success within the three-pulse bounded planning budget.",
            "oracle_reachable": False,
            "first_successful_arm": None,
        }
    return {
        "route": "ranking_limited",
        "reason": "A declared privileged successor oracle reached strict pose success; a correct action ranking exists under this bank and horizon budget.",
        "oracle_reachable": True,
        "first_successful_arm": first_successful,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = ("one_pulse", "two_pulse", "three_pulse_bounded")
    return {
        "root_count": int(len(rows)),
        "arms": {
            arm: {
                "pose_success_rate": float(np.mean([row["arms"][arm]["pose_success"] for row in rows])),
                "mean_final_xy": float(np.mean([row["arms"][arm]["final_xy"] for row in rows])),
                "mean_final_angle": float(np.mean([row["arms"][arm]["final_angle"] for row in rows])),
                "mean_terminal_cost": float(np.mean([row["arms"][arm]["final_cost"] for row in rows])),
            }
            for arm in arms
        },
        "routing": {
            "ranking_limited_roots": int(sum(row["routing"]["route"] == "ranking_limited" for row in rows)),
            "candidate_or_horizon_limited_roots": int(sum(row["routing"]["route"] == "candidate_or_horizon_limited" for row in rows)),
        },
    }


def real_successor(env: gym.Env, state: State, goal: State, action: Action, pulse_steps: int) -> State:
    """Roll out one candidate from a reset copy of the true simulator state."""

    observation, _ = env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    current = np.asarray(observation["state"], dtype=np.float64).copy()
    for _ in range(int(pulse_steps)):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return current


def run_arm(
    env: gym.Env, root: dict[str, np.ndarray], *, horizon: int, args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one receding-horizon privileged oracle arm from an immutable root."""

    current = np.asarray(root["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()

    def bank(state: State) -> np.ndarray:
        return candidate_actions(
            env,
            state,
            tuple(float(value) for value in args.tangent_magnitudes),
            float(args.inward_magnitude),
            goal=goal,
            boundary_action_frame=True,
            contact_normal_magnitudes=tuple(float(value) for value in args.contact_normal_magnitudes),
        )

    root_bank = bank(current)
    trace: list[dict[str, Any]] = []
    for macro_step in range(int(args.macro_steps)):
        if is_pose_success(current, goal):
            break
        choice = select_horizon_oracle(
            current,
            horizon=int(horizon),
            action_bank_fn=bank,
            successor_fn=lambda state, action: real_successor(env, state, goal, action, int(args.pulse_steps)),
            cost_fn=lambda state: pose_cost(state, goal),
            beam_width=int(args.three_pulse_beam_width),
        )
        before = current.copy()
        current = real_successor(env, current, goal, choice.first_action, int(args.pulse_steps))
        trace.append({
            "macro_step": int(macro_step),
            "selected_first_index": int(choice.first_index),
            "selected_sequence_indices": list(choice.sequence_indices),
            "lookahead_terminal_cost": float(choice.terminal_cost),
            "expanded_nodes": int(choice.expanded_nodes),
            "bounded_search": bool(choice.bounded),
            "action": choice.first_action.tolist(),
            "angle_before": float(angle_error(before[4], goal[4])),
            "angle_after": float(angle_error(current[4], goal[4])),
            "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
        })
    return {
        "initial_cost": float(pose_cost(root["state"], goal)),
        "final_cost": float(pose_cost(current, goal)),
        "initial_angle": float(angle_error(root["state"][4], goal[4])),
        "final_angle": float(angle_error(current[4], goal[4])),
        "final_xy": float(np.linalg.norm(current[2:4] - goal[2:4])),
        "pose_success": is_pose_success(current, goal),
        "root_candidate_count": int(len(root_bank)),
        "root_action_bank_sha256": action_bank_fingerprint(root_bank),
        "trace": trace,
    }


def run_root(env: gym.Env, root_id: int, root: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    if is_pose_success(root["state"], root["goal"]):
        raise ValueError("terminal root passed into oracle reachability audit")
    arms = {
        "one_pulse": run_arm(env, root, horizon=1, args=args),
        "two_pulse": run_arm(env, root, horizon=2, args=args),
        "three_pulse_bounded": run_arm(env, root, horizon=3, args=args),
    }
    fingerprints = {arm["root_action_bank_sha256"] for arm in arms.values()}
    if len(fingerprints) != 1:
        raise RuntimeError("horizon arms did not receive identical root action banks")
    return {
        "root_id": int(root_id),
        "root_sha256": root_fingerprint(root),
        "initial_pose_success": False,
        "root_action_bank_sha256": next(iter(fingerprints)),
        "root_candidate_count": int(arms["one_pulse"]["root_candidate_count"]),
        "arms": arms,
        "routing": route_root(arms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202610211)
    parser.add_argument("--macro-steps", type=int, default=18)
    parser.add_argument("--pulse-steps", type=int, default=2)
    parser.add_argument("--three-pulse-beam-width", type=int, default=8)
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+", default=[0.06, 0.10, 0.16])
    parser.add_argument("--contact-normal-magnitudes", type=float, nargs="+", default=[0.02, 0.04, 0.08])
    args = parser.parse_args()
    if args.episodes < 1 or args.macro_steps < 1 or args.pulse_steps < 1:
        parser.error("episodes, macro-steps, and pulse-steps must be positive")
    if args.three_pulse_beam_width < 1:
        parser.error("three-pulse-beam-width must be positive")
    return args


def main() -> None:
    args = parse_args()
    device, runtime = resolve_device(args.device)
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        roots = collect_nonterminal_boundary_contact_roots(
            env, count=int(args.episodes), seed=int(args.seed) + 2000,
        )
        rows = [run_root(env, root_id, root, args) for root_id, root in enumerate(roots)]
    finally:
        env.close()
    report = {
        "schema": "pusht_reachability_horizon_oracle.v1",
        "claim_boundary": (
            "Privileged real-successor ceiling and mechanism routing only. The routing labels do not evaluate any learned/RGB/LeWM ranker, "
            "do not demonstrate a deployable policy, and do not establish causal physical dynamics or universal world modeling."
        ),
        "protocol": {
            "environment": "swm/PushT-v1",
            "root_protocol": "boundary_contact",
            "root_seed": int(args.seed) + 2000,
            "episodes": int(len(rows)),
            "terminal_root_exclusion": (
                "all roots are unique and fail the strict pose-success predicate before any oracle decision"
            ),
            "receding_horizon_arms": ["one_pulse", "two_pulse", "three_pulse_bounded"],
            "macro_steps": int(args.macro_steps),
            "pulse_steps": int(args.pulse_steps),
            "three_pulse_beam_width": int(args.three_pulse_beam_width),
            "runtime_successor_access": True,
            "runtime_privileged_state_access": True,
            "identical_root_and_bank_contract": "All arms start from copied identical roots and must hash to an identical ordered root action bank.",
            "action_bank_config": {
                "boundary_action_frame": True,
                "inward_magnitude": float(args.inward_magnitude),
                "tangent_magnitudes": [float(value) for value in args.tangent_magnitudes],
                "contact_normal_magnitudes": [float(value) for value in args.contact_normal_magnitudes],
            },
            **runtime,
            "physics_execution_note": "PushT simulator successor rollouts are CPU physics; CUDA is preflighted and allocated for runtime integrity, not presented as a physics acceleration result.",
        },
        "summary": summarize(rows),
        "rows": rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "device": str(device), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
