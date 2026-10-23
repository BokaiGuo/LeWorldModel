"""Phase-conditioned PushT candidate-bank feasibility audit.

This independent audit asks a deliberately narrow question: does a *declared*
approach/contact/orientation candidate-bank schedule make more of the same
PushT roots feasible to a privileged real-successor oracle than the existing
fixed boundary-frame bank?  Both arms share roots, a phase subgoal schedule,
pulse budget, and the oracle.  The only changing object is candidate-bank
synthesis.

The simulator state and real successors used here are privileged.  Results
therefore bound candidate-bank feasibility only.  They do not evaluate a
learned ranker, RGB observation, a frozen LeWM, or deployment performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import stable_worldmodel  # noqa: F401 -- registers PushT

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_successor_oracle_ceiling import successor as real_successor
from experiments.run_pusht_mode_aware_shared_candidate import (
    angle_error,
    boundary_inward_normal,
    candidate_actions,
    make_boundary_contact_roots,
    physical_contact_geometry,
    pose_cost,
)
from experiments.run_pusht_signed_torque_audit import make_terminal_roots


PHASE_NAMES = ("approach", "contact", "orientation")


@dataclass(frozen=True)
class PhaseContract:
    """Frozen, state-only schedule; values are not fit on outcome roots."""

    contact_gap_px: float = 24.0
    orientation_xy_tube_px: float = 28.0
    approach_magnitudes: tuple[float, ...] = (0.35, 0.55)
    approach_lateral_magnitude: float = 0.12
    contact_normal_magnitudes: tuple[float, ...] = (0.04, 0.08, 0.12)
    contact_tangent_magnitudes: tuple[float, ...] = (0.06, 0.10, 0.16)
    orientation_normal_magnitudes: tuple[float, ...] = (0.04, 0.08)
    orientation_tangent_magnitudes: tuple[float, ...] = (0.06, 0.10, 0.16, 0.22)


DEFAULT_CONTRACT = PhaseContract()


def action_bank_fingerprint(actions: np.ndarray) -> str:
    """Hash the ordered float64 candidate matrix, including candidate order."""

    value = np.ascontiguousarray(np.asarray(actions, dtype=np.float64))
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != 2:
        raise ValueError("candidate bank must be a non-empty [candidate, 2] array")
    if not np.isfinite(value).all():
        raise ValueError("candidate bank contains non-finite values")
    return hashlib.sha256(value.tobytes()).hexdigest()


def root_fingerprint(root: dict[str, np.ndarray]) -> str:
    """Hash the exact source state/goal contract shared by both oracle arms."""

    payload = np.concatenate((
        np.asarray(root["state"], dtype=np.float64),
        np.asarray(root["goal"], dtype=np.float64),
    ))
    if payload.shape != (14,):
        raise ValueError("PushT root must contain state and goal vectors of shape (7,)")
    return hashlib.sha256(np.ascontiguousarray(payload).tobytes()).hexdigest()


def phase_from_observables(
    *, contact_gap_px: float, xy_error_px: float, contract: PhaseContract = DEFAULT_CONTRACT,
) -> str:
    """Return the schedule phase from current geometry, never a successor."""

    if not np.isfinite(contact_gap_px) or not np.isfinite(xy_error_px):
        raise ValueError("phase observables must be finite")
    if contact_gap_px > float(contract.contact_gap_px):
        return "approach"
    if xy_error_px > float(contract.orientation_xy_tube_px):
        return "contact"
    return "orientation"


def phase_observables(
    env: gym.Env, state: np.ndarray, goal: np.ndarray,
) -> dict[str, float | str]:
    """Compute the declared state-only phase inputs from the current state."""

    geometry = physical_contact_geometry(env, np.asarray(state, dtype=np.float64))
    xy_error = float(np.linalg.norm(np.asarray(state, dtype=np.float64)[2:4] - np.asarray(goal, dtype=np.float64)[2:4]))
    gap = float(geometry["distance"])
    return {
        "phase": phase_from_observables(contact_gap_px=gap, xy_error_px=xy_error),
        "contact_gap_px": gap,
        "xy_error_px": xy_error,
    }


def _approach_actions(env: gym.Env, state: np.ndarray, contract: PhaseContract) -> np.ndarray:
    """Fixed fan of contact-seeking actions in the observable boundary frame."""

    inward = np.asarray(boundary_inward_normal(env, state), dtype=np.float64)
    tangent = np.asarray([-inward[1], inward[0]], dtype=np.float64)
    actions: list[np.ndarray] = []
    for magnitude in contract.approach_magnitudes:
        for side in (-1.0, 0.0, 1.0):
            action = float(magnitude) * inward + side * float(contract.approach_lateral_magnitude) * tangent
            if np.max(np.abs(action)) > 1.0 + 1.0e-8:
                raise ValueError("declared approach action exceeds PushT action bounds")
            actions.append(action)
    return np.asarray(actions, dtype=np.float64)


def phase_conditioned_bank(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, contract: PhaseContract = DEFAULT_CONTRACT,
) -> tuple[str, np.ndarray, dict[str, float | str]]:
    """Synthesize a deterministic bank from the current declared phase only."""

    observables = phase_observables(env, state, goal)
    phase = str(observables["phase"])
    if phase == "approach":
        actions = _approach_actions(env, state, contract)
    elif phase == "contact":
        actions = candidate_actions(
            env, state, contract.contact_tangent_magnitudes,
            float(contract.contact_normal_magnitudes[0]), goal=goal,
            boundary_action_frame=True,
            contact_normal_magnitudes=contract.contact_normal_magnitudes,
        )
    elif phase == "orientation":
        actions = candidate_actions(
            env, state, contract.orientation_tangent_magnitudes,
            float(contract.orientation_normal_magnitudes[0]), goal=goal,
            boundary_action_frame=True,
            contact_normal_magnitudes=contract.orientation_normal_magnitudes,
        )
    else:  # defensive: phase_from_observables is the only producer.
        raise RuntimeError(f"unknown phase: {phase}")
    actions = np.asarray(actions, dtype=np.float64)
    action_bank_fingerprint(actions)
    if np.any(np.linalg.norm(actions, axis=1) <= 1.0e-10):
        raise ValueError("phase-conditioned bank cannot contain no-op candidates")
    return phase, actions, observables


def existing_fixed_bank(env: gym.Env, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Exact legacy boundary-frame bank used by the earlier ceiling audit."""

    actions = candidate_actions(
        env, state, (0.06, 0.10, 0.16), 0.08, goal=goal,
        boundary_action_frame=True, contact_normal_magnitudes=(0.02, 0.04, 0.08),
    )
    action_bank_fingerprint(actions)
    return np.asarray(actions, dtype=np.float64)


def phase_subgoal_cost(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, phase: str,
    contract: PhaseContract = DEFAULT_CONTRACT,
) -> float:
    """Predeclared phase objective shared by the fixed and synthesis arms."""

    value = np.asarray(state, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    gap = float(physical_contact_geometry(env, value)["distance"])
    xy_error = float(np.linalg.norm(value[2:4] - target[2:4]))
    angular_error = float(angle_error(value[4], target[4]))
    if phase == "approach":
        return gap + 0.02 * xy_error + 0.01 * angular_error
    if phase == "contact":
        # Keep a small angular barrier so translation cannot spend unlimited
        # orientation error; this exact same objective is used by both arms.
        return xy_error + 4.0 * max(0.0, angular_error - np.pi / 2.0)
    if phase == "orientation":
        # Once inside the frozen XY tube, angle dominates while leaving the
        # tube is explicitly penalized.
        return 30.0 * angular_error + 1.5 * max(0.0, xy_error - float(contract.orientation_xy_tube_px))
    raise ValueError(f"unknown phase: {phase}")


def is_pose_success(state: np.ndarray, goal: np.ndarray) -> bool:
    return bool(
        np.linalg.norm(np.asarray(state)[2:4] - np.asarray(goal)[2:4]) < 20.0
        and angle_error(float(np.asarray(state)[4]), float(np.asarray(goal)[4])) < np.pi / 9.0
    )


def resolve_device(requested: str) -> tuple[torch.device, dict[str, Any]]:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(requested)
    torch.ones(32, device=device).sum().item()
    return device, {
        "resolved_device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_allocation_verified": bool(device.type == "cuda"),
    }


def run_arm(
    env: gym.Env, root: dict[str, np.ndarray], *, arm: str, macro_steps: int,
    pulse_steps: int, contract: PhaseContract,
) -> dict[str, Any]:
    """Execute a one-pulse real-successor feasibility oracle for one bank."""

    if arm not in {"existing_fixed_bank", "phase_conditioned_bank"}:
        raise ValueError(f"unknown arm: {arm}")
    current = np.asarray(root["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()
    trace: list[dict[str, Any]] = []
    phase_counts = {name: 0 for name in PHASE_NAMES}
    for macro_step in range(int(macro_steps)):
        if is_pose_success(current, goal):
            break
        phase, synthesis_actions, observables = phase_conditioned_bank(env, current, goal, contract)
        # Re-synthesis must be exact.  This stops hidden RNG or test-root
        # adaptation from entering the candidate bank.
        repeat_phase, repeat_actions, _ = phase_conditioned_bank(env, current, goal, contract)
        synthesis_hash = action_bank_fingerprint(synthesis_actions)
        if repeat_phase != phase or action_bank_fingerprint(repeat_actions) != synthesis_hash:
            raise RuntimeError("phase-conditioned bank is not deterministic at a fixed state")
        actions = synthesis_actions if arm == "phase_conditioned_bank" else existing_fixed_bank(env, current, goal)
        selected_bank_hash = action_bank_fingerprint(actions)
        successors = [
            real_successor(env, current, goal, action, int(pulse_steps)) for action in actions
        ]
        scores = np.asarray(
            [phase_subgoal_cost(env, outcome, goal, phase, contract) for outcome in successors],
            dtype=np.float64,
        )
        index = int(np.argmin(scores))
        before = current.copy()
        current = np.asarray(successors[index], dtype=np.float64).copy()
        phase_counts[phase] += 1
        trace.append({
            "macro_step": int(macro_step), "phase": phase,
            "contact_gap_px": float(observables["contact_gap_px"]),
            "xy_error_px": float(observables["xy_error_px"]),
            "candidate_count": int(len(actions)),
            "selected_index": index,
            "selected_action": np.asarray(actions[index], dtype=np.float64).tolist(),
            "selected_bank_sha256": selected_bank_hash,
            "phase_synthesis_bank_sha256": synthesis_hash,
            "objective_before": float(phase_subgoal_cost(env, before, goal, phase, contract)),
            "objective_after": float(scores[index]),
            "angle_before": float(angle_error(before[4], goal[4])),
            "angle_after": float(angle_error(current[4], goal[4])),
            "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
        })
    return {
        "initial_cost": float(pose_cost(root["state"], goal)),
        "final_cost": float(pose_cost(current, goal)),
        "final_xy": float(np.linalg.norm(current[2:4] - goal[2:4])),
        "final_angle": float(angle_error(current[4], goal[4])),
        "pose_success": is_pose_success(current, goal),
        "phase_counts": phase_counts,
        "trace": trace,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = ("existing_fixed_bank", "phase_conditioned_bank")
    result: dict[str, Any] = {"root_count": int(len(rows)), "arms": {}}
    for arm in arms:
        result["arms"][arm] = {
            "pose_success_count": int(sum(bool(row["arms"][arm]["pose_success"]) for row in rows)),
            "pose_success_rate": float(np.mean([row["arms"][arm]["pose_success"] for row in rows])),
            "mean_final_xy": float(np.mean([row["arms"][arm]["final_xy"] for row in rows])),
            "mean_final_angle": float(np.mean([row["arms"][arm]["final_angle"] for row in rows])),
            "mean_final_cost": float(np.mean([row["arms"][arm]["final_cost"] for row in rows])),
        }
    repaired = [
        row for row in rows
        if row["arms"]["phase_conditioned_bank"]["pose_success"]
        and not row["arms"]["existing_fixed_bank"]["pose_success"]
    ]
    regressed = [
        row for row in rows
        if row["arms"]["existing_fixed_bank"]["pose_success"]
        and not row["arms"]["phase_conditioned_bank"]["pose_success"]
    ]
    result["feasibility_delta"] = {
        "repaired_root_count": int(len(repaired)),
        "regressed_root_count": int(len(regressed)),
        "net_pose_success_delta": int(len(repaired) - len(regressed)),
        "interpretation": (
            "Exploratory privileged-oracle feasibility delta only; a positive delta does not validate a learned or RGB ranker."
        ),
    }
    return result


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202610212)
    parser.add_argument("--macro-steps", type=int, default=18)
    parser.add_argument("--pulse-steps", type=int, default=2)
    parser.add_argument("--root-protocol", choices=("terminal_pose", "boundary_contact"), default="terminal_pose")
    args = parser.parse_args()
    if args.episodes < 1 or args.macro_steps < 1 or args.pulse_steps < 1:
        parser.error("episodes, macro-steps, and pulse-steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    device, runtime = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    contract = DEFAULT_CONTRACT
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        root_seed = int(args.seed) + 2000
        roots = (
            make_terminal_roots(env, int(args.episodes), root_seed, root_protocol="terminal_pose")
            if args.root_protocol == "terminal_pose"
            else make_boundary_contact_roots(env, int(args.episodes), root_seed)
        )
        rows: list[dict[str, Any]] = []
        for root_id, root in enumerate(roots):
            fixed = run_arm(env, root, arm="existing_fixed_bank", macro_steps=args.macro_steps, pulse_steps=args.pulse_steps, contract=contract)
            phase = run_arm(env, root, arm="phase_conditioned_bank", macro_steps=args.macro_steps, pulse_steps=args.pulse_steps, contract=contract)
            rows.append({
                "root_id": int(root_id),
                "root_sha256": root_fingerprint(root),
                "arms": {"existing_fixed_bank": fixed, "phase_conditioned_bank": phase},
            })
    finally:
        env.close()
    manifest = {
        "schema": "pusht_phase_conditioned_candidate_bank_manifest.v1",
        "git_revision": _git_revision(),
        "command_contract": {
            "root_protocol": args.root_protocol, "root_seed": int(args.seed) + 2000,
            "episodes": int(args.episodes), "macro_steps": int(args.macro_steps),
            "pulse_steps": int(args.pulse_steps), "contract": asdict(contract),
        },
        "runtime": {
            **runtime,
            "physics_execution_note": "PushT successor rollouts are CPU physics; CUDA allocation is preflighted for host runtime integrity and is not presented as physics acceleration.",
        },
    }
    report = {
        "schema": "pusht_phase_conditioned_candidate_bank_oracle.v1",
        "claim_boundary": (
            "Privileged real-successor feasibility oracle only. This experiment makes no learned-ranker, RGB, frozen-LeWM, causal-dynamics, or deployable-planning claim."
        ),
        "manifest": manifest,
        "summary": summarize(rows),
        "rows": rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "device": str(device), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
