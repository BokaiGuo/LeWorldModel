"""Create a fresh PointMaze pool with position-matched, velocity-opposed histories.

This is not a learned intervention. Each generated pair coasts from adjusted
initial positions so that at the C1 source frame its positions are nearly the
same while its velocities are opposed. The prior RGB history can reveal the
approach direction; a position/goal-only task readout cannot. A later static
candidate-bank preflight must still prove action-pair contrast and proxy
rejection before any visual model is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark.extend_pointmaze_state_action_pool import rollout_trajectory


SCHEMA = "pointmaze_velocity_matched_state_pool.v1"
SOURCE_FRAME, TOTAL_STEPS = 40, 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coast_to_source(target_qpos: np.ndarray, velocity: np.ndarray, *, iterations: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Back-solve a safe initial qpos whose zero-control coast reaches target."""
    target = np.asarray(target_qpos, dtype=np.float64).reshape(2)
    speed = np.asarray(velocity, dtype=np.float64).reshape(2)
    controls = np.zeros((TOTAL_STEPS, 2), dtype=np.float64)
    initial_qpos = target.copy()
    for _ in range(int(iterations)):
        states = rollout_trajectory(np.r_[initial_qpos, speed], controls)
        initial_qpos += target - states[SOURCE_FRAME, :2]
    states = rollout_trajectory(np.r_[initial_qpos, speed], controls)
    return states, initial_qpos


def _pair_is_matched(
    positive: np.ndarray, negative: np.ndarray, *, position_tolerance: float, minimum_speed: float,
    regime: str = "opposed_velocity",
    target_qpos: np.ndarray | None = None, history_qpos_bounds: tuple[float, float] | None = None,
) -> bool:
    pos_gap = float(np.linalg.norm(positive[SOURCE_FRAME, :2] - negative[SOURCE_FRAME, :2]))
    velocities = positive[SOURCE_FRAME, 2:] * negative[SOURCE_FRAME, 2:]
    target_ok = True if target_qpos is None else bool(
        np.linalg.norm(positive[SOURCE_FRAME, :2] - np.asarray(target_qpos, dtype=np.float64).reshape(2)) <= float(position_tolerance)
        and np.linalg.norm(negative[SOURCE_FRAME, :2] - np.asarray(target_qpos, dtype=np.float64).reshape(2)) <= float(position_tolerance)
    )
    history_ok = True
    if history_qpos_bounds is not None:
        low, high = (float(value) for value in history_qpos_bounds)
        history = np.concatenate((positive[: SOURCE_FRAME + 1, :2], negative[: SOURCE_FRAME + 1, :2]), axis=0)
        history_ok = bool(np.all(history >= low) and np.all(history <= high))
    if regime == "opposed_velocity":
        velocity_ok = bool(
            float(np.linalg.norm(positive[SOURCE_FRAME, 2:])) >= float(minimum_speed)
            and float(np.linalg.norm(negative[SOURCE_FRAME, 2:])) >= float(minimum_speed)
            and float(np.sum(velocities)) < 0.0
        )
    elif regime == "zero_velocity":
        velocity_ok = bool(
            float(np.linalg.norm(positive[SOURCE_FRAME, 2:])) <= float(minimum_speed)
            and float(np.linalg.norm(negative[SOURCE_FRAME, 2:])) <= float(minimum_speed)
        )
    else:
        raise ValueError(f"unsupported pool regime: {regime!r}")
    return bool(pos_gap <= float(position_tolerance) and velocity_ok and target_ok and history_ok)


def build_pool(
    input_root: Path, output: Path, *, pair_count: int = 416, seed: int = 202611241,
    velocity_magnitude: float = 1.5, position_tolerance: float = 0.025, minimum_source_speed: float = 0.08,
    target_goal: np.ndarray | None = None, target_radius: float | None = None,
    history_qpos_bounds: tuple[float, float] | None = None,
    regime: str = "opposed_velocity",
) -> dict[str, Any]:
    """Append `2*pair_count` independently generated velocity-matched roots.

    With ``target_goal`` and ``target_radius``, source positions lie on a
    declared ring around the task goal and the opposed velocities are radial.
    This is a distinct controlled mechanism for testing whether incoming
    approach direction changes a braking-versus-acceleration preference; it
    does not inspect any candidate action outcome.
    """
    if int(pair_count) < 1:
        raise ValueError("pair_count must be positive")
    if regime not in {"opposed_velocity", "zero_velocity"}:
        raise ValueError("regime must be opposed_velocity or zero_velocity")
    if regime == "opposed_velocity" and float(velocity_magnitude) <= 0.0:
        raise ValueError("opposed_velocity pool needs positive velocity_magnitude")
    if regime == "zero_velocity" and float(velocity_magnitude) != 0.0:
        raise ValueError("zero_velocity pool requires velocity_magnitude=0")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty velocity-matched pool: {output}")
    states_path, actions_path = Path(input_root) / "states.pth", Path(input_root) / "actions.pth"
    if not states_path.is_file() or not actions_path.is_file() or not (Path(input_root) / "obses").is_dir():
        raise FileNotFoundError("input root needs states.pth, actions.pth, and obses/")
    source_states = torch.load(states_path, map_location="cpu", weights_only=True).detach().cpu().numpy().astype(np.float32)
    source_actions = torch.load(actions_path, map_location="cpu", weights_only=True).detach().cpu().numpy().astype(np.float32)
    if source_states.ndim != 3 or source_states.shape[1:] != (TOTAL_STEPS, 4) or source_actions.shape != (len(source_states), TOTAL_STEPS, 2):
        raise ValueError("PointMaze pool tensors must be [roots,100,4] and [roots,100,2]")
    rng = np.random.default_rng(int(seed))
    candidates = source_states[:, SOURCE_FRAME, :2]
    goal = None if target_goal is None else np.asarray(target_goal, dtype=np.float64).reshape(-1)
    if goal is not None and (goal.shape != (2,) or not np.isfinite(goal).all() or target_radius is None or float(target_radius) <= 0.0):
        raise ValueError("goal-conditioned velocity pool needs finite target_goal=[2] and positive target_radius")
    if goal is None and target_radius is not None:
        raise ValueError("target_radius requires target_goal")
    if history_qpos_bounds is not None:
        lower, upper = (float(value) for value in history_qpos_bounds)
        if not np.isfinite([lower, upper]).all() or not lower < upper:
            raise ValueError("history_qpos_bounds must be finite increasing bounds")
    generated_states, generated_actions, pairs = [], [], []
    attempts, maximum_attempts = 0, int(pair_count) * 32
    while len(pairs) < int(pair_count) and attempts < maximum_attempts:
        attempts += 1
        direction = rng.normal(size=2)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        target = (
            candidates[int(rng.integers(len(candidates)))].astype(np.float64)
            if goal is None else goal + float(target_radius) * direction
        )
        positive, initial_positive = _coast_to_source(target, float(velocity_magnitude) * direction)
        negative, initial_negative = _coast_to_source(target, -float(velocity_magnitude) * direction)
        if not _pair_is_matched(
            positive, negative, position_tolerance=position_tolerance, minimum_speed=minimum_source_speed,
            target_qpos=target, history_qpos_bounds=history_qpos_bounds, regime=regime,
        ):
            continue
        future_controls = rng.uniform(-1.0, 1.0, size=(TOTAL_STEPS - SOURCE_FRAME, 2)).astype(np.float64)
        plus_actions = np.vstack((np.zeros((SOURCE_FRAME, 2), dtype=np.float64), future_controls))
        minus_actions = np.vstack((np.zeros((SOURCE_FRAME, 2), dtype=np.float64), future_controls))
        plus = rollout_trajectory(np.r_[initial_positive, float(velocity_magnitude) * direction], plus_actions)
        minus = rollout_trajectory(np.r_[initial_negative, -float(velocity_magnitude) * direction], minus_actions)
        if not _pair_is_matched(
            plus, minus, position_tolerance=position_tolerance, minimum_speed=minimum_source_speed,
            target_qpos=target, history_qpos_bounds=history_qpos_bounds, regime=regime,
        ):
            continue
        pair_index = len(pairs)
        generated_states.extend((plus, minus)); generated_actions.extend((plus_actions, minus_actions))
        pairs.append({
            "pair_index": pair_index, "target_source_qpos": target.astype(float).tolist(), "direction": direction.astype(float).tolist(),
            "plus_initial_qpos": initial_positive.astype(float).tolist(), "minus_initial_qpos": initial_negative.astype(float).tolist(),
            "plus_source_state": plus[SOURCE_FRAME].astype(float).tolist(), "minus_source_state": minus[SOURCE_FRAME].astype(float).tolist(),
            "source_position_distance": float(np.linalg.norm(plus[SOURCE_FRAME, :2] - minus[SOURCE_FRAME, :2])),
            "source_velocity_dot": float(np.dot(plus[SOURCE_FRAME, 2:], minus[SOURCE_FRAME, 2:])),
        })
    if len(pairs) != int(pair_count):
        raise RuntimeError(f"velocity-matched pool exhausted after {attempts} attempts: built {len(pairs)}/{pair_count} pairs")
    output.mkdir(parents=True, exist_ok=False)
    generated_states_array = np.asarray(generated_states, dtype=np.float32)
    generated_actions_array = np.asarray(generated_actions, dtype=np.float32)
    torch.save(torch.from_numpy(np.concatenate((source_states, generated_states_array), axis=0)), output / "states.pth")
    torch.save(torch.from_numpy(np.concatenate((source_actions, generated_actions_array), axis=0)), output / "actions.pth")
    lengths_path = Path(input_root) / "seq_lengths.pth"
    if lengths_path.is_file():
        lengths = torch.load(lengths_path, map_location="cpu", weights_only=True)
        torch.save(torch.cat((lengths, torch.full((len(generated_states_array),), TOTAL_STEPS, dtype=lengths.dtype))), output / "seq_lengths.pth")
    (output / "obses").symlink_to((Path(input_root) / "obses").resolve(), target_is_directory=True)
    report = {
        "schema": SCHEMA, "source_root": str(Path(input_root).resolve()), "source_states_sha256": _sha256(states_path),
        "source_actions_sha256": _sha256(actions_path), "seed": int(seed), "pair_count": int(pair_count),
        "new_episode_count": int(len(generated_states_array)), "base_root_count": int(len(source_states)),
        "extended_root_count": int(len(source_states) + len(generated_states_array)),
        "new_root_id_range": [int(len(source_states)), int(len(source_states) + len(generated_states_array) - 1)],
        "source_frame": SOURCE_FRAME, "regime": regime, "velocity_magnitude": float(velocity_magnitude), "position_tolerance": float(position_tolerance),
        "minimum_source_speed": float(minimum_source_speed), "attempt_count": int(attempts), "pairs": pairs,
        "target_construction": (
            "empirical_source_frame_positions" if goal is None else "goal_centered_radial_ring"
        ),
        "target_goal": None if goal is None else goal.astype(float).tolist(),
        "target_radius": None if goal is None else float(target_radius),
        "history_qpos_bounds": None if history_qpos_bounds is None else [float(value) for value in history_qpos_bounds],
        "states_sha256": _sha256(output / "states.pth"), "actions_sha256": _sha256(output / "actions.pth"),
        "claim_boundary": "Controlled state-pool construction only; no candidate selection, encoder fit, outcome label, or planning claim.",
    }
    (output / "velocity_matched_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=416)
    parser.add_argument("--seed", type=int, default=202611241)
    parser.add_argument("--goal-qpos", type=float, nargs=2, default=None)
    parser.add_argument("--target-radius", type=float, default=None)
    parser.add_argument("--history-qpos-bounds", type=float, nargs=2, default=None)
    parser.add_argument("--regime", choices=("opposed_velocity", "zero_velocity"), default="opposed_velocity")
    args = parser.parse_args()
    values = vars(args)
    values["target_goal"] = values.pop("goal_qpos")
    values["history_qpos_bounds"] = None if values["history_qpos_bounds"] is None else tuple(values["history_qpos_bounds"])
    print(json.dumps(build_pool(**values), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
