"""Generate a geometry-stratified PointMaze extension without model selection.

The ordinary random-reset pool often makes RGB-greedy and a one-step oracle
choose the same action.  This generator instead samples starts whose direct
free-space segment to one declared RGB maze goal intersects a static wall.
The condition uses only legacy maze geometry, before any model or policy is
evaluated.  It is a challenge-pool constructor, not an oracle-labelled test
filter: every accepted root is retained and receives independently sampled
controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.audit_pointmaze_goal_geometry import legacy_collision_geometry
from experiments.decision_benchmark.extend_pointmaze_state_action_pool import rollout_trajectory


SCHEMA = "pointmaze_geometry_stratified_obstacle_extension.v1"
DEFAULT_GOAL = np.asarray([1.8, 2.8], dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_free(qpos: np.ndarray, *, clearance: float) -> bool:
    geometry = legacy_collision_geometry()
    world = np.asarray(qpos, dtype=np.float64) + np.asarray(geometry["particle_body_world_offset"], dtype=np.float64)
    required = float(geometry["particle_radius"]) + float(clearance)
    for wall in geometry["walls"]:
        delta = np.maximum(np.abs(world - np.asarray(wall["center"], dtype=np.float64)) - np.asarray(wall["half_extent"], dtype=np.float64), 0.0)
        if float(np.linalg.norm(delta)) <= required:
            return False
    return True


def _direct_segment_blocked(start: np.ndarray, goal: np.ndarray, *, clearance: float) -> bool:
    return any(
        not _is_free((1.0 - fraction) * np.asarray(start) + fraction * np.asarray(goal), clearance=clearance)
        for fraction in np.linspace(0.0, 1.0, 81)
    )


def obstacle_intersection_starts(goal: np.ndarray = DEFAULT_GOAL, *, clearance: float = 0.08) -> list[np.ndarray]:
    """Return fixed free lattice starts whose direct path to goal is blocked."""

    target = np.asarray(goal, dtype=np.float64).reshape(-1)
    if target.shape != (2,) or not _is_free(target, clearance=clearance):
        raise ValueError("challenge goal must be a free two-dimensional qpos cell")
    cells = [np.asarray([x, y], dtype=np.float64) for x in (-0.2, 0.8, 1.8, 2.8, 3.8) for y in (-0.2, 0.8, 1.8, 2.8, 3.8)]
    starts = [cell for cell in cells if _is_free(cell, clearance=clearance) and _direct_segment_blocked(cell, target, clearance=clearance)]
    if len(starts) < 2:
        raise RuntimeError("static maze geometry yielded too few obstacle-intersection starts")
    return starts


def _sample_start_labels(*, episode_count: int, start_count: int, rng: np.random.Generator, balanced_start_cells: bool) -> list[int]:
    """Sample static strata before trajectories or policies are constructed."""
    if start_count < 2:
        raise ValueError("need at least two static start cells")
    if not balanced_start_cells:
        return [int(rng.integers(start_count)) for _ in range(int(episode_count))]
    if int(episode_count) % int(start_count):
        raise ValueError("balanced start-cell generation needs episode_count divisible by start-cell count")
    labels = np.repeat(np.arange(start_count, dtype=np.int64), int(episode_count) // int(start_count))
    return [int(value) for value in rng.permutation(labels)]


def build(input_root: Path, output: Path, *, episode_count: int, seed: int, goal: np.ndarray = DEFAULT_GOAL, clearance: float = 0.08, balanced_start_cells: bool = False) -> dict[str, object]:
    if int(episode_count) < 208:
        raise ValueError("need at least 208 new roots for isolated design, primary, and replication panels")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty extension: {output}")
    states_path, actions_path = input_root / "states.pth", input_root / "actions.pth"
    if not states_path.is_file() or not actions_path.is_file() or not (input_root / "obses").is_dir():
        raise FileNotFoundError("input root needs states.pth, actions.pth, and obses/")
    states = torch.load(states_path, map_location="cpu", weights_only=True).detach().cpu().numpy().astype(np.float32)
    actions = torch.load(actions_path, map_location="cpu", weights_only=True).detach().cpu().numpy().astype(np.float32)
    if states.ndim != 3 or states.shape[1:] != (100, 4) or actions.shape != (states.shape[0], 100, 2):
        raise ValueError("input state/action tensors must be [roots,100,4] and [roots,100,2]")
    starts = obstacle_intersection_starts(np.asarray(goal, dtype=np.float64), clearance=clearance)
    rng = np.random.default_rng(int(seed))
    initial: list[np.ndarray] = []
    labels = _sample_start_labels(episode_count=int(episode_count), start_count=len(starts), rng=rng, balanced_start_cells=bool(balanced_start_cells))
    for index in labels:
        # The label was fixed before jittering. Resample only the continuous
        # offset, never a different geometry stratum.
        attempts = 0
        while True:
            attempts += 1
            candidate = starts[index] + rng.uniform(-0.06, 0.06, size=2)
            if not _is_free(candidate, clearance=clearance) or not _direct_segment_blocked(candidate, goal, clearance=clearance):
                if attempts >= 10000:
                    raise RuntimeError("could not sample a valid jitter within its fixed start-cell stratum")
                continue
            initial.append(np.r_[candidate, rng.uniform(-0.10, 0.10, size=2)])
            break
    generated_actions = rng.uniform(-1.0, 1.0, size=(int(episode_count), 100, 2)).astype(np.float64)
    generated_states = np.stack([rollout_trajectory(start, controls) for start, controls in zip(initial, generated_actions)]).astype(np.float32)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(torch.from_numpy(np.concatenate((states, generated_states), axis=0)), output / "states.pth")
    torch.save(torch.from_numpy(np.concatenate((actions, generated_actions.astype(np.float32)), axis=0)), output / "actions.pth")
    if (input_root / "seq_lengths.pth").is_file():
        lengths = torch.load(input_root / "seq_lengths.pth", map_location="cpu", weights_only=True)
        torch.save(torch.cat((lengths, torch.full((int(episode_count),), 100, dtype=lengths.dtype))), output / "seq_lengths.pth")
    (output / "obses").symlink_to((input_root / "obses").resolve(), target_is_directory=True)
    report = {
        "schema": SCHEMA, "source_root": str(input_root.resolve()), "seed": int(seed),
        "new_episode_count": int(episode_count), "base_root_count": int(len(states)),
        "extended_root_count": int(len(states) + episode_count), "new_root_id_range": [int(len(states)), int(len(states) + episode_count - 1)],
        "challenge_goal_qpos": np.asarray(goal, dtype=np.float64).tolist(), "clearance": float(clearance),
        "static_start_cells_qpos": [value.tolist() for value in starts], "sampled_start_cell_indices": labels,
        "start_cell_sampling": "equal_allocation_then_seeded_permutation" if balanced_start_cells else "iid_uniform_static_start_cell",
        "selection_contract": "static obstacle-intersection geometry only; no learned model, candidate outcome, oracle rollout, or controller result read during root construction",
        "states_sha256": _sha256(output / "states.pth"), "actions_sha256": _sha256(output / "actions.pth"),
    }
    (output / "challenge_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=320)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--goal", type=float, nargs=2, default=DEFAULT_GOAL.tolist())
    parser.add_argument("--clearance", type=float, default=0.08)
    parser.add_argument("--balanced-start-cells", action="store_true", help="allocate equally across declared static obstacle strata")
    report = build(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
