"""Create a provenance-bound, independent PointMaze trajectory extension.

The original 2,000-root state/action archive is exhausted once every prior
candidate panel is treated as read.  This tool never overwrites that archive:
it creates a new sibling dataset with the original tensors plus newly rolled
Mujoco trajectories under a declared seed.  Stored RGB chart episodes are
read-only and linked only for renderer calibration; new trajectories are
rendered on demand by the V6/V7 builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _legacy_pointmaze_xml


SCHEMA = "pointmaze_independent_state_action_extension.v1"
DEFAULT_EPISODES, DEFAULT_SEED = 320, 202610273


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rollout_trajectory(initial_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Roll a single MuJoCo trajectory on the exact candidate-bank physics."""
    import mujoco

    initial = np.asarray(initial_state, dtype=np.float64)
    controls = np.asarray(actions, dtype=np.float64)
    if initial.shape != (4,) or controls.ndim != 2 or controls.shape[1] != 2 or not np.isfinite(initial).all() or not np.isfinite(controls).all():
        raise ValueError("initial state must be [4] and actions must be finite [steps,2]")
    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    data = mujoco.MjData(model)
    data.qpos[:] = initial[:2]
    data.qvel[:] = initial[2:]
    mujoco.mj_forward(model, data)
    states = np.empty((len(controls), 4), dtype=np.float32)
    states[0] = np.r_[data.qpos[:2], data.qvel[:2]]
    for index in range(1, len(controls)):
        data.ctrl[:] = np.clip(controls[index - 1], -1.0, 1.0)
        mujoco.mj_step(model, data)
        states[index] = np.r_[data.qpos[:2], data.qvel[:2]]
    if not np.isfinite(states).all():
        raise RuntimeError("generated PointMaze rollout is non-finite")
    return states


def build_extension(input_root: Path, output: Path, *, episode_count: int = DEFAULT_EPISODES, seed: int = DEFAULT_SEED) -> dict[str, object]:
    if int(episode_count) < 208:
        raise ValueError("need at least 208 new roots for 144 evaluation plus 64 disjoint design roots")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty extension: {output}")
    states_path, actions_path = input_root / "states.pth", input_root / "actions.pth"
    if not states_path.is_file() or not actions_path.is_file() or not (input_root / "obses").is_dir():
        raise FileNotFoundError("input root needs states.pth, actions.pth, and obses/")
    states = torch.load(states_path, map_location="cpu", weights_only=True).detach().cpu().numpy().astype(np.float32)
    actions = torch.load(actions_path, map_location="cpu", weights_only=True).detach().cpu().numpy().astype(np.float32)
    if states.ndim != 3 or states.shape[1:] != (100, 4) or actions.shape != (states.shape[0], 100, 2):
        raise ValueError("input state/action tensors must be [roots,100,4] and [roots,100,2]")
    rng = np.random.default_rng(int(seed))
    # Sample distinct empirical reset states and perturb them continuously. The
    # resulting root identity is a newly rolled trajectory, not a relabelled
    # archive episode; the source-index ledger is intentionally not exported.
    base_indices = rng.choice(len(states), size=int(episode_count), replace=False if episode_count <= len(states) else True)
    initial = states[base_indices, 0].astype(np.float64)
    initial[:, :2] += rng.uniform(-0.025, 0.025, size=(len(initial), 2))
    initial[:, 2:] += rng.uniform(-0.10, 0.10, size=(len(initial), 2))
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
        "schema": SCHEMA, "source_root": str(input_root.resolve()), "source_states_sha256": _sha256(states_path),
        "source_actions_sha256": _sha256(actions_path), "seed": int(seed), "new_episode_count": int(episode_count),
        "base_root_count": int(len(states)), "extended_root_count": int(len(states) + episode_count),
        "new_root_id_range": [int(len(states)), int(len(states) + episode_count - 1)],
        "trajectory_backend": "MuJoCo legacy PointMaze XML used by candidate-bank simulator",
        "reset_protocol": "distinct empirical reset draws plus continuous qpos/qvel perturbation; random independent 100-step controls",
        "stored_rgb_role": "read-only symlink for fixed renderer calibration only; new state/action trajectories are renderer-on-demand",
        "states_sha256": _sha256(output / "states.pth"), "actions_sha256": _sha256(output / "actions.pth"),
    }
    (output / "extension_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    print(json.dumps(build_extension(**vars(parser.parse_args())), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
