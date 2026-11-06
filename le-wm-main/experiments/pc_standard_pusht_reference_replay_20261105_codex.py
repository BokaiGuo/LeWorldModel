"""Reference-only replay for the existing 50-row local PushT panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import h5py
import numpy as np


def panel_rows(data, seed=42, count=50, offset=25):
    episodes = data["episode_idx"][:]
    steps = data["step_idx"][:]
    _, inverse = np.unique(episodes, return_inverse=True)
    lengths = np.zeros(inverse.max() + 1, dtype=np.int64)
    np.maximum.at(lengths, inverse, steps + 1)
    valid = np.flatnonzero(steps <= lengths[inverse] - offset - 1)
    # Exactly mirror eval.py, including its last-valid-row exclusion.
    selection = np.random.default_rng(seed).choice(len(valid) - 1, size=count, replace=False)
    return np.sort(valid[selection])


def run(data_path, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401, registers the environment

    output.mkdir(parents=True, exist_ok=False)
    stat = data_path.stat()
    results = []
    with h5py.File(data_path, "r") as data:
        rows = panel_rows(data)
        for row in rows:
            row = int(row)
            record = {"row": row, "error": None, "steps": [], "completed": False}
            env = None
            try:
                state = np.asarray(data["state"][row:row + 26], dtype=np.float64)
                actions = np.asarray(data["action"][row:row + 25], dtype=np.float64)
                episodes = np.asarray(data["episode_idx"][row:row + 26])
                steps = np.asarray(data["step_idx"][row:row + 26])
                if (state.shape != (26, 7) or actions.shape != (25, 2)
                        or not np.isfinite(state).all() or not np.isfinite(actions).all()
                        or not np.all(episodes == episodes[0])
                        or not np.array_equal(steps, np.arange(steps[0], steps[0] + 26))):
                    raise ValueError("invalid or noncontiguous reference window")
                record.update(episode=int(episodes[0]), start_step=int(steps[0]),
                              source_window_sha256=hashlib.sha256(state.tobytes() + actions.tobytes()).hexdigest(),
                              out_of_bounds_commands=int(np.count_nonzero(np.any(np.abs(actions) > 1, axis=1))))
                env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                               relative=True, render_action=False, with_target=True)
                obs, _ = env.reset(seed=int(episodes[0]), options={"state": state[0], "goal_state": state[-1]})
                base = env.unwrapped
                actual = np.asarray(obs["state"], dtype=np.float64)
                record["initial_reference_state"] = state[0].tolist()
                record["initial_actual_state"] = actual.tolist()
                record["goal_state"] = state[-1].tolist()
                record["initial_max_abs_error"] = float(np.max(np.abs(actual - state[0])))
                for i, action in enumerate(actions):
                    obs, _, terminated, truncated, _ = base.step(action.copy())
                    actual = np.asarray(obs["state"], dtype=np.float64)
                    if not np.isfinite(actual).all():
                        raise ValueError(f"nonfinite actual state at step {i + 1}")
                    record["steps"].append({
                        "step": i + 1, "action": action.tolist(),
                        "reference_state": state[i + 1].tolist(), "actual_state": actual.tolist(),
                        "max_abs_error": float(np.max(np.abs(actual - state[i + 1]))),
                        "agent_xy_error": float(np.linalg.norm(actual[:2] - state[i + 1, :2])),
                        "object_xy_error": float(np.linalg.norm(actual[2:4] - state[i + 1, 2:4])),
                        "terminated": bool(terminated), "truncated": bool(truncated),
                    })
                record["completed"] = True
                record["ever_success"] = any(s["terminated"] for s in record["steps"])
                record["final_success"] = record["steps"][-1]["terminated"]
            except Exception:
                record["error"] = traceback.format_exc()
            finally:
                if env is not None:
                    env.close()
            results.append(record)
            print(f"{len(results)}/50 row={row} completed={record['completed']}", flush=True)

    completed = [r for r in results if r["completed"]]
    report = {
        "protocol": "reference-only; actions[t] to state[t+1]; continue after success",
        "dataset": str(data_path), "dataset_size": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns,
        "dataset_full_hash_verified": False, "seed": 42, "n_planned": len(rows),
        "n_completed": len(completed), "n_failed": len(results) - len(completed),
        "n_ever_success": sum(r["ever_success"] for r in completed),
        "n_final_success": sum(r["final_success"] for r in completed),
        "records": results,
    }
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("Reference-only replay; no reference actions enter a planner.", flush=True)
    run(args.data, args.output)
