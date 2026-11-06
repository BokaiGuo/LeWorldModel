"""Evaluation-only replay of the local seed-42 TwoRoom reference panel."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import h5py
import numpy as np


def select_rows(dataset):
    episodes = dataset["ep_idx"][:]
    steps = dataset["step_idx"][:]
    _, inverse = np.unique(episodes, return_inverse=True)
    lengths = np.zeros(inverse.max() + 1, dtype=np.int64)
    np.maximum.at(lengths, inverse, steps + 1)
    valid = np.flatnonzero(steps <= lengths[inverse] - 26)
    return np.sort(valid[np.random.default_rng(42).choice(len(valid) - 1, size=50, replace=False)])


def run(data_path, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output.mkdir(parents=True, exist_ok=False)
    records = []
    with h5py.File(data_path, "r") as data:
        for row in select_rows(data):
            row = int(row)
            entry = {"row": row, "completed": False, "steps": [], "error": None}
            env = None
            try:
                positions = np.asarray(data["proprio"][row:row + 26])
                actions = np.asarray(data["action"][row:row + 25])
                eps = np.asarray(data["ep_idx"][row:row + 26])
                steps = np.asarray(data["step_idx"][row:row + 26])
                if (positions.shape != (26, 2) or actions.shape != (25, 2)
                        or not np.isfinite(positions).all() or not np.isfinite(actions).all()
                        or not np.all(eps == eps[0])
                        or not np.array_equal(steps, np.arange(steps[0], steps[0] + 26))):
                    raise ValueError("invalid reference window")
                entry.update(episode=int(eps[0]), start_step=int(steps[0]),
                             goal_position=positions[-1].tolist(),
                             out_of_bounds_commands=int(np.any(np.abs(actions) > 1, axis=1).sum()))
                env = gym.make("swm/TwoRoom-v1")
                env.reset()
                base = env.unwrapped
                base._set_state(positions[0])
                base._set_goal_state(positions[-1])
                entry["initial_position"] = base.agent_position.detach().cpu().numpy().tolist()
                for i, action in enumerate(actions):
                    _, _, terminated, truncated, info = base.step(action.copy())
                    actual = np.asarray(info["proprio"])
                    if not np.isfinite(actual).all():
                        raise ValueError("nonfinite observed position")
                    entry["steps"].append({"step": i + 1, "action": action.tolist(),
                                           "reference_position": positions[i + 1].tolist(),
                                           "actual_position": actual.tolist(),
                                           "position_error": float(np.linalg.norm(actual - positions[i + 1])),
                                           "terminated": bool(terminated), "truncated": bool(truncated)})
                entry["completed"] = True
                entry["ever_success"] = any(x["terminated"] for x in entry["steps"])
                entry["final_success"] = entry["steps"][-1]["terminated"]
            except Exception:
                entry["error"] = traceback.format_exc()
            finally:
                if env is not None:
                    env.close()
            records.append(entry)
            print(f"{len(records)}/50 row={row} completed={entry['completed']}", flush=True)
    complete = [x for x in records if x["completed"]]
    stat = data_path.stat()
    result = {"dataset": str(data_path), "dataset_size": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns,
              "n_planned": 50, "n_completed": len(complete), "n_failed": 50 - len(complete),
              "n_ever_success": sum(x["ever_success"] for x in complete),
              "n_final_success": sum(x["final_success"] for x in complete),
              "protocol": "reference only; action[t] to state[t+1]; continue after success",
              "records": records}
    with (output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/tworoom.h5"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("TwoRoom reference-only replay; reference actions never enter a planner.", flush=True)
    run(args.data, args.output)
