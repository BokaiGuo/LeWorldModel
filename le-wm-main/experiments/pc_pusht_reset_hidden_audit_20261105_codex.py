"""Compare repeated native-path PushT initialization and zero-action rollout."""
import argparse
import json
import traceback
from pathlib import Path

import h5py
import numpy as np

from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def snapshot(base):
    return np.array([*base.agent.position, *base.agent.velocity, base.agent.angular_velocity,
                     *base.block.position, *base.block.velocity, base.block.angle,
                     base.block.angular_velocity], dtype=np.float64)


def run(data_path, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output.open("x", encoding="utf-8").close()
    with h5py.File(data_path, "r") as data:
        rows = panel_rows(data)
        records = []
        for index, row in enumerate(rows):
            state = np.asarray(data["state"][row], dtype=np.float64)
            goal = np.asarray(data["state"][row + 25], dtype=np.float64)
            for mode in ("unseeded", "fixed_seed"):
                envs = []
                record = {"index": index, "row": int(row), "mode": mode, "error": None}
                try:
                    for _ in range(2):
                        env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                                       relative=True, render_action=False, with_target=True)
                        envs.append(env)
                        env.reset(seed=None if mode == "unseeded" else 104729 + index)
                        env.unwrapped._set_state(state.copy())
                        env.unwrapped._set_goal_state(goal.copy())
                    first, second = (env.unwrapped for env in envs)
                    a, b = snapshot(first), snapshot(second)
                    record["initial_hidden_max_abs_diff"] = float(np.max(np.abs(a - b)))
                    record["first_initial"] = a.tolist()
                    record["second_initial"] = b.tolist()
                    differences = []
                    for _ in range(25):
                        x = first.step(np.zeros(2))[0]["state"]
                        y = second.step(np.zeros(2))[0]["state"]
                        differences.append(float(np.max(np.abs(x - y))))
                    record["rollout_max_abs_diff"] = max(differences)
                    record["step_max_abs_diffs"] = differences
                except Exception:
                    record["error"] = traceback.format_exc()
                finally:
                    for env in envs:
                        env.close()
                records.append(record)
            print(f"{index + 1}/{len(rows)} tasks audited", flush=True)
    result = {"n_tasks": len(rows), "records": records}
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)
