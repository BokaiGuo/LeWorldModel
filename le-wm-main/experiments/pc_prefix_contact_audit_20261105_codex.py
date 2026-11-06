"""Replay selected interventions with body-specific contact logging."""
import argparse
import json
import traceback
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def run(data_path, chain_path, previous_path, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    previous = json.loads(previous_path.read_text())
    first = json.loads(chain_path.read_text().splitlines()[0])
    if previous["n_tasks"] != 50 or previous["n_errors"] or len(first["actions"]) != 50:
        raise ValueError("missing complete selected-chain intervention")
    output.open("x", encoding="utf-8").close()
    records = []
    with h5py.File(data_path, "r") as data:
        rows = panel_rows(data)
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        for index, row in enumerate(rows):
            entry = {"index": index, "row": int(row), "error": []}
            plan = scaler.inverse_transform(
                np.asarray(first["actions"][index], dtype=np.float32).reshape(25, 2))
            for mode in ("plan", "tail_only"):
                env = None
                try:
                    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
                    env.reset(seed=771001 + index)
                    base = env.unwrapped
                    base._set_state(np.asarray(data["state"][row], dtype=np.float64))
                    base._set_goal_state(np.asarray(data["state"][row + 25], dtype=np.float64))
                    active = [False]

                    def contact(arbiter, space, unused):
                        base._handle_collision(arbiter, space, unused)
                        bodies = [shape.body for shape in arbiter.shapes]
                        if any(body is base.agent for body in bodies) and any(
                            body is base.block for body in bodies
                        ):
                            active[0] = True

                    base.space.on_collision(0, 0, post_solve=contact)
                    commands = plan.copy()
                    if mode == "tail_only":
                        commands[:10] = 0
                    states = [np.asarray(base._get_obs(), dtype=np.float64)]
                    flags = []
                    for action in commands:
                        active[0] = False
                        obs, _, _, _, _ = base.step(action.copy())
                        states.append(np.asarray(obs["state"], dtype=np.float64))
                        flags.append(active[0])
                    error = float(np.max(np.abs(np.asarray(states) -
                                                np.asarray(previous["records"][index][mode]["states"]))))
                    entry[mode] = {
                        "state_max_abs_error": error, "state_parity": error <= 1e-4,
                        "agent_object_contact_steps": flags,
                        "contact_prefix_steps": sum(flags[:10]), "contact_tail_steps": sum(flags[10:]),
                        "out_of_bounds_commands": int(np.any(np.abs(commands) > 1, axis=1).sum()),
                    }
                except Exception:
                    entry["error"].append({"mode": mode, "error": traceback.format_exc()})
                finally:
                    if env is not None:
                        env.close()
            records.append(entry)
            print(f"contact audit {index + 1}/50", flush=True)
    report = {"n_tasks": len(records), "n_errors": sum(bool(x["error"]) for x in records),
              "records": records}
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5"))
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.chains, args.previous, args.output)
