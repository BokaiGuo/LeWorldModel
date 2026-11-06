"""Physical first-plan interventions, gated by recorded trajectory parity."""
import argparse
import json
import traceback
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def run(data_path, chain_path, parity_path, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    parity = json.loads(parity_path.read_text())
    if parity["n_tasks"] != 50 or parity["n_parity"] != 50 or parity["n_errors"]:
        raise ValueError("selected-chain physical parity has not passed")
    first = json.loads(chain_path.read_text().splitlines()[0])
    if first["solve"] != 0 or len(first["actions"]) != 50:
        raise ValueError("invalid first-plan chain record")
    output.open("x", encoding="utf-8").close()
    records = []
    with h5py.File(data_path, "r") as data:
        rows = panel_rows(data)
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        for index, row in enumerate(rows):
            entry = {"index": index, "row": int(row), "errors": []}
            planned = scaler.inverse_transform(
                np.asarray(first["actions"][index], dtype=np.float32).reshape(25, 2))
            target = np.asarray(data["state"][row + 25], dtype=np.float64)
            for prefix, tail, mode in ((True, True, "plan"), (True, False, "prefix_only"),
                                       (False, True, "tail_only"), (False, False, "zero")):
                env = None
                try:
                    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
                    env.reset(seed=771001 + index)
                    base = env.unwrapped
                    base._set_state(np.asarray(data["state"][row], dtype=np.float64))
                    base._set_goal_state(target.copy())
                    commands = planned.copy()
                    if not prefix:
                        commands[:10] = 0
                    if not tail:
                        commands[10:] = 0
                    states = [np.asarray(base._get_obs(), dtype=np.float64).tolist()]
                    first_success = None
                    for step, action in enumerate(commands, start=1):
                        observation, _, terminated, _, _ = base.step(action.copy())
                        states.append(np.asarray(observation["state"], dtype=np.float64).tolist())
                        if terminated and first_success is None:
                            first_success = step
                    states_array = np.asarray(states)
                    error = np.linalg.norm(states_array[:, 2:4] - target[2:4], axis=1)
                    entry[mode] = {
                        "first_success_step": first_success,
                        "success_25": first_success is not None,
                        "out_of_bounds_commands": int(np.any(np.abs(commands) > 1, axis=1).sum()),
                        "object_error_0": float(error[0]), "object_error_10": float(error[10]),
                        "object_error_25": float(error[25]),
                        "states": states,
                    }
                except Exception:
                    entry["errors"].append({"mode": mode, "error": traceback.format_exc()})
                finally:
                    if env is not None:
                        env.close()
            if not entry["errors"]:
                entry["exploratory_witness"] = bool(
                    entry["plan"]["object_error_0"] - entry["plan"]["object_error_10"] <= 1e-3
                    and entry["plan"]["success_25"]
                    and not entry["prefix_only"]["success_25"]
                    and not entry["tail_only"]["success_25"]
                    and all(entry[k]["out_of_bounds_commands"] == 0
                            for k in ("plan", "prefix_only", "tail_only", "zero")))
            records.append(entry)
            print(f"selected-chain intervention {index + 1}/50", flush=True)
    report = {"n_tasks": len(records), "n_errors": sum(bool(r["errors"]) for r in records),
              "n_exploratory_witnesses": sum(r.get("exploratory_witness", False) for r in records),
              "records": records}
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5"))
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.chains, args.parity, args.output)
