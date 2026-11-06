"""Evaluate paired first-plan physical outcomes only after trace parity passes."""
import argparse
import json
import traceback
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def run(data_path, audit_path, parity_path, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    parity = json.loads(parity_path.read_text())
    if parity["n_tasks"] != 50 or parity["n_parity"] != 50 or parity["n_errors"] != 0:
        raise ValueError("physical replay parity has not passed for all tasks")
    audit = json.loads(audit_path.read_text().splitlines()[0])
    if audit["mode"] != "mean" or len(audit["mean_actions"]) != 50:
        raise ValueError("invalid first-plan audit")
    output.open("x", encoding="utf-8").close()
    records = []
    with h5py.File(data_path, "r") as data:
        rows = panel_rows(data)
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        for index, row in enumerate(rows):
            record = {"index": index, "row": int(row), "errors": []}
            for mode in ("mean", "sample"):
                env = None
                try:
                    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
                    env.reset(seed=771001 + index)
                    base = env.unwrapped
                    base._set_state(np.asarray(data["state"][row], dtype=np.float64))
                    base._set_goal_state(np.asarray(data["state"][row + 25], dtype=np.float64))
                    chain = np.asarray(audit[f"{mode}_actions"][index], dtype=np.float32).reshape(25, 2)
                    physical = scaler.inverse_transform(chain)
                    hit = None
                    for step, action in enumerate(physical, start=1):
                        _, _, terminated, _, _ = base.step(action.copy())
                        if terminated:
                            hit = step
                            break
                    record[mode] = {
                        "success_25": hit is not None,
                        "first_success_step": hit,
                        "out_of_bounds_all_25": int(np.any(np.abs(physical) > 1, axis=1).sum()),
                        "out_of_bounds_executed": int(np.any(np.abs(physical[:hit]) > 1, axis=1).sum())
                            if hit is not None else int(np.any(np.abs(physical) > 1, axis=1).sum()),
                        "model_cost": audit[f"{mode}_costs"][index],
                    }
                except Exception:
                    record["errors"].append({"mode": mode, "error": traceback.format_exc()})
                finally:
                    if env is not None:
                        env.close()
            records.append(record)
            print(f"counterfactual {index + 1}/50", flush=True)
    result = {"n_tasks": len(records), "n_errors": sum(bool(r["errors"]) for r in records),
              "protocol": "fixed initial physical state; open-loop first 25 steps; no replan",
              "records": records}
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.audit, args.parity, args.output)
