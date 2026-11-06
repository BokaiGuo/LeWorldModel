"""Fail-closed parity check before evaluating first-plan candidate counterfactuals."""
import argparse
import json
import traceback
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def run(data_path, audit_path, trace_path, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output.open("x", encoding="utf-8").close()
    audit = json.loads(audit_path.read_text().splitlines()[0])
    if audit["mode"] != "mean" or len(audit["mean_actions"]) != 50:
        raise ValueError("expected 50 mean-selected first plans")
    traces = {}
    for line in trace_path.read_text().splitlines():
        rec = json.loads(line)
        traces.setdefault(rec["reset_seed"], []).append(rec)
    records = []
    with h5py.File(data_path, "r") as data:
        rows = panel_rows(data)
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        for index, row in enumerate(rows):
            trace = traces.get(771001 + index, [])
            record = {"index": index, "row": int(row), "error": None, "parity": False}
            env = None
            try:
                if not trace or [t["step"] for t in trace] != list(range(1, len(trace) + 1)):
                    raise ValueError("missing or non-contiguous real trace")
                env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
                env.reset(seed=771001 + index)
                base = env.unwrapped
                base._set_state(np.asarray(data["state"][row], dtype=np.float64))
                base._set_goal_state(np.asarray(data["state"][row + 25], dtype=np.float64))
                chain = np.asarray(audit["mean_actions"][index], dtype=np.float32).reshape(25, 2)
                physical_actions = scaler.inverse_transform(chain)
                max_action = max_state = 0.0
                for step, actual in enumerate(trace[:25]):
                    max_state = max(max_state, float(np.max(np.abs(snapshot(base) - actual["before"]))))
                    action = physical_actions[step]
                    max_action = max(max_action, float(np.max(np.abs(action - actual["action"]))))
                    base.step(action.copy())
                    max_state = max(max_state, float(np.max(np.abs(snapshot(base) - actual["after"]))))
                record.update(steps_compared=min(25, len(trace)), max_action_diff=max_action,
                              max_physical_state_diff=max_state,
                              out_of_bounds_commands=int(np.any(np.abs(physical_actions) > 1, axis=1).sum()),
                              parity=max_action <= 1e-6 and max_state <= 1e-4)
            except Exception:
                record["error"] = traceback.format_exc()
            finally:
                if env is not None:
                    env.close()
            records.append(record)
            print(f"parity {index + 1}/50: {record['parity']}", flush=True)
    result = {"n_tasks": len(records), "n_parity": sum(r["parity"] for r in records),
              "n_errors": sum(r["error"] is not None for r in records), "records": records,
              "counterfactual_evaluated": False}
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.audit, args.trace, args.output)
