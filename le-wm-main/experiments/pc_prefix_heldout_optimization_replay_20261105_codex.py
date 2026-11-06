"""Parity-gated physical replay of independently optimized held-out tails."""
import json
import traceback
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    basepath = root / "pc_prefix_heldout_optimization_seed42_20261105_codex"
    output = root / "pc_prefix_heldout_optimization_replay_seed42_20261105_codex.json"
    if output.exists():
        raise FileExistsError(output)
    audit = [json.loads(line) for line in basepath.with_suffix(".audit.jsonl").read_text().splitlines()]
    trace = load(basepath.with_suffix(".trace.jsonl"))
    if not audit or audit[0]["solve"] != 0 or len(audit[0]["returned_actions"]) != 50:
        raise ValueError("missing first returned chain for 50 tasks")
    probes = {}
    for iteration in audit[0]["iterations"]:
        data = iteration["heldout_effect"]
        if data is not None:
            for offset, index in enumerate(range(iteration["batch_start"], iteration["batch_end"])):
                if index in probes:
                    raise ValueError("duplicate heldout context")
                probes[index] = {key: data[key][offset] for key in (
                    "parent_cost", "construction_probe_cost", "optimized_cost",
                    "parent_actions", "optimized_actions")}
    if set(probes) != set(range(50)) or set(trace) != set(range(771001, 771051)):
        raise ValueError("incomplete task IDs")

    dataset = Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5")
    result = {"parity": [], "counterfactual": [], "errors": [], "counterfactual_evaluated": False}
    with h5py.File(dataset, "r") as data:
        rows = panel_rows(data)
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])

        def setup(env, index):
            env.reset(seed=771001 + index)
            env.unwrapped._set_state(np.asarray(data["state"][rows[index]], dtype=np.float64))
            env.unwrapped._set_goal_state(np.asarray(data["state"][rows[index] + 25], dtype=np.float64))
            return env.unwrapped

        for index in range(50):
            env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
            try:
                physical = scaler.inverse_transform(np.asarray(audit[0]["returned_actions"][index],
                                                                dtype=np.float32).reshape(25, 2))
                actor = setup(env, index)
                max_action = max_state = 0.0
                events = trace[771001 + index][:25]
                if not events:
                    raise ValueError("empty trace")
                for step, actual in enumerate(events):
                    max_state = max(max_state, float(np.max(np.abs(snapshot(actor) - actual["before"]))))
                    action = physical[step]
                    max_action = max(max_action, float(np.max(np.abs(action - actual["action"]))))
                    actor.step(action.copy())
                    max_state = max(max_state, float(np.max(np.abs(snapshot(actor) - actual["after"]))))
                result["parity"].append({"index": index, "steps": len(events),
                                         "action_diff": max_action, "state_diff": max_state,
                                         "pass": max_action <= 1e-6 and max_state <= 1e-4})
            except Exception:
                result["errors"].append({"phase": "parity", "index": index,
                                         "traceback": traceback.format_exc()})
            finally:
                env.close()
        if len(result["parity"]) == 50 and all(x["pass"] for x in result["parity"]):
            for index in range(50):
                env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
                try:
                    item = probes[index]
                    records = []
                    for parent in range(2):
                        chains = [("parent", item["parent_actions"][parent], item["parent_cost"][parent])]
                        chains += [("optimized", item["optimized_actions"][parent],
                                    item["optimized_cost"][parent])]
                        for kind, chain, cost in chains:
                            physical = scaler.inverse_transform(np.asarray(chain, dtype=np.float32).reshape(25, 2))
                            actor = setup(env, index)
                            hit, executed = None, 0
                            for step, action in enumerate(physical, 1):
                                _, _, terminated, truncated, _ = actor.step(action.copy())
                                executed = step
                                if terminated:
                                    hit = step
                                    break
                                if truncated:
                                    break
                            records.append({"parent": parent, "chain": kind, "model_cost": cost,
                                            "success": hit is not None, "first_success_step": hit,
                                            "out_of_bounds_plan": int(np.any(np.abs(physical) > 1, axis=1).sum()),
                                            "out_of_bounds_executed": int(np.any(np.abs(physical[:executed]) > 1, axis=1).sum())})
                    result["counterfactual"].append({"index": index, "chains": records,
                                                     "parent_cost": item["parent_cost"],
                                                     "construction_probe_cost": item["construction_probe_cost"],
                                                     "optimized_cost": item["optimized_cost"]})
                except Exception:
                    result["errors"].append({"phase": "counterfactual", "index": index,
                                             "traceback": traceback.format_exc()})
                finally:
                    env.close()
                print(f"physical heldout task {index + 1}/50", flush=True)
            result["counterfactual_evaluated"] = True
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if result["errors"] or len(result["parity"]) != 50 or not all(x["pass"] for x in result["parity"]):
        raise RuntimeError(f"parity or evaluation failed; see {output}")


if __name__ == "__main__":
    main()
