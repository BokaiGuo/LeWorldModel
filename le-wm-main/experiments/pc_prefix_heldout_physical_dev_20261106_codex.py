"""Physical evaluator for already frozen model-only prefix rankings."""
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def summarize(physical, controls):
    if set(physical) != {"B", "optimized"} or any(len(values) != 12 for values in physical.values()):
        raise ValueError("expected both physical outcomes for 12 prefixes")
    summary = {"all": {key: int(sum(values)) for key, values in physical.items()}}
    for name, control in controls.items():
        retained = control["retained"]
        if len(retained) != 4 or len(set(retained)) != 4 or any(not 0 <= i < 12 for i in retained):
            raise ValueError("invalid frozen retention")
        summary[name] = {key: int(sum(values[i] for i in retained))
                         for key, values in physical.items()}
    return summary


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    output = root / "pc_prefix_heldout_physical_dev_20261106_codex.json"
    if output.exists():
        raise FileExistsError(output)
    chain_data = json.loads((root / "pc_prefix_heldout_allocation_chains_dev_20261106_codex.json").read_text())
    baseline = json.loads((root / "pc_prefix_heldout_allocation_dev_20261106_codex.json").read_text())
    for item, old in zip(chain_data["records"], baseline["records"], strict=True):
        stripped = {k: v for k, v in item.items() if k not in ("B_best_chains", "optimized_best_chains")}
        if stripped != old:
            raise ValueError("regenerated model results differ from original")
    print("Replay B and optimized chains from original second-plan physical states; labels remain evaluation-only.",
          flush=True)
    traces = load(root / "pc_branch_second_replan_lookahead_seed42_20261106_codex.trace.jsonl")
    results = []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for item in chain_data["records"]:
                task = item["task"]
                row, trace = panel_rows(data)[task], traces[771001 + task]
                outcomes = {}
                for label, key in (("B", "B_best_chains"), ("optimized", "optimized_best_chains")):
                    outcomes[label] = []
                    for chain in item[key]:
                        normalized = np.asarray(chain, dtype=np.float32).reshape(25, 2)
                        plan = scaler.inverse_transform(normalized)
                        if np.abs(plan).max() > 1.0 + 1e-6:
                            raise ValueError("physical action outside legal bounds")
                        env.reset(seed=771001 + task)
                        actor = env.unwrapped
                        actor._set_state(np.asarray(data["state"][row], dtype=np.float64))
                        actor._set_goal_state(np.asarray(data["state"][row + 25], dtype=np.float64))
                        for entry in trace[:25]:
                            if not np.allclose(snapshot(actor), entry["before"], atol=1e-4, rtol=0):
                                raise ValueError("before-state mismatch")
                            actor.step(np.asarray(entry["action"], dtype=np.float32))
                            if not np.allclose(snapshot(actor), entry["after"], atol=1e-4, rtol=0):
                                raise ValueError("after-state mismatch")
                        hit = None
                        for step, action in enumerate(plan, 1):
                            _, _, terminated, truncated, _ = actor.step(action.copy())
                            if truncated:
                                raise ValueError("unexpected truncation")
                            if terminated and hit is None:
                                hit = step
                        outcomes[label].append(hit)
                flags = {key: [x is not None for x in values] for key, values in outcomes.items()}
                results.append({"task": task, "first_success_steps": outcomes,
                                "summary": summarize(flags, item["controls"])})
        finally:
            env.close()
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"records": results, "model_sha256": chain_data["model_sha256"]},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
