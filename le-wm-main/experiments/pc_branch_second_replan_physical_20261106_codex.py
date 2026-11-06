"""Replay captured second-plan pool after parity-checked actual first plan."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def run(mode, task):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if task not in (15, 25, 35):
        raise ValueError("task outside declared development roster")
    root = Path(__file__).resolve().parents[1] / "results"
    stem = f"pc_branch_second_replan_{mode}_seed42_20261106_codex"
    output = root / f"{stem}.task_{task:02d}.physical.json"
    if output.exists():
        raise FileExistsError(output)
    original = f"pc_branch_effect_{mode}_task42_solver42_20261105_codex"
    if (root / f"{stem}.trace.jsonl").read_bytes() != (root / f"{original}.trace.jsonl").read_bytes():
        raise ValueError("capture execution trace differs from existing run")
    trace = load(root / f"{stem}.trace.jsonl")[771001 + task]
    if len(trace) < 26 or any(x["terminated"] for x in trace[:25]):
        raise ValueError("task did not actually reach a second plan")
    files = sorted((root / f"{stem}.capture").glob(f"task_{task:02d}_replan_call_*.pt"))
    if len(files) != 20:
        raise ValueError("missing second-plan model batches")
    batches = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    if sum(x["actions"].shape[1] for x in batches) != 3000:
        raise ValueError("invalid second-plan budget")
    bank, keys = [], {}
    for call, batch in enumerate(batches):
        if batch["task"] != task or batch["call"] != call:
            raise ValueError("wrong task or call ordering")
        for slot, (chain, cost) in enumerate(zip(batch["actions"][0].numpy(), batch["cost"][0].numpy(), strict=True)):
            key = chain.tobytes()
            if key not in keys:
                keys[key] = len(bank)
                bank.append({"actions": chain, "costs": [], "occurrences": []})
            bank[keys[key]]["costs"].append(float(cost))
            bank[keys[key]]["occurrences"].append([call, slot])
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        row = panel_rows(data)[task]
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        initial = np.asarray(data["state"][row], dtype=np.float64)
        goal = np.asarray(data["state"][row + 25], dtype=np.float64)
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        records = []
        try:
            for index, item in enumerate(bank):
                env.reset(seed=771001 + task)
                actor = env.unwrapped
                actor._set_state(initial.copy())
                actor._set_goal_state(goal.copy())
                for previous in trace[:25]:
                    if not np.allclose(snapshot(actor), previous["before"], rtol=0, atol=1e-4):
                        raise ValueError("first-plan physical state mismatch")
                    actor.step(np.asarray(previous["action"], dtype=np.float32))
                    if not np.allclose(snapshot(actor), previous["after"], rtol=0, atol=1e-4):
                        raise ValueError("first-plan physical outcome mismatch")
                if not np.allclose(snapshot(actor), trace[25]["before"], rtol=0, atol=1e-4):
                    raise ValueError("second-plan initial state mismatch")
                physical = scaler.inverse_transform(item["actions"].reshape(25, 2))
                if not np.isfinite(physical).all() or np.any(np.abs(physical) > 1):
                    raise ValueError("invalid action")
                hit = None
                for step, action in enumerate(physical, 1):
                    _, _, terminated, truncated, _ = actor.step(action.copy())
                    if truncated:
                        raise ValueError("unexpected truncation")
                    if terminated and hit is None:
                        hit = step
                records.append({"index": index, "occurrences": item["occurrences"], "costs": item["costs"],
                                "first_success_step": hit})
                if (index + 1) % 500 == 0:
                    print(f"{mode} task {task}: {index + 1}/{len(bank)}", flush=True)
        finally:
            env.close()
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"mode": mode, "task": task, "unique": len(bank), "records": records}, stream, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lookahead", "effect"), required=True)
    parser.add_argument("--task", type=int, choices=(15, 25, 35), required=True)
    args = parser.parse_args()
    run(args.mode, args.task)
