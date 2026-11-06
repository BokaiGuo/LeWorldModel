"""Replay all distinct captured plans in the fixed first planning context."""
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


def run(mode, task=0, panel=False, offset5=False):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    if offset5 and not panel:
        raise ValueError("offset5 requires panel")
    roster = (5, 15, 25, 35, 45) if offset5 else (0, 10, 20, 30, 40)
    if task not in roster or (not panel and task != 0):
        raise ValueError("task outside predeclared roster")
    name = (f"pc_branch_panel_offset5_{mode}_seed42_20261106_codex" if offset5 else
            f"pc_branch_panel_{mode}_seed42_20261105_codex" if panel else
            f"pc_branch_full_pool_{mode}_seed42_20261105_codex")
    output = root / (f"{name}.task_{task:02d}.physical.json" if panel else f"{name}.physical.json")
    if output.exists():
        raise FileExistsError(output)
    capture = root / f"{name}.capture"
    files = sorted(capture.glob(f"task_{task:02d}_call_*.pt" if panel else "call_*.pt"))
    if len(files) != 20:
        raise ValueError("incomplete input")
    trace = load(root / f"{name}.trace.jsonl")
    reset_seed = 771001 + task
    if reset_seed not in trace:
        raise ValueError("missing original state trace")
    batches = [torch.load(file, map_location="cpu", weights_only=False) for file in files]
    if sum(x["actions"].shape[1] for x in batches) != 3000:
        raise ValueError("candidate budget mismatch")
    bank, occurrences, keys = [], [], {}
    for call, batch in enumerate(batches):
        if batch["call"] != call:
            raise ValueError("call order mismatch")
        for slot, (chain, cost) in enumerate(zip(batch["actions"][0].numpy(), batch["cost"][0].numpy(), strict=True)):
            key = chain.tobytes()
            if key not in keys:
                keys[key] = len(bank)
                bank.append({"actions": chain, "costs": [], "occurrences": []})
            bank[keys[key]]["costs"].append(float(cost))
            bank[keys[key]]["occurrences"].append([call, slot])
    print(f"{mode}: replay {len(bank)} distinct scored plans; no physical labels enter planner", flush=True)
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        row = panel_rows(data)[task]
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        initial = np.asarray(data["state"][row], dtype=np.float64)
        target = np.asarray(data["state"][row + 25], dtype=np.float64)
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        records = []
        try:
            for index, item in enumerate(bank):
                env.reset(seed=reset_seed)
                actor = env.unwrapped
                actor._set_state(initial.copy())
                actor._set_goal_state(target.copy())
                if not np.allclose(snapshot(actor), trace[reset_seed][0]["before"], rtol=0, atol=1e-4):
                    raise ValueError("initial state differs from actual execution")
                physical = scaler.inverse_transform(item["actions"].reshape(25, 2))
                if not np.isfinite(physical).all() or np.any(np.abs(physical) > 1):
                    raise ValueError("invalid physical action")
                hit, states = None, []
                for step, action in enumerate(physical, 1):
                    obs, _, terminated, truncated, _ = actor.step(action.copy())
                    if truncated:
                        raise ValueError("unexpected truncation")
                    if terminated and hit is None:
                        hit = step
                    if step % 5 == 0:
                        states.append(np.asarray(obs["state"]).tolist())
                records.append({"index": index, "occurrences": item["occurrences"], "costs": item["costs"],
                                "first_success_step": hit, "states_5_10_15_20_25": states})
                if (index + 1) % 300 == 0:
                    print(f"{mode}: {index + 1}/{len(bank)}", flush=True)
        finally:
            env.close()
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"mode": mode, "source": str(capture), "initial_task_index": task,
                   "scored_occurrences": 3000, "unique": len(bank), "records": records},
                  stream, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lookahead", "effect"), required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--offset5", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.task, args.panel, args.offset5)
