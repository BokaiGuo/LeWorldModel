"""Physically replay each distinct first-plan candidate from the sealed pool."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_pusht_scored_first_pool_run_20261107_codex import BASE
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import (
    BASE as PRIOR, DATA, differences, records,
)
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def main(task):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if task not in range(4):
        raise ValueError("task must be 0..3")
    output = BASE / f"physical_{task}.json"
    if output.exists():
        raise FileExistsError(output)
    mismatch = []
    old_audit, new_audit = records(PRIOR / "action_only.audit.jsonl"), records(BASE / "audit.jsonl")
    differences(old_audit, new_audit, "audit", mismatch)
    key = lambda x: (x["reset_seed"], x["step"])
    old_trace = sorted(records(PRIOR / "action_only.trace.jsonl"), key=key)
    new_trace = sorted(records(BASE / "trace.jsonl"), key=key)
    differences(old_trace, new_trace, "trace", mismatch)
    if mismatch:
        raise ValueError(f"recording changed prior run: {mismatch[:3]}")
    trace = [entry for entry in new_trace if entry["reset_seed"] == 871001 + task]
    if not trace:
        raise ValueError("missing physical trace for task")
    bank, indices = [], {}
    max_score_diff = 0.
    for local_call in range(5):
        call = task * 5 + local_call
        batch = torch.load(BASE / "capture" / f"call_{call:03}.pt", map_location="cpu", weights_only=True)
        actions = batch["actions"].numpy()[0]
        cost = batch["cost"].numpy()[0]
        if batch["call"] != call or len(actions) != (30, 200, 200, 200, 2)[local_call]:
            raise ValueError("unexpected query order or size")
        pred = batch["prediction"][:, :, -1]
        goal = batch["goal"][:, -1, None]
        recomputed = ((pred - goal) ** 2).sum(-1).numpy()[0]
        max_score_diff = max(max_score_diff, float(np.max(np.abs(recomputed - cost))))
        np.testing.assert_allclose(recomputed, cost, atol=0.002, rtol=0.0002)
        for slot, (action, score) in enumerate(zip(actions, cost, strict=True)):
            identity = action.tobytes()
            if identity not in indices:
                indices[identity] = len(bank)
                bank.append({"action": action, "occurrences": []})
            bank[indices[identity]]["occurrences"].append({"call": local_call, "slot": slot, "cost": float(score)})
    selected = np.asarray(new_audit[0]["environments"][task]["selected_actions"], np.float32)
    selected_index = indices.get(selected.tobytes())
    if selected_index is None:
        raise ValueError("selected action not in scored pool")
    with h5py.File(DATA, "r") as data:
        row = int(panel_rows(data, seed=85, count=4)[task])
        action_data = data["action"][:]
        scaler = StandardScaler().fit(action_data[~np.isnan(action_data).any(axis=1)])
        initial, goal = np.asarray(data["state"][row], np.float64), np.asarray(data["state"][row + 25], np.float64)
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        results = []
        try:
            for index, item in enumerate(bank):
                env.reset(seed=871001 + task)
                actor = env.unwrapped
                actor._set_state(initial.copy())
                actor._set_goal_state(goal.copy())
                np.testing.assert_allclose(snapshot(actor), trace[0]["before"], atol=1e-4, rtol=0)
                action = scaler.inverse_transform(item["action"].reshape(25, 2))
                if not np.isfinite(action).all() or np.abs(action).max() > 1:
                    raise ValueError("non-legal action")
                first, states, final = None, [], False
                for step, physical in enumerate(action, 1):
                    obs, _, term, trunc, _ = actor.step(physical.copy())
                    if trunc:
                        raise ValueError("unexpected truncation")
                    if term and first is None:
                        first = step
                    final = bool(term)
                    if index == selected_index and step <= min(len(trace), 25):
                        np.testing.assert_allclose(physical, trace[step - 1]["action"], atol=1e-6, rtol=0)
                        np.testing.assert_allclose(snapshot(actor), trace[step - 1]["after"], atol=1e-4, rtol=0)
                    if step % 5 == 0:
                        states.append(np.asarray(obs["state"]).tolist())
                results.append({"index": index, "action_sha256": hashlib.sha256(item["action"].tobytes()).hexdigest(),
                                "occurrences": item["occurrences"], "first_success_step": first,
                                "final_success": final, "packet_states": states})
                if (index + 1) % 100 == 0:
                    print(f"task {task}: replayed {index + 1}/{len(bank)}", flush=True)
        finally:
            env.close()
    with output.open("x") as stream:
        json.dump({"task": task, "data_row": row, "unique": len(bank), "occurrences": 632,
                   "selected_index": selected_index, "max_model_score_reconstruction_error": max_score_diff,
                   "records": results}, stream, allow_nan=False)
    print(f"task {task}: complete, {len(bank)} unique candidates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=int)
    main(parser.parse_args().task)
