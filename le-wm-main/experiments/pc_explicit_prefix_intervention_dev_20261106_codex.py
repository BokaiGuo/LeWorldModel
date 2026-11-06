"""Common-tail physical interventions on every actually selected first plan."""
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def intervention_commands(plan, index):
    plan = np.asarray(plan, dtype=np.float32)
    if plan.shape != (25, 2) or not np.isfinite(plan).all() or (np.abs(plan) > 1.0 + 1e-6).any():
        raise ValueError("invalid physical chain")
    zero = plan.copy()
    zero[:10] = 0
    random = plan.copy()
    random[:10] = np.random.default_rng(770000 + index).uniform(-1, 1, size=(10, 2))
    return {"selected": plan.copy(), "zero_prefix": zero, "random_prefix": random}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    stem = "pc_explicit_prefix_lookahead_dev1106a"
    output = root / f"{stem}.selected_first_plan_intervention.json"
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads((root / f"{stem}.audit.jsonl").read_text().splitlines()[0])
    if audit["solve"] != 0 or len(audit["environments"]) != 10:
        raise ValueError("unexpected first-solve audit")
    trace = load(root / f"{stem}.trace.jsonl")
    print("Evaluate all selected first plans against matched-tail zero and random prefix interventions.", flush=True)
    records = []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        rows = panel_rows(data, seed=42, count=10)
        all_actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(all_actions[~np.isnan(all_actions).any(axis=1)])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for index, (row, audit_row) in enumerate(zip(rows, audit["environments"], strict=True)):
                if audit_row["index"] != index:
                    raise ValueError("audit index mismatch")
                plan = scaler.inverse_transform(np.asarray(audit_row["selected_actions"], dtype=np.float32).reshape(25, 2))
                plans = intervention_commands(plan, index)
                expected = trace[771001 + index][:25]
                if not expected:
                    raise ValueError("missing executed trace")
                target = np.asarray(data["state"][row + 25], dtype=np.float64)
                outcomes = {}
                for name, commands in plans.items():
                    env.reset(seed=771001 + index)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][row], dtype=np.float64))
                    actor._set_goal_state(target.copy())
                    states = [snapshot(actor).tolist()]
                    hit = None
                    for t, action in enumerate(commands):
                        before = snapshot(actor)
                        _, _, terminated, truncated, _ = actor.step(action.copy())
                        after = snapshot(actor)
                        if truncated:
                            raise ValueError("unexpected truncation")
                        if name == "selected" and t < len(expected):
                            e = expected[t]
                            if not (np.allclose(action, e["action"], atol=1e-6, rtol=0)
                                    and np.allclose(before, e["before"], atol=1e-4, rtol=0)
                                    and np.allclose(after, e["after"], atol=1e-4, rtol=0)
                                    and bool(terminated) == e["terminated"]):
                                raise ValueError(f"selected action or trace mismatch: {index=} {t=}")
                        states.append(after.tolist())
                        if terminated and hit is None:
                            hit = t + 1
                    arr = np.asarray(states)
                    object_dist = np.linalg.norm(arr[:, 5:7] - target[None, 2:4], axis=1)
                    object_displacement = np.linalg.norm(arr[:, 5:7] - arr[0, 5:7], axis=1)
                    outcomes[name] = {"first_success_step": hit, "object_goal_distance_0_10_25":
                                      object_dist[[0, 10, 25]].tolist(),
                                      "object_displacement_max": float(object_displacement.max()),
                                      "prefix_object_displacement": float(object_displacement[10]),
                                      "states": states}
                records.append({"index": index, "row": int(row), "executed_prefix_steps": len(expected),
                                "outcomes": outcomes})
        finally:
            env.close()
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"source": stem, "records": records}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
