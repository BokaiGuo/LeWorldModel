"""Re-observed second-plan alignment on already captured development pools."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def representatives(records):
    """Physical labels are exclusively for post-hoc diagnostic selection."""
    passed = [r for r in records if r["first_success_step"] is not None]
    final = [r for r in records if r["occurrences"][0][0] == 19]
    selected = {"best_model": min(records, key=lambda r: r["costs"][0]),
                "selected_mean": min(final, key=lambda r: r["costs"][0])}
    if passed:
        selected["best_success"] = min(passed, key=lambda r: r["costs"][0])
    return selected


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lookahead", "effect"), required=True)
    parser.add_argument("--task", type=int, choices=(15, 25, 35), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "results"
    stem = f"pc_branch_second_replan_{args.mode}_seed42_20261106_codex"
    output = root / f"{stem}.task_{args.task:02d}.alignment_panel.json"
    if output.exists():
        raise FileExistsError(output)
    physical = json.loads((root / f"{stem}.task_{args.task:02d}.physical.json").read_text())["records"]
    selected = representatives(physical)
    trace = load(root / f"{stem}.trace.jsonl")[771001 + args.task]
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    results = []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        row = panel_rows(data)[args.task]
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        goal = np.asarray(data["pixels"][row + 25])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for label, record in selected.items():
                call, slot = record["occurrences"][0]
                batch = torch.load(root / f"{stem}.capture" / f"task_{args.task:02d}_replan_call_{call:02d}.pt",
                                   map_location="cpu", weights_only=False)
                predicted = batch["prediction"][0, slot]
                target = batch["goal"].reshape(-1, predicted.shape[-1])[-1]
                predicted_costs = (predicted - target).square().sum(-1)
                if not np.isclose(float(predicted_costs[-1]), record["costs"][0], rtol=1e-4, atol=1e-5):
                    raise ValueError("prediction/cost mismatch")
                env.reset(seed=771001 + args.task)
                actor = env.unwrapped
                actor._set_state(np.asarray(data["state"][row], dtype=np.float64))
                actor._set_goal_state(np.asarray(data["state"][row + 25], dtype=np.float64))
                for entry in trace[:25]:
                    if not np.allclose(snapshot(actor), entry["before"], atol=1e-4, rtol=0):
                        raise ValueError("initial plan before-state mismatch")
                    actor.step(np.asarray(entry["action"], dtype=np.float32))
                    if not np.allclose(snapshot(actor), entry["after"], atol=1e-4, rtol=0):
                        raise ValueError("initial plan after-state mismatch")
                frames = [native_world_pixels(actor.render())]
                normalized = batch["actions"][0, slot].numpy().reshape(25, 2)
                plan = scaler.inverse_transform(normalized)
                roundtrip = float(np.max(np.abs(scaler.transform(plan) - normalized)))
                hit = None
                for step, action in enumerate(plan, 1):
                    _, _, terminated, truncated, _ = actor.step(action.copy())
                    if truncated:
                        raise ValueError("unexpected truncation")
                    if terminated and hit is None:
                        hit = step
                    if step % 5 == 0:
                        frames.append(native_world_pixels(actor.render()))
                if hit != record["first_success_step"]:
                    raise ValueError("physical census mismatch")
                with torch.inference_mode():
                    inputs = torch.stack([transform(x) for x in [goal, *frames]]).to("cuda")
                    emb = model.encode({"pixels": inputs[:, None]})["emb"][:, 0].cpu()
                if predicted.shape != emb[1:].shape:
                    raise ValueError("prediction/observation clock shape mismatch")
                goal_gap = float((emb[0] - target).square().sum())
                if goal_gap > 1e-5:
                    raise ValueError("goal mismatch")
                results.append({"label": label, "call": call, "slot": slot,
                                "first_success_step": hit, "physical_action_min": float(plan.min()),
                                "physical_action_max": float(plan.max()), "action_roundtrip_max_abs": roundtrip,
                                "predicted_goal_cost": predicted_costs.tolist(),
                                "actual_goal_cost": (emb[1:] - emb[0]).square().sum(-1).tolist(),
                                "prediction_actual_squared_difference": (predicted - emb[1:]).square().sum(-1).tolist(),
                                "goal_gap": goal_gap})
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"mode": args.mode, "task": args.task, "time_steps": [0, 5, 10, 15, 20, 25],
                   "selection": "post-hoc physical labels for diagnostics only",
                   "records": results, "model_sha256": fingerprint}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
