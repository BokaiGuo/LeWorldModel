"""Evaluation-only temporal latent alignment for deterministic pool representatives."""
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
    stem = (f"pc_branch_panel_offset5_{mode}_seed42_20261106_codex" if offset5 else
            f"pc_branch_panel_{mode}_seed42_20261105_codex" if panel else
            f"pc_branch_full_pool_{mode}_seed42_20261105_codex")
    output = root / (f"{stem}.task_{task:02d}.alignment.json" if panel else f"{stem}.alignment.json")
    if output.exists():
        raise FileExistsError(output)
    physical = json.loads((root / (f"{stem}.task_{task:02d}.physical.json" if panel else
                                  f"{stem}.physical.json")).read_text())
    records = physical["records"]
    if len(records) != 3000 or physical["unique"] != 3000:
        raise ValueError("incomplete physical census")
    best = min(records, key=lambda x: x["costs"][0])
    passed = [r for r in records if r["first_success_step"] is not None]
    failed = [r for r in records if r["first_success_step"] is None]
    final = [r for r in records if r["occurrences"][0][0] == 19]
    if len(final) != 2:
        raise ValueError("missing final means")
    chosen = {"best_model": best, "selected_mean": min(final, key=lambda x: x["costs"][0])}
    if passed:
        chosen["best_success"] = min(passed, key=lambda x: x["costs"][0])
    if failed:
        chosen["best_failed"] = min(failed, key=lambda x: x["costs"][0])
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    results = []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        row = panel_rows(data)[task]
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        goal = np.asarray(data["pixels"][row + 25])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for label, record in chosen.items():
                call, slot = record["occurrences"][0]
                filename = f"task_{task:02d}_call_{call:02d}.pt" if panel else f"call_{call:02d}.pt"
                batch = torch.load(root / f"{stem}.capture" / filename, map_location="cpu", weights_only=False)
                predicted = batch["prediction"][0, slot]
                pred_goal = batch["goal"].reshape(-1, predicted.shape[-1])[-1]
                if predicted.shape[0] != 6:
                    raise ValueError("unexpected model clock")
                predicted_scores = (predicted - pred_goal).square().sum(-1)
                if not np.isclose(float(predicted_scores[-1]), record["costs"][0], rtol=1e-4, atol=1e-5):
                    raise ValueError("captured prediction does not reproduce model cost")
                plan = scaler.inverse_transform(batch["actions"][0, slot].numpy().reshape(25, 2))
                env.reset(seed=771001 + task)
                actor = env.unwrapped
                actor._set_state(np.asarray(data["state"][row], dtype=np.float64))
                actor._set_goal_state(np.asarray(data["state"][row + 25], dtype=np.float64))
                frames = [native_world_pixels(actor.render())]
                physical_states = []
                first = None
                for step, action in enumerate(plan, 1):
                    obs, _, terminated, truncated, _ = actor.step(action.copy())
                    if truncated:
                        raise ValueError("unexpected truncation")
                    if terminated and first is None:
                        first = step
                    if step % 5 == 0:
                        frames.append(native_world_pixels(actor.render()))
                        physical_states.append(np.asarray(obs["state"]).tolist())
                if first != record["first_success_step"] or not np.allclose(
                        physical_states, record["states_5_10_15_20_25"], atol=1e-4, rtol=0):
                    raise ValueError("physical replay differs from census")
                with torch.inference_mode():
                    inputs = torch.stack([transform(frame) for frame in [goal, *frames]]).to("cuda")
                    emb = model.encode({"pixels": inputs[:, None]})["emb"][:, 0].cpu()
                goal_gap = float((emb[0] - pred_goal).square().sum())
                if goal_gap > 1e-5:
                    raise ValueError(f"goal encoding mismatch {goal_gap}")
                results.append({"label": label, "pool_index": record["index"], "call": call, "slot": slot,
                                "first_success_step": first, "goal_embedding_squared_difference": goal_gap,
                                "times": [0, 5, 10, 15, 20, 25],
                                "predicted_goal_cost": predicted_scores.tolist(),
                                "actual_goal_cost": (emb[1:] - emb[0]).square().sum(-1).tolist(),
                                "prediction_actual_squared_difference": (predicted - emb[1:]).square().sum(-1).tolist()})
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model fingerprint changed")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"mode": mode, "model_sha256": fingerprint, "records": results}, stream, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lookahead", "effect"), required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--offset5", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.task, args.panel, args.offset5)
