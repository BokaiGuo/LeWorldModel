"""Compare fixed-chain rollout and evaluation-only teacher-forced predictions."""
import hashlib
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
from experiments.pc_second_replan_alignment_panel_20261106_codex import representatives
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def teacher_predictions(model, observations, actions, history_size=3):
    """Use actual observation history up to t and actions up to and including t."""
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions) + 1:
        raise ValueError("inconsistent observation/action clock")
    if history_size < 1:
        raise ValueError("invalid history")
    predictions = []
    for t in range(len(actions)):
        start = max(0, t + 1 - history_size)
        obs = observations[None, start:t + 1]
        acts = model.action_encoder(actions[None, start:t + 1])
        predictions.append(model.predict(obs, acts)[0, -1])
    return torch.stack(predictions)


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    output = root / "pc_teacher_forced_rollout_panel_dev_20261106_codex.json"
    if output.exists():
        raise FileExistsError(output)
    print("Compare teacher-forced and autoregressive next latent on fixed replayed candidate chains.", flush=True)
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    results = []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for mode in ("lookahead", "effect"):
                stem = f"pc_branch_second_replan_{mode}_seed42_20261106_codex"
                traces = load(root / f"{stem}.trace.jsonl")
                for task in (15, 25, 35):
                    row = panel_rows(data)[task]
                    goal = np.asarray(data["pixels"][row + 25])
                    records = json.loads((root / f"{stem}.task_{task:02d}.physical.json").read_text())["records"]
                    trace = traces[771001 + task]
                    for label, record in representatives(records).items():
                        call, slot = record["occurrences"][0]
                        path = root / f"{stem}.capture" / f"task_{task:02d}_replan_call_{call:02d}.pt"
                        batch = torch.load(path, map_location="cpu", weights_only=False)
                        predicted = batch["prediction"][0, slot].to("cuda")
                        target = batch["goal"].reshape(-1, predicted.shape[-1])[-1].to("cuda")
                        normalized = batch["actions"][0, slot].numpy()
                        if normalized.shape != (5, 10):
                            raise ValueError("unexpected packet shape")
                        plan = scaler.inverse_transform(normalized.reshape(25, 2))
                        if np.max(np.abs(scaler.transform(plan) - normalized.reshape(25, 2))) > 1e-5:
                            raise ValueError("action scaler mismatch")
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
                        frames = [native_world_pixels(actor.render())]
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
                            raise ValueError("success mismatch")
                        with torch.inference_mode():
                            pixels = torch.stack([transform(x) for x in [goal, *frames]]).to("cuda")
                            emb = model.encode({"pixels": pixels[:, None]})["emb"][:, 0]
                            teacher = teacher_predictions(model, emb[1:], torch.as_tensor(normalized, device="cuda"))
                        initial_gap = float((predicted[0] - emb[1]).square().sum())
                        goal_gap = float((target - emb[0]).square().sum())
                        first_gap = float((teacher[0] - predicted[1]).square().sum())
                        if initial_gap > 1e-5 or goal_gap > 1e-5 or first_gap > 1e-4:
                            raise ValueError(f"parity failed: {initial_gap=} {goal_gap=} {first_gap=}")
                        results.append({"mode": mode, "task": task, "label": label,
                                        "call": call, "slot": slot, "first_success_step": hit,
                                        "action_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
                                        "initial_gap": initial_gap, "goal_gap": goal_gap, "first_gap": first_gap,
                                        "autoregressive_sq_error": (predicted[1:] - emb[2:]).square().sum(-1).cpu().tolist(),
                                        "teacher_forced_sq_error": (teacher - emb[2:]).square().sum(-1).cpu().tolist()})
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"model_sha256": fingerprint, "history_size": 3, "records": results},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
