"""Frozen small prefix-by-shared-tail matrix: model, encoding, physical outcome."""
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_bounded_cem_20261105_codex import normalized_bounds
from experiments.pc_prefix_heldout_allocation_dev_20261106_codex import score
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "pc_prefix_tail_interaction_matrix_dev_20261106_codex.json"
TASKS = (15, 25, 35)


def interaction_summary(predicted, actual, tolerance):
    p, a = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if p.shape != (4, 4) or a.shape != (4, 4) or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("expected two finite 4x4 matrices")
    def residual(x):
        return x - x.mean(0, keepdims=True) - x.mean(1, keepdims=True) + x.mean()
    agreeing = disagreeing = tied = 0
    for i, j in itertools.combinations(range(4), 2):
        for k, l in itertools.combinations(range(4), 2):
            dp = p[i, k] - p[i, l] - p[j, k] + p[j, l]
            da = a[i, k] - a[i, l] - a[j, k] + a[j, l]
            if abs(dp) <= tolerance or abs(da) <= tolerance:
                tied += 1
            elif dp * da > 0:
                agreeing += 1
            else:
                disagreeing += 1
    return {"predicted_additive_residual_rms": float(np.sqrt(np.mean(residual(p) ** 2))),
            "actual_additive_residual_rms": float(np.sqrt(np.mean(residual(a) ** 2))),
            "difference_of_differences_agree": agreeing,
            "difference_of_differences_disagree": disagreeing, "ties": tied,
            "tolerance": tolerance}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    print("Evaluate 4x4 shared-tail matrices on three previously exposed development tasks.", flush=True)
    prior = json.loads((RESULTS / "pc_prefix_heldout_allocation_dev_20261106_codex.json").read_text())
    prior_by_task = {row["task"]: row for row in prior["records"]}
    traces = load(RESULTS / "pc_branch_second_replan_lookahead_seed42_20261106_codex.trace.jsonl")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["model_sha256"]:
        raise ValueError("checkpoint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows = []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        low, high = normalized_bounds(scaler.mean_, scaler.scale_, 10, 1e-6, "cuda", torch.float32)
        panel = panel_rows(data)
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for task in TASKS:
                capture = RESULTS / "pc_branch_second_replan_lookahead_seed42_20261106_codex.capture" / f"task_{task:02d}_replan_call_00.pt"
                batch = torch.load(capture, map_location="cpu", weights_only=False)
                prefixes = batch["actions"][0, :4, :2].to("cuda")
                gen = torch.Generator(device="cuda").manual_seed(12001 + task)
                tails = torch.randn(8, 3, 10, generator=gen, device="cuda").clamp(min=low, max=high)[:4]
                chains = torch.cat((prefixes[:, None].expand(-1, 4, -1, -1),
                                    tails[None].expand(4, -1, -1, -1)), dim=2).reshape(16, 5, 10)
                with torch.inference_mode():
                    predicted = score(model, batch["context"], chains).cpu().numpy().reshape(4, 4)
                    repeated = score(model, batch["context"], chains).cpu().numpy().reshape(4, 4)
                repeat_delta = float(np.max(np.abs(predicted - repeated)))
                frames, hits, start_deltas = [], [], []
                for chain in chains.cpu().numpy():
                    plan = scaler.inverse_transform(chain.reshape(25, 2))
                    if not np.isfinite(plan).all() or np.max(np.abs(plan)) > 1 + 1e-6:
                        raise ValueError("illegal action")
                    env.reset(seed=771001 + task)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][panel[task]], dtype=np.float64))
                    actor._set_goal_state(np.asarray(data["state"][panel[task] + 25], dtype=np.float64))
                    for entry in traces[771001 + task][:25]:
                        if not np.allclose(snapshot(actor), entry["before"], atol=1e-4, rtol=0):
                            raise ValueError("prior replay before-state mismatch")
                        actor.step(np.asarray(entry["action"], dtype=np.float32))
                        if not np.allclose(snapshot(actor), entry["after"], atol=1e-4, rtol=0):
                            raise ValueError("prior replay after-state mismatch")
                    start_pixels = transform(native_world_pixels(actor.render()))
                    start_deltas.append(float((start_pixels - batch["context"]["pixels"][0, 0, -1]).abs().max()))
                    first_hit = None
                    for step, action in enumerate(plan, 1):
                        _, _, terminated, truncated, _ = actor.step(action.copy())
                        if truncated:
                            raise ValueError("unexpected truncation")
                        if terminated and first_hit is None:
                            first_hit = step
                    frames.append(transform(native_world_pixels(actor.render())))
                    hits.append(first_hit)
                with torch.inference_mode():
                    inp = torch.stack(frames).to("cuda")
                    features = model.encode({"pixels": inp[:, None]})["emb"][:, 0]
                    goal = batch["goal"].to("cuda").reshape(1, -1)
                    actual = (features - goal).square().sum(-1).cpu().numpy().reshape(4, 4)
                tolerance = max(1e-5, 4 * repeat_delta)
                rows.append({"task": task, "predicted_cost": predicted.tolist(),
                             "actual_encoded_cost": actual.tolist(),
                             "first_success_step": np.asarray(hits, dtype=object).reshape(4, 4).tolist(),
                             "context_pixel_max_abs": max(start_deltas),
                             "model_repeat_max_abs": repeat_delta,
                             "summary": interaction_summary(predicted, actual, tolerance)})
                print(f"task {task}: 16 chains replayed", flush=True)
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model parameters changed")
    with OUT.open("x") as stream:
        json.dump({"scope": "exposed development tasks; not an independent confirmation",
                   "model_sha256": fingerprint, "records": rows}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
