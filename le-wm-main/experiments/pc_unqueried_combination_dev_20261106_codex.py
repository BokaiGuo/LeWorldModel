"""Frozen held-out pairing and genuinely new-tail prediction development panel."""
import json
import argparse
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
OUT = RESULTS / "pc_unqueried_combination_dev_20261106_codex.json"
TASKS = (0, 5, 10, 20, 30, 40, 45)
ARMS = ("parent", "probe_min", "additive", "description", "identity_shuffled")


def predict(a_tails, b_tails, a_cost, observed, parent, shuffle_seed):
    a_tails, b_tails = np.asarray(a_tails), np.asarray(b_tails)
    a_cost, observed, parent = np.asarray(a_cost), np.asarray(observed), np.asarray(parent)
    if a_tails.shape != (4, 30) or b_tails.ndim != 2 or b_tails.shape[1] != 30:
        raise ValueError("invalid tails")
    if a_cost.shape != (4, 4) or observed.shape != (4, 4) or parent.shape != (4,):
        raise ValueError("invalid scores or mask")
    # Additive least squares uses only observed cells.
    pairs = np.argwhere(observed)
    design = np.zeros((len(pairs), 8))
    for n, (i, j) in enumerate(pairs):
        design[n, i], design[n, 4 + j] = 1., 1.
    beta = np.linalg.lstsq(design, a_cost[observed], rcond=None)[0]
    # Center column effects so zero represents a previously unseen tail.
    row_effect = beta[:4] + beta[4:].mean()
    column_effect = beta[4:] - beta[4:].mean()
    rows = {k: np.empty((4, len(b_tails))) for k in ARMS}
    rng = np.random.default_rng(shuffle_seed)
    for i in range(4):
        idx = np.flatnonzero(observed[i])
        if len(idx) < 2:
            raise ValueError("each row needs at least two observations")
        x, y = a_tails[idx], a_cost[i, idx]
        center_x, center_y = x.mean(0), y.mean()
        xc = x - center_x
        def linear(target):
            w = xc.T @ np.linalg.solve(xc @ xc.T + np.eye(len(x)), target - target.mean())
            return center_y + (b_tails - center_x) @ w
        rows["parent"][i] = parent[i]
        rows["probe_min"][i] = y.min()
        rows["additive"][i] = row_effect[i]
        rows["description"][i] = linear(y)
        rows["identity_shuffled"][i] = linear(y[rng.permutation(len(y))])
    return rows, column_effect


def evaluate(predictions, truth, success, mask):
    result = {}
    for name, values in predictions.items():
        errors = (values[mask] - truth[mask]) ** 2
        candidates = np.argwhere(mask)
        choice = candidates[np.argmin(values[mask])]
        i, j = map(int, choice)
        hit = success[i][j]
        result[name] = {"mse": float(errors.mean()), "selected": [i, j],
                        "first_success_step": hit, "prefix_already_success": hit is not None and hit <= 10,
                        "tail_success": hit is not None and hit > 10}
    return result


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-audit", action="store_true")
    args = parser.parse_args()
    output = (RESULTS / "pc_unqueried_combination_trace_audit_dev_20261106_codex.json"
              if args.trace_audit else OUT)
    if output.exists():
        raise FileExistsError(output)
    print("Frozen seven-task held-out pairing and new-tail development evaluation.", flush=True)
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    records = []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        low, high = normalized_bounds(scaler.mean_, scaler.scale_, 10, 1e-6, "cuda", torch.float32)
        panel = panel_rows(data)
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for task in TASKS:
                base = ("pc_branch_panel_offset5_lookahead_seed42_20261106_codex.capture"
                        if task in (5, 45) else "pc_branch_panel_lookahead_seed42_20261105_codex.capture")
                batch = torch.load(RESULTS / base / f"task_{task:02d}_call_00.pt",
                                   map_location="cpu", weights_only=False)
                prefixes = batch["actions"][0, :4, :2].to("cuda")
                parent = batch["cost"][0, :4].numpy()
                def make_tails(seed):
                    gen = torch.Generator(device="cuda").manual_seed(seed)
                    return torch.randn(4, 3, 10, generator=gen, device="cuda").clamp(min=low, max=high)
                a, b = make_tails(73000 + task), make_tails(83000 + task)
                tails = torch.cat((a, b))
                chains = torch.cat((prefixes[:, None].expand(-1, 8, -1, -1),
                                    tails[None].expand(4, -1, -1, -1)), dim=2).reshape(32, 5, 10)
                with torch.inference_mode():
                    model_cost = score(model, batch["context"], chains).cpu().numpy().reshape(4, 8)
                stem = base.removesuffix(".capture")
                physical_trace = load(RESULTS / f"{stem}.trace.jsonl") if args.trace_audit else None
                expected = physical_trace[771001 + task][0]["before"] if physical_trace else None
                images, hits, start_deltas, state_deltas, final_states = [], [], [], [], []
                data_pixels = transform(np.asarray(data["pixels"][panel[task]]))
                data_pixel_delta = float((data_pixels - batch["context"]["pixels"][0, 0, -1]).abs().max())
                for chain in chains.cpu().numpy():
                    physical = scaler.inverse_transform(chain.reshape(25, 2))
                    if not np.isfinite(physical).all() or np.abs(physical).max() > 1 + 1e-6:
                        raise ValueError("invalid physical action")
                    env.reset(seed=771001 + task)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][panel[task]], dtype=np.float64))
                    actor._set_goal_state(np.asarray(data["state"][panel[task] + 25], dtype=np.float64))
                    if expected is not None:
                        state_deltas.append(float(np.max(np.abs(snapshot(actor) - expected))))
                        if state_deltas[-1] > 1e-4:
                            raise ValueError("initial physical snapshot differs from captured runner")
                    start = transform(native_world_pixels(actor.render()))
                    start_deltas.append(float((start - batch["context"]["pixels"][0, 0, -1]).abs().max()))
                    hit = None
                    for step, action in enumerate(physical, 1):
                        _, _, terminated, truncated, _ = actor.step(action.copy())
                        if truncated:
                            raise ValueError("unexpected truncation")
                        if terminated and hit is None:
                            hit = step
                    hits.append(hit)
                    images.append(transform(native_world_pixels(actor.render())))
                    observation = actor._get_obs()
                    final_states.append(np.asarray(observation["state"] if isinstance(observation, dict)
                                                   else observation).tolist())
                with torch.inference_mode():
                    embeddings = []
                    for start in range(0, len(images), 16):
                        inp = torch.stack(images[start:start + 16]).to("cuda")
                        embeddings.append(model.encode({"pixels": inp[:, None]})["emb"][:, 0])
                    emb = torch.cat(embeddings)
                    goal = batch["goal"].to("cuda").reshape(1, -1)
                    actual_cost = (emb - goal).square().sum(-1).cpu().numpy().reshape(4, 8)
                hit_matrix = np.asarray(hits, dtype=object).reshape(4, 8).tolist()
                a_np, b_np = a.cpu().numpy().reshape(4, 30), b.cpu().numpy().reshape(4, 30)
                mask = np.fromfunction(lambda i, j: (i + j) % 4 == 0, (4, 4), dtype=int).astype(bool)
                pair_predictions, column_effect = predict(a_np, a_np, model_cost[:, :4], ~mask, parent, 93000 + task)
                pair_predictions["additive"] += column_effect[None, :]
                new_predictions, _ = predict(a_np, b_np, model_cost[:, :4], np.ones((4, 4), dtype=bool),
                                             parent, 93000 + task)
                pair = evaluate(pair_predictions, model_cost[:, :4], [row[:4] for row in hit_matrix], mask)
                new = evaluate(new_predictions, model_cost[:, 4:], [row[4:] for row in hit_matrix],
                               np.ones((4, 4), dtype=bool))
                records.append({"task": task, "episode": int(data["episode_idx"][panel[task]]),
                                "model_cost": model_cost.tolist(), "actual_encoded_cost": actual_cost.tolist(),
                                "first_success_step": hit_matrix, "start_pixel_max_abs": max(start_deltas),
                                "captured_vs_dataset_pixel_max_abs": data_pixel_delta,
                                "captured_vs_replay_initial_state_max_abs": max(state_deltas) if state_deltas else None,
                                "final_state": np.asarray(final_states).reshape(4, 8, -1).tolist(),
                                "goal_state": np.asarray(data["state"][panel[task] + 25]).tolist(),
                                "pairing": pair, "new_tail": new,
                                "predictions": {"pairing": {k: v.tolist() for k, v in pair_predictions.items()},
                                                "new_tail": {k: v.tolist() for k, v in new_predictions.items()}}})
                print(f"task {task}: 32 chains complete", flush=True)
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with output.open("x") as stream:
        json.dump({"scope": "seven previously exposed development episodes", "model_sha256": fingerprint,
                   "records": records}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
