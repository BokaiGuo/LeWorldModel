"""Matched-budget model tail replanning for actual first-plan prefix interventions."""
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
from experiments.pc_explicit_prefix_intervention_dev_20261106_codex import intervention_commands
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def model_cost(model, pixels, goal, candidates):
    n = candidates.shape[0]
    info = {"pixels": pixels[None, None, None].expand(1, n, 1, -1, -1, -1),
            "goal": goal[None, None, None].expand(1, n, 1, -1, -1, -1),
            "action": torch.zeros(1, n, 1, 10, device=candidates.device)}
    cost = model.get_cost(info, candidates[None])[0]
    if cost.shape != (n,) or not torch.isfinite(cost).all():
        raise ValueError("invalid predicted costs")
    return cost


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    stem = "pc_explicit_prefix_lookahead_dev1106a"
    output = root / f"{stem}.equal_replan_intervention.json"
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads((root / f"{stem}.audit.jsonl").read_text().splitlines()[0])
    if audit["solve"] != 0 or len(audit["environments"]) != 10:
        raise ValueError("invalid audit")
    traces = load(root / f"{stem}.trace.jsonl")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    results = []
    print("Evaluate matched-budget model tail optimization for selected, zero, and random prefixes.", flush=True)
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        low, high = normalized_bounds(scaler.mean_, scaler.scale_, 10, 1e-6, "cuda", torch.float32)
        rows = panel_rows(data, seed=42, count=10)
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for index, (row, entry) in enumerate(zip(rows, audit["environments"], strict=True)):
                if entry["index"] != index:
                    raise ValueError("index mismatch")
                original = np.asarray(entry["selected_actions"], dtype=np.float32)
                physical_original = scaler.inverse_transform(original.reshape(25, 2))
                interventions = intervention_commands(physical_original, index)
                initial = np.stack([scaler.transform(interventions[name]).reshape(5, 10)
                                    for name in ("selected", "zero_prefix", "random_prefix")])
                centers = torch.as_tensor(initial, device="cuda")
                pixels = transform(np.asarray(data["pixels"][row])).to("cuda")
                goal = transform(np.asarray(data["pixels"][row + 25])).to("cuda")
                with torch.inference_mode():
                    best_cost = model_cost(model, pixels, goal, centers)
                    audit_cost = entry["branch_final_costs"][entry["winner"]]
                    if not np.isclose(float(best_cost[0]), audit_cost, atol=1e-4, rtol=1e-4):
                        raise ValueError(f"selected-model-cost mismatch task {index}: {best_cost[0]} vs {audit_cost}")
                    best = centers.clone()
                    fixed = centers[:, :2].clone()
                    std = torch.ones_like(centers)
                    generator = torch.Generator(device="cuda").manual_seed(880000 + index)
                    for _ in range(3):
                        noise = torch.randn(40, 5, 10, generator=generator, device="cuda")
                        candidates = (centers[:, None] + std[:, None] * noise[None]).clamp(min=low, max=high)
                        candidates[:, 0] = centers
                        candidates[:, :, :2] = fixed[:, None]
                        costs = model_cost(model, pixels, goal, candidates.reshape(120, 5, 10)).reshape(3, 40)
                        values, inds = costs.topk(4, largest=False, dim=1)
                        elite = torch.gather(candidates, 1, inds[..., None, None].expand(-1, -1, 5, 10))
                        improved = values[:, 0] < best_cost
                        best[improved] = elite[improved, 0]
                        best_cost = torch.minimum(best_cost, values[:, 0])
                        centers, std = elite.mean(1), elite.std(1)
                        centers[:, :2] = fixed
                records = {}
                for j, name in enumerate(("selected", "zero_prefix", "random_prefix")):
                    plan = scaler.inverse_transform(best[j].cpu().numpy().reshape(25, 2))
                    if not np.isfinite(plan).all() or np.abs(plan).max() > 1.0 + 1e-6:
                        raise ValueError("invalid physical action")
                    env.reset(seed=771001 + index)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][row], dtype=np.float64))
                    target = np.asarray(data["state"][row + 25], dtype=np.float64)
                    actor._set_goal_state(target.copy())
                    expected_first = traces[771001 + index][0]["before"]
                    if not np.allclose(snapshot(actor), expected_first, atol=1e-4, rtol=0):
                        raise ValueError("physical reset mismatch")
                    snapshots = [snapshot(actor)]
                    hit = None
                    for step, action in enumerate(plan, 1):
                        _, _, terminated, truncated, _ = actor.step(action.copy())
                        if truncated:
                            raise ValueError("unexpected truncation")
                        if terminated and hit is None:
                            hit = step
                        snapshots.append(snapshot(actor))
                    states = np.asarray(snapshots)
                    records[name] = {"model_cost": float(best_cost[j]), "first_success_step": hit,
                                     "selected_normalized_chain": best[j].cpu().tolist(),
                                     "object_error_0_10_25": np.linalg.norm(
                                         states[[0, 10, 25], 5:7] - target[None, 2:4], axis=1).tolist(),
                                     "max_object_movement": float(np.linalg.norm(
                                         states[:, 5:7] - states[0, 5:7], axis=1).max())}
                results.append({"index": index, "row": int(row), "scored_model_candidates": 363,
                                "selected_original_cost": float(audit_cost), "controls": records})
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model mutated")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"model_sha256": fingerprint, "records": results}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
