"""Fixed A-library action-identity ablation with previously replayed outcomes."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_bounded_cem_20261105_codex import normalized_bounds
from experiments.pc_prefix_heldout_allocation_dev_20261106_codex import score
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "pc_action_identity_allocation_dev_20261106_codex.json"


def descriptor(tails, costs, ridge=1.0):
    tails = np.asarray(tails, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if tails.ndim != 2 or costs.shape != (len(tails),) or len(tails) < 2:
        raise ValueError("expected one cost per action vector")
    if not np.isfinite(tails).all() or not np.isfinite(costs).all() or ridge <= 0:
        raise ValueError("invalid data or ridge")
    x = tails - tails.mean(axis=0)
    y = costs - costs.mean()
    weights = x.T @ np.linalg.solve(x @ x.T + ridge * np.eye(len(x)), y)
    return float(costs.mean() - np.linalg.norm(weights))


def summarize(scores, model_record, physical_record):
    indices = np.argsort(np.asarray(scores), kind="stable")[:4]
    outcomes = physical_record["first_success_steps"]
    result = {"retained": indices.tolist()}
    for label in ("B", "optimized"):
        successes = sum(outcomes[label][i] is not None for i in indices)
        result[label + "_success_chains"] = int(successes)
        result[label + "_any_success"] = bool(successes)
    result["best_B_model_cost"] = min(model_record["B_min"][i] for i in indices)
    result["best_optimized_model_cost"] = min(model_record["optimized_min"][i] for i in indices)
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    print("Recreate A queries; compare identity-sensitive descriptor against fixed controls, "
          "then inspect frozen B/optimization physical outcomes.", flush=True)
    baseline_path = RESULTS / "pc_prefix_heldout_allocation_dev_20261106_codex.json"
    physical_path = RESULTS / "pc_prefix_heldout_physical_dev_20261106_codex.json"
    baseline = json.loads(baseline_path.read_text())
    physical = json.loads(physical_path.read_text())
    physics_by_task = {row["task"]: row for row in physical["records"]}
    if [row["task"] for row in baseline["records"]] != [15, 25, 35]:
        raise ValueError("task panel changed")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != baseline["model_sha256"] or fingerprint != physical["model_sha256"]:
        raise ValueError("model fingerprint mismatch")
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    low, high = normalized_bounds(scaler.mean_, scaler.scale_, 10, 1e-6, "cuda", torch.float32)
    rows = []
    with torch.inference_mode():
        for prior in baseline["records"]:
            task = prior["task"]
            capture = RESULTS / "pc_branch_second_replan_lookahead_seed42_20261106_codex.capture" / f"task_{task:02d}_replan_call_00.pt"
            batch = torch.load(capture, map_location="cpu", weights_only=False)
            prefixes = batch["actions"][0, :12, :2].to("cuda")
            generator = torch.Generator(device="cuda").manual_seed(12001 + task)
            tails = torch.randn(8, 3, 10, generator=generator, device="cuda").clamp(min=low, max=high)
            chains = torch.cat((prefixes[:, None].expand(-1, 8, -1, -1),
                                tails[None].expand(12, -1, -1, -1)), dim=2).reshape(-1, 5, 10)
            costs = score(model, batch["context"], chains).reshape(12, 8).cpu().numpy()
            if not np.allclose(costs.min(1), prior["A_min"], rtol=1e-5, atol=1e-5):
                raise ValueError("A minimum did not reproduce")
            if not np.allclose(costs.mean(1), prior["A_mean"], rtol=1e-5, atol=1e-5):
                raise ValueError("A mean did not reproduce")
            tail_np = tails.cpu().numpy().reshape(8, 30)
            shuffle_rng = np.random.default_rng(62001 + task)
            permutations = [shuffle_rng.permutation(8) for _ in range(12)]
            description = [descriptor(tail_np, costs[i]) for i in range(12)]
            shuffled = [descriptor(tail_np, costs[i, permutations[i]]) for i in range(12)]
            scores = {"parent": prior["parent_cost"], "probe_min": prior["A_min"],
                      "probe_mean": prior["A_mean"], "description": description,
                      "identity_shuffled": shuffled}
            rows.append({"task": task, "shuffle": [p.tolist() for p in permutations],
                         "scores": {k: np.asarray(v).tolist() for k, v in scores.items()},
                         "arms": {k: summarize(v, prior, physics_by_task[task]) for k, v in scores.items()}})
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    result = {"scope": "exposed development panel, not independent confirmation",
              "model_sha256": fingerprint, "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
              "physical_sha256": hashlib.sha256(physical_path.read_bytes()).hexdigest(),
              "A_queries_per_task": 96, "B_and_optimization": "previously measured",
              "records": rows}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(OUT), "records": rows}, indent=2))


if __name__ == "__main__":
    main()
