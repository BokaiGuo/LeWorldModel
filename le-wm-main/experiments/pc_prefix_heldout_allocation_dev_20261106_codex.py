"""Held-out suffix quality and fixed-opportunity tail optimization, model-only."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_bounded_cem_20261105_codex import normalized_bounds
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint


def retention_stats(rank_score, heldout, optimized, k=4):
    rank_score, heldout, optimized = map(np.asarray, (rank_score, heldout, optimized))
    if any(x.ndim != 1 for x in (rank_score, heldout, optimized)) or not (
        len(rank_score) == len(heldout) == len(optimized) and 1 <= k <= len(heldout)
    ):
        raise ValueError("invalid retention arrays")
    if not all(np.isfinite(x).all() for x in (rank_score, heldout, optimized)):
        raise ValueError("nonfinite score")
    retained = np.argsort(rank_score, kind="stable")[:k]
    best = np.argsort(heldout, kind="stable")[:k]
    return {"retained": retained.tolist(), "heldout_topk_hit": len(set(retained) & set(best)),
            "best_heldout_cost": float(heldout[retained].min()),
            "best_optimized_cost": float(optimized[retained].min())}


def score(model, context, chains):
    n = len(chains)
    # No physical state, pose, success flag or realized result is provided.
    info = {key: context[key].to("cuda").expand(1, n, *context[key].shape[2:]).clone()
            for key in ("pixels", "goal")}
    info["action"] = torch.zeros((1, n, 1, chains.shape[-1]), device="cuda")
    costs = model.get_cost(info, chains[None])[0]
    if costs.shape != (n,) or not torch.isfinite(costs).all():
        raise ValueError("invalid model scores")
    return costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-chains", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "results"
    output = root / ("pc_prefix_heldout_allocation_chains_dev_20261106_codex.json"
                     if args.save_chains else "pc_prefix_heldout_allocation_dev_20261106_codex.json")
    if output.exists():
        raise FileExistsError(output)
    print("Score held-out tails and independent fixed-prefix optimization; no physical labels enter selection.",
          flush=True)
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    low, high = normalized_bounds(scaler.mean_, scaler.scale_, 10, 1e-6, "cuda", torch.float32)
    results = []
    with torch.inference_mode():
        for task in (15, 25, 35):
            stem = "pc_branch_second_replan_lookahead_seed42_20261106_codex"
            path = root / f"{stem}.capture" / f"task_{task:02d}_replan_call_00.pt"
            batch = torch.load(path, map_location="cpu", weights_only=False)
            parents = batch["actions"][0, :12].to("cuda")
            prefixes = parents[:, :2].clone()
            parent_cost = batch["cost"][0, :12].numpy()
            generator_a = torch.Generator(device="cuda").manual_seed(12001 + task)
            generator_b = torch.Generator(device="cuda").manual_seed(22001 + task)
            generator_opt = torch.Generator(device="cuda").manual_seed(32001 + task)

            def library(count, generator):
                tails = torch.randn(count, 3, 10, generator=generator, device="cuda")
                tails = tails.clamp(min=low, max=high)
                all_prefixes = prefixes[:, None].expand(-1, count, -1, -1)
                all_tails = tails[None].expand(12, -1, -1, -1)
                return torch.cat((all_prefixes, all_tails), dim=2).reshape(-1, 5, 10)

            chain_a = library(8, generator_a)
            chain_b = library(24, generator_b)
            cost_a = score(model, batch["context"], chain_a).reshape(12, 8)
            cost_b = score(model, batch["context"], chain_b).reshape(12, 24)
            winner_b = cost_b.argmin(1)
            b_tails = chain_b.reshape(12, 24, 5, 10)[torch.arange(12, device="cuda"), winner_b]
            centers = b_tails.clone()
            std = torch.ones_like(centers)
            best = cost_b.min(1).values.clone()
            best_chain = b_tails.clone()
            for _ in range(3):
                noise = torch.randn(12, 16, 5, 10, generator=generator_opt, device="cuda")
                candidates = (centers[:, None] + noise * std[:, None]).clamp(min=low, max=high)
                candidates[:, 0] = centers
                candidates[:, :, :2] = prefixes[:, None]
                costs = score(model, batch["context"], candidates.reshape(-1, 5, 10)).reshape(12, 16)
                values, indices = torch.topk(costs, 4, largest=False, dim=1)
                elites = torch.gather(candidates, 1, indices[..., None, None].expand(-1, -1, 5, 10))
                improved = values[:, 0] < best
                best_chain[improved] = elites[improved, 0]
                centers, std = elites.mean(1), elites.std(1)
                centers[:, :2] = prefixes
                best = torch.minimum(best, values[:, 0])
            heldout = cost_b.min(1).values.cpu().numpy()
            optimized = best.cpu().numpy()
            probe_min, probe_mean = cost_a.min(1).values.cpu().numpy(), cost_a.mean(1).cpu().numpy()
            permutation = np.random.default_rng(42001 + task).permutation(12)
            controls = {"parent": parent_cost, "probe_min": probe_min, "probe_mean": probe_mean,
                        "shuffled_probe_min": probe_min[permutation]}
            record = {"task": task, "seeds": {"A": 12001 + task, "B": 22001 + task,
                                                     "optimization": 32001 + task, "shuffle": 42001 + task},
                            "scored": {"A": 96, "B": 288, "optimization": 576,
                                       "parent_preexisting": 12},
                            "parent_cost": parent_cost.tolist(), "A_min": probe_min.tolist(),
                            "A_mean": probe_mean.tolist(), "B_min": heldout.tolist(),
                            "optimized_min": optimized.tolist(), "shuffle": permutation.tolist(),
                            "controls": {name: retention_stats(value, heldout, optimized)
                                         for name, value in controls.items()}}
            if args.save_chains:
                record["B_best_chains"] = b_tails.cpu().tolist()
                record["optimized_best_chains"] = best_chain.cpu().tolist()
            results.append(record)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"model_sha256": fingerprint, "records": results}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
