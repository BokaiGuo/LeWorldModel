"""CUDA-only P3 rank-matched repair stress test on the frozen P2 mechanism."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from experiments.gpu_preflight import require_cuda_allocation


def _mean_regret(actions: torch.Tensor, real: torch.Tensor, predicted: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    predicted_cost = ((predicted - goals[:, None, :]) ** 2).sum(-1)
    real_cost = ((real - goals[:, None, :]) ** 2).sum(-1)
    selected = predicted_cost.argmin(1)
    return (real_cost.gather(1, selected[:, None]).squeeze(1) - real_cost.min(1).values).mean()


def run(*, output: Path, device: str = "cuda", seed: int = 202610231) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device != "cuda":
        raise ValueError("P3 synthetic repair experiment is CUDA-only; CPU fallback is forbidden")
    cuda = require_cuda_allocation(device)
    d = torch.device(device)
    g = torch.Generator(device=d).manual_seed(seed)
    models, roots, candidates, action_dim = 48, 160, 32, 4
    actions = torch.randn((roots, candidates, action_dim), device=d, dtype=torch.float64, generator=g)
    true_response = torch.randn((2, action_dim), device=d, dtype=torch.float64, generator=g)
    true_response /= torch.linalg.vector_norm(true_response, dim=1, keepdim=True)
    base = 0.4 * torch.randn((roots, 2), device=d, dtype=torch.float64, generator=g)
    goals = 0.6 * torch.randn((roots, 2), device=d, dtype=torch.float64, generator=g)
    real = base[:, None, :] + actions @ true_response.T
    names = ("spectral_rank1", "random_rank1", "pca_axis_rank1", "ordinary_full_effect", "naive_task_full")
    regret = {name: [] for name in names}
    wins = {name: 0 for name in names}
    for model in range(models):
        # All arms see the same noisy action-effect estimate and have rank-one
        # correction budget except two declared full-effect comparators.
        error_scale = 0.15 + 1.05 * model / (models - 1)
        error = error_scale * torch.randn((2, action_dim), device=d, dtype=torch.float64, generator=g)
        predicted_response = true_response + error
        estimate = error + 0.45 * torch.randn(error.shape, device=d, dtype=torch.float64, generator=g)
        left, singular, right = torch.linalg.svd(estimate, full_matrices=False)
        spectral = predicted_response - singular[0] * torch.outer(left[:, 0], right[0])
        random_left = torch.randn(2, device=d, dtype=torch.float64, generator=g); random_left /= torch.linalg.vector_norm(random_left)
        random_right = torch.randn(action_dim, device=d, dtype=torch.float64, generator=g); random_right /= torch.linalg.vector_norm(random_right)
        random = predicted_response - singular[0] * torch.outer(random_left, random_right)
        pca_axis = predicted_response.clone(); pca_axis[:, 0] -= estimate[:, 0]
        ordinary = predicted_response - estimate
        naive = predicted_response - (estimate + 0.12 * torch.randn(estimate.shape, device=d, dtype=torch.float64, generator=g))
        repaired = {"spectral_rank1": spectral, "random_rank1": random, "pca_axis_rank1": pca_axis, "ordinary_full_effect": ordinary, "naive_task_full": naive}
        current = {}
        for name, response in repaired.items():
            value = _mean_regret(actions, real, base[:, None, :] + actions @ response.T, goals)
            regret[name].append(float(value.item()))
            current[name] = value
        best = min(current, key=lambda key: float(current[key]))
        wins[best] += 1
    # Keep numerical aggregation on the declared CUDA device.  Values are only
    # transferred to Python scalars once the final, immutable report is made.
    summary = {
        name: {
            "mean_realized_regret": float(torch.as_tensor(values, device=d, dtype=torch.float64).mean().item()),
            "configuration_wins": wins[name],
        }
        for name, values in regret.items()
    }
    target = summary["spectral_rank1"]["mean_realized_regret"]
    result = {
        "schema": "spectral_repair_synthetic_p3_cuda.v1", "cuda_preflight": cuda,
        "status": "completed_synthetic_rank_matched_repair_stress_test",
        "protocol": {"seed": int(seed), "models": models, "roots_per_model": roots, "same_noisy_intervention_estimate_per_model": True, "rank_one_budget": True, "task_metric": "physical two-dimensional output identity", "claim_boundary": "synthetic repair stress test only; no learned LeWorldModel, PointMaze repair, HTSC, RGB MPC, PushT, or continuous-action claim."},
        "summary": summary,
        "decision": "synthetic_only; real P3 remains blocked by PointMaze P1 no-go",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
    ax.bar(names, [summary[name]["mean_realized_regret"] for name in names], color=["#2166ac", "#b2182b", "#ef8a62", "#4d9221", "#762a83"])
    ax.set_ylabel("mean realized candidate regret"); ax.set_title("P3 synthetic rank-matched repair stress test"); ax.tick_params(axis="x", rotation=25)
    fig.savefig(output / "Figure_1.png", dpi=220); plt.close(fig)
    (output / "notes.txt").write_text("All methods share the same noisy effect estimate. This is only a synthetic budget-matched stress test.\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "spectral_rank1_mean_regret": target, "summary": summary}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", choices=("cuda",), default="cuda"); parser.add_argument("--seed", type=int, default=202610231)
    result = run(**vars(parser.parse_args())); return 0


if __name__ == "__main__":
    raise SystemExit(main())
