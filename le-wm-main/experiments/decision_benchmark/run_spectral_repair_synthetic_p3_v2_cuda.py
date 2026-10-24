"""CUDA-only P3v2 calibration with a task-weighted, rank-matched repair test.

The legacy P3 stress test compared a rank-one spectral correction to an
unrestricted full-matrix correction and applied an unweighted SVD.  That
comparison cannot test whether a weak but task-critical mode is preferentially
recovered.  This successor keeps the old result immutable and tests only the
declared capacity-matched question: does a rank-one correction selected in the
task metric beat rank-one ordinary-SVD, PCA-axis, and random controls?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "spectral_repair_synthetic_p3_v2_cuda.v1"
RANK_MATCHED_ARMS = ("decision_weighted_rank1", "ordinary_svd_rank1", "pca_axis_rank1", "random_rank1")


def _cost(values: torch.Tensor, goals: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    delta = values - goals[:, None, :]
    return torch.einsum("rci,ij,rcj->rc", delta, metric, delta)


def _mean_regret(
    actions: torch.Tensor, real: torch.Tensor, response: torch.Tensor, base: torch.Tensor,
    goals: torch.Tensor, metric: torch.Tensor,
) -> torch.Tensor:
    predicted = base[:, None, :] + actions @ response.T
    selected = _cost(predicted, goals, metric).argmin(dim=1)
    realized = _cost(real, goals, metric)
    return (realized.gather(1, selected[:, None]).squeeze(1) - realized.min(dim=1).values).mean()


def _rank_one(matrix: torch.Tensor) -> torch.Tensor:
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    return singular[0] * torch.outer(left[:, 0], right[0])


def _decision_weighted_rank_one(estimate: torch.Tensor, metric_sqrt: torch.Tensor) -> torch.Tensor:
    """Best rank-one correction in the declared task-output metric."""

    weighted = metric_sqrt @ estimate
    return torch.linalg.solve(metric_sqrt, _rank_one(weighted))


def run(*, output: Path, device: str = "cuda", seed: int = 202610240, task_weight: float = 64.0) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device != "cuda":
        raise ValueError("P3v2 synthetic repair calibration is CUDA-only; CPU fallback is forbidden")
    if not torch.isfinite(torch.tensor(float(task_weight))) or float(task_weight) <= 1.0:
        raise ValueError("task-weight must be finite and greater than one")
    cuda = require_cuda_allocation(device)
    torch_device, dtype = torch.device(device), torch.float64
    generator = torch.Generator(device=torch_device).manual_seed(int(seed))
    models, roots, candidates, action_dim = 40, 192, 48, 4
    metric = torch.diag(torch.tensor([float(task_weight), 1.0], device=torch_device, dtype=dtype))
    metric_sqrt = torch.diag(torch.tensor([float(task_weight) ** 0.5, 1.0], device=torch_device, dtype=dtype))

    # The action direction with the largest raw variance is nuisance-dominant;
    # the lower-energy second direction controls the strongly weighted task
    # output.  This is the claimed P3 mechanism, made explicit rather than
    # hidden in an unweighted Frobenius reconstruction objective.
    action_scale = torch.tensor([3.0, 0.35, 1.0, 1.0], device=torch_device, dtype=dtype)
    actions = torch.randn((roots, candidates, action_dim), device=torch_device, dtype=dtype, generator=generator) * action_scale
    true_response = torch.tensor([[0.18, 0.95, 0.0, 0.0], [0.85, 0.0, 0.12, 0.0]], device=torch_device, dtype=dtype)
    base = 0.35 * torch.randn((roots, 2), device=torch_device, dtype=dtype, generator=generator)
    goals = 0.55 * torch.randn((roots, 2), device=torch_device, dtype=dtype, generator=generator)
    real = base[:, None, :] + actions @ true_response.T
    regret = {name: [] for name in (*RANK_MATCHED_ARMS, "full_effect_capacity_reference", "no_repair")}

    for index in range(models):
        nuisance_scale = 0.70 + 0.40 * index / (models - 1)
        decision_scale = 0.11 + 0.09 * index / (models - 1)
        error = torch.zeros((2, action_dim), device=torch_device, dtype=dtype)
        error[1, 0] = nuisance_scale
        error[0, 1] = decision_scale
        estimate = error + 0.015 * torch.randn(error.shape, device=torch_device, dtype=dtype, generator=generator)
        predicted_response = true_response + error
        correction = {
            "decision_weighted_rank1": _decision_weighted_rank_one(estimate, metric_sqrt),
            "ordinary_svd_rank1": _rank_one(estimate),
            "pca_axis_rank1": torch.outer(estimate[:, 0], torch.tensor([1.0, 0.0, 0.0, 0.0], device=torch_device, dtype=dtype)),
            "full_effect_capacity_reference": estimate,
            "no_repair": torch.zeros_like(estimate),
        }
        random_left = torch.randn(2, device=torch_device, dtype=dtype, generator=generator)
        random_right = torch.randn(action_dim, device=torch_device, dtype=dtype, generator=generator)
        random_left /= torch.linalg.vector_norm(random_left); random_right /= torch.linalg.vector_norm(random_right)
        correction["random_rank1"] = torch.linalg.vector_norm(correction["decision_weighted_rank1"]) * torch.outer(random_left, random_right)
        for name, value in correction.items():
            regret[name].append(float(_mean_regret(
                actions, real, predicted_response - value, base, goals, metric,
            ).item()))

    summary = {
        name: {"mean_realized_regret": float(torch.as_tensor(values, device=torch_device, dtype=dtype).mean().item())}
        for name, values in regret.items()
    }
    target = summary["decision_weighted_rank1"]["mean_realized_regret"]
    matched_pass = all(target < summary[name]["mean_realized_regret"] for name in RANK_MATCHED_ARMS if name != "decision_weighted_rank1")
    gates = {
        "cuda_allocation": True,
        "decision_weighted_rank1_beats_all_rank_matched_controls": matched_pass,
        "full_effect_is_capacity_reference_not_primary_gate": True,
    }
    result = {
        "schema": SCHEMA,
        "status": "passed_task_weighted_rank_matched_calibration" if all(gates.values()) else "failed_task_weighted_rank_matched_calibration",
        "cuda_preflight": cuda,
        "protocol": {
            "seed": int(seed), "models": models, "roots_per_model": roots, "candidate_count": candidates,
            "rank_matched_arms": list(RANK_MATCHED_ARMS), "task_metric": f"diag({float(task_weight):g},1) in physical task-output coordinates",
            "task_output_0_weight": float(task_weight),
            "low_energy_task_critical_mode": "output-0/action-1", "high_energy_nuisance_mode": "output-1/action-0",
            "full_effect_capacity_reference_excluded_from_primary_gate": True,
            "claim_boundary": "CUDA synthetic mechanism calibration only; it does not establish a learned LeWorldModel repair, HTSC certificate, RGB MPC, PushT, or transfer result.",
        },
        "summary": summary,
        "gates": gates,
        "decision": "synthetic_only; PointMaze P1 and candidate-bank eligibility remain independent gates for a real repair.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(8.2, 3.7), constrained_layout=True)
    names = tuple(summary)
    axis.bar(names, [summary[name]["mean_realized_regret"] for name in names], color=["#2166ac", "#b2182b", "#ef8a62", "#762a83", "#4d9221", "#777777"])
    axis.set_ylabel("mean realised task-metric regret")
    axis.set_title("P3v2 task-weighted, capacity-matched rank-one repair calibration")
    axis.tick_params(axis="x", rotation=25)
    figure.savefig(output / "Figure_1.png", dpi=220)
    plt.close(figure)
    (output / "notes.txt").write_text(
        "Primary gate compares only equal-rank repairs. The full-effect arm is a capacity reference and cannot defeat a rank-one method by construction.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "gates": gates, "summary": summary}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=202610240)
    parser.add_argument("--task-weight", type=float, default=64.0)
    result = run(**vars(parser.parse_args()))
    return 0 if result["status"] == "passed_task_weighted_rank_matched_calibration" else 2


if __name__ == "__main__":
    raise SystemExit(main())
