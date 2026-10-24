"""CUDA-only P2 calibration for the Decision-Spectral Mismatch computation.

This deliberately controlled linear intervention world contains a two-
dimensional task output and high-dimensional action-dependent nuisance output.
Model configurations have independently varied task-response and nuisance
response errors.  The preregistered question is whether the task-weighted
response mismatch tracks actual candidate-ranking regret better than generic
latent endpoint MSE.  It validates computation, not a learned-world-model
claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "decision_spectral_synthetic_p2_cuda.v1"


def _root_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.transpose(-1, -2)))
    if torch.any(values <= 0):
        raise RuntimeError("declared metric/candidate covariance is not positive definite")
    return (vectors * values.sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = torch.arange(1, values.numel() + 1, device=values.device, dtype=torch.float64)
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    x, y = _rank(left), _rank(right)
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1].item())


def _permutation_p(left: torch.Tensor, right: torch.Tensor, *, seed: int, draws: int = 20000) -> float:
    generator = torch.Generator(device=left.device).manual_seed(seed)
    observed = abs(_spearman(left, right))
    count = 0
    for _ in range(draws):
        permutation = torch.randperm(len(right), device=right.device, generator=generator)
        count += int(abs(_spearman(left, right[permutation])) >= observed)
    return float((count + 1) / (draws + 1))


def _plot(path: Path, mismatch: list[float], generic: list[float], regret: list[float]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), constrained_layout=True)
    for axis, x, title in zip(
        axes,
        (mismatch, generic),
        ("Task-output decision-spectral mismatch", "Generic latent endpoint MSE"),
    ):
        axis.scatter(x, regret, s=22, color="#2166ac", alpha=0.85)
        axis.set_xlabel(title)
        axis.set_ylabel("real candidate-ranking regret")
        axis.grid(alpha=0.25)
    figure.suptitle("Controlled P2 mechanism calibration; each dot is one fixed model configuration")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(*, output: Path, device: str = "cuda", seed: int = 202610230, model_count: int = 48,
        root_count: int = 128, candidate_count: int = 32, action_dimension: int = 4,
        nuisance_dimension: int = 16) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device != "cuda":
        raise ValueError("P2 synthetic mechanism calibration is CUDA-only; CPU fallback is forbidden")
    if min(model_count, root_count, candidate_count, action_dimension, nuisance_dimension) < 2:
        raise ValueError("all synthetic dimensions/counts must be at least two")
    cuda = require_cuda_allocation(device)
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(int(seed))
    dtype = torch.float64
    print("[P2] CUDA-only controlled task-response versus nuisance-response mechanism calibration")

    actions = torch.randn((root_count, candidate_count, action_dimension), device=torch_device, dtype=dtype, generator=generator)
    action_centered = actions - actions.mean(dim=1, keepdim=True)
    covariance = action_centered.transpose(1, 2) @ action_centered / float(candidate_count - 1)
    covariance = covariance + 1.0e-5 * torch.eye(action_dimension, device=torch_device, dtype=dtype).expand(root_count, -1, -1)
    covariance_sqrt = _root_sqrt(covariance)
    true_response = torch.randn((2, action_dimension), device=torch_device, dtype=dtype, generator=generator)
    true_response = true_response / torch.linalg.vector_norm(true_response, dim=1, keepdim=True)
    nuisance_response = 2.5 * torch.randn((nuisance_dimension, action_dimension), device=torch_device, dtype=dtype, generator=generator)
    root_base = 0.4 * torch.randn((root_count, 2), device=torch_device, dtype=dtype, generator=generator)
    goals = 0.6 * torch.randn((root_count, 2), device=torch_device, dtype=dtype, generator=generator)
    true_task = root_base[:, None, :] + actions @ true_response.T
    true_nuisance = actions @ nuisance_response.T

    task_scales = torch.linspace(0.0, 1.2, model_count, device=torch_device, dtype=dtype)
    records: list[dict[str, float]] = []
    mismatch_values, generic_values, regret_values = [], [], []
    for index, scale in enumerate(task_scales):
        task_direction = torch.randn(true_response.shape, device=torch_device, dtype=dtype, generator=generator)
        task_direction = task_direction / torch.linalg.vector_norm(task_direction)
        nuisance_error = torch.randn(nuisance_response.shape, device=torch_device, dtype=dtype, generator=generator)
        nuisance_error = nuisance_error / torch.linalg.vector_norm(nuisance_error)
        nuisance_scale = 0.25 + 2.75 * torch.rand((), device=torch_device, dtype=dtype, generator=generator)
        predicted_response = true_response + scale * task_direction
        predicted_task = root_base[:, None, :] + actions @ predicted_response.T
        predicted_nuisance = actions @ (nuisance_response + nuisance_scale * nuisance_error).T
        predicted_cost = ((predicted_task - goals[:, None, :]) ** 2).sum(dim=-1)
        realized_cost = ((true_task - goals[:, None, :]) ** 2).sum(dim=-1)
        selected = predicted_cost.argmin(dim=1)
        selected_cost = realized_cost.gather(1, selected[:, None]).squeeze(1)
        regret = (selected_cost - realized_cost.min(dim=1).values).mean()
        delta = predicted_response - true_response
        operators = delta.expand(root_count, -1, -1) @ covariance_sqrt
        spectral = torch.linalg.svdvals(operators)[:, 0].mean()
        generic = torch.cat((predicted_task - true_task, predicted_nuisance - true_nuisance), dim=-1).square().mean().sqrt()
        record = {
            "configuration": float(index), "declared_task_error_scale": float(scale.item()),
            "nuisance_error_scale": float(nuisance_scale.item()), "decision_spectral_norm": float(spectral.item()),
            "generic_latent_endpoint_rmse": float(generic.item()), "mean_realized_regret": float(regret.item()),
        }
        records.append(record)
        mismatch_values.append(spectral)
        generic_values.append(generic)
        regret_values.append(regret)

    mismatch = torch.stack(mismatch_values)
    generic = torch.stack(generic_values)
    regret = torch.stack(regret_values)
    metric_spearman = _spearman(mismatch, regret)
    generic_spearman = _spearman(generic, regret)
    # Gauge check in task-output coordinates: y'=Ty, Q'=T^-T Q T^-1.
    transform = torch.tensor([[1.1, -0.3], [0.25, 0.9]], device=torch_device, dtype=dtype)
    transformed_metric = torch.linalg.inv(transform).T @ torch.linalg.inv(transform)
    original_singular = torch.linalg.svdvals((predicted_response - true_response) @ covariance_sqrt[0])
    transformed_singular = torch.linalg.svdvals(
        _root_sqrt(transformed_metric) @ transform @ (predicted_response - true_response) @ covariance_sqrt[0]
    )
    gauge_max_abs_error = float((original_singular - transformed_singular).abs().max().item())
    gates = {
        "cuda_allocation": True,
        "gauge_invariant_to_tolerance": gauge_max_abs_error <= 1.0e-10,
        "decision_mismatch_spearman_at_least_0p65": metric_spearman >= 0.65,
        "decision_mismatch_exceeds_generic_by_0p20": metric_spearman - generic_spearman >= 0.20,
    }
    status = "passed_synthetic_formula_calibration" if all(gates.values()) else "failed_synthetic_formula_calibration"
    result = {
        "schema": SCHEMA, "status": status, "cuda_preflight": cuda,
        "protocol": {
            "seed": int(seed), "model_count": int(model_count), "root_count": int(root_count),
            "candidate_count": int(candidate_count), "action_dimension": int(action_dimension),
            "task_output_dimension": 2, "nuisance_dimension": int(nuisance_dimension),
            "task_metric": "identity in true physical task-output coordinates",
            "outcome_unit": "model configuration averaged over independently sampled roots",
            "preregistered_gates": gates,
            "claim_boundary": "CUDA synthetic formula calibration only; not learned visual-world-model, continuous-action, intervention-repair, HTSC, or RGB-MPC evidence.",
        },
        "summary": {
            "decision_spectral_mismatch_vs_regret_spearman": metric_spearman,
            "generic_latent_rmse_vs_regret_spearman": generic_spearman,
            "decision_spectral_two_sided_permutation_p": _permutation_p(mismatch, regret, seed=seed + 101),
            "generic_latent_two_sided_permutation_p": _permutation_p(generic, regret, seed=seed + 102),
            "gauge_singular_value_max_abs_error": gauge_max_abs_error,
        },
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(output / "Figure_1.png", [value["decision_spectral_norm"] for value in records], [value["generic_latent_endpoint_rmse"] for value in records], [value["mean_realized_regret"] for value in records])
    (output / "notes.txt").write_text(
        "P2 GPU-only synthetic calibration. Task-response and nuisance-response errors are independently sampled. "
        "Pass only validates formula recovery in this declared linear construction; it is not a real-world-model repair result.\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# P2 decision-spectral synthetic calibration\n\n"
        "This CUDA-only controlled linear system checks task-output mismatch, nuisance separation, candidate regret, and gauge invariance. "
        "It must not be cited as PointMaze, PushT, HTSC, or RGB-only MPC evidence.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "summary": result["summary"]}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=202610230)
    args = parser.parse_args()
    result = run(**vars(args))
    return 0 if result["status"] == "passed_synthetic_formula_calibration" else 2


if __name__ == "__main__":
    raise SystemExit(main())
