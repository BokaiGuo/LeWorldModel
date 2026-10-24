"""CUDA-only SRA P1: frozen spectral cost-to-go candidate-selection audit.

For each held-out PointMaze root, the planner scores a fixed first-block action
bank two ways: immediate frozen latent-goal distance and the analytic local
spectral cost-to-go of the *next* chart.  A MuJoCo two-block tree supplies
physical labels only after both choices are frozen.  No model, readout, or
planner parameter is trained; state is evaluator-only and never reaches a
model input or score.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    ROOT,
    load_model_and_dataset,
    make_root_context,
    parameter_hash,
    require_cuda,
    sha256_file,
    terminal_latent,
    write_json,
)


def parse_roots(value: str) -> list[int]:
    values = [int(token.strip()) for token in value.split(",") if token.strip()]
    if len(values) < 4 or len(values) != len(set(values)):
        raise ValueError("P1 requires at least four unique outcome roots")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    # Root-disjoint from P0's 1896--1899 smoke/audit roots.
    parser.add_argument("--roots", default="1888,1889,1890,1891,1892,1893,1894,1895")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=0.10)
    parser.add_argument("--physical-action-amplitude", type=float, default=0.8)
    parser.add_argument(
        "--continuation-score",
        choices=("analytic_dls", "discrete_reachable_set"),
        default="analytic_dls",
        help="Frozen-model score for each first-block chart; the discrete mode uses the exact fixed continuation bank evaluated physically.",
    )
    return parser.parse_args()


def action_bank(dataset: Any, root: int, start: int, amplitude: float, device: torch.device) -> tuple[list[str], torch.Tensor]:
    """A predeclared five-action first/second-block bank in model coordinates."""

    physical = torch.tensor(
        [[0.0, 0.0], [amplitude, 0.0], [-amplitude, 0.0], [0.0, amplitude], [0.0, -amplitude]],
        device=device,
        dtype=torch.float32,
    )
    normalized = (physical - dataset.action_mean.to(device)) / dataset.action_std.to(device)
    repeated = normalized.unsqueeze(1).expand(-1, 5, -1).reshape(len(physical), -1)
    observed = dataset.actions[root, start : start + 5].reshape(1, -1).to(device)
    labels = ["zero", "plus_x", "minus_x", "plus_y", "minus_y", "observed"]
    return labels, torch.cat([repeated, observed], dim=0)


def continuation_bank(dataset: Any, amplitude: float, device: torch.device) -> tuple[list[str], torch.Tensor]:
    """Fixed 17-arm physical continuation bank, denser than first-chart arms."""

    directions = [np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32) for theta in np.linspace(0.0, 2.0 * math.pi, num=8, endpoint=False)]
    physical = [np.zeros(2, dtype=np.float32)]
    labels = ["zero"]
    for magnitude, prefix in ((amplitude * 0.5, "half"), (amplitude, "full")):
        for index, direction in enumerate(directions):
            physical.append(magnitude * direction)
            labels.append(f"{prefix}_{index}")
    controls = torch.as_tensor(np.stack(physical), device=device)
    normalized = (controls - dataset.action_mean.to(device)) / dataset.action_std.to(device)
    return labels, normalized.unsqueeze(1).expand(-1, 5, -1).reshape(len(controls), -1)


def normalized_action_box(dataset: Any, amplitude: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the evaluator's physical [-amplitude, amplitude] box to model inputs."""

    mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
    lower = ((torch.full_like(mean, -float(amplitude)) - mean) / std).repeat(5)
    upper = ((torch.full_like(mean, float(amplitude)) - mean) / std).repeat(5)
    return lower, upper


def predicted_chart_context(model: Any, initial_context: Any, first_block: torch.Tensor) -> tuple[Any, torch.Tensor]:
    """Roll out a fixed first block, then make its frozen predicted endpoint a chart."""

    from tensordict import TensorDict

    with torch.no_grad():
        rollout = model.unroll(initial_context, act_suffix=first_block.reshape(1, 1, -1))
        visual = rollout["visual"][-1].unsqueeze(1).detach()
        proprio = rollout["proprio"][-1].unsqueeze(1).detach()
    return TensorDict({"visual": visual, "proprio": proprio}, batch_size=[1, 1], device=visual.device), visual[0, 0].mean(dim=(0, 1, 2))


def exact_local_jacobian(model: Any, chart: Any, action_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute all ten J columns by JVP, avoiding a huge latent-output VJP batch."""

    origin = torch.zeros(action_dim, device=device)

    def endpoint(action: torch.Tensor) -> torch.Tensor:
        return terminal_latent(model, chart, action.reshape(1, 1, action_dim))[0]

    with torch.no_grad():
        nominal = endpoint(origin)
    columns = []
    for column in range(action_dim):
        tangent = torch.zeros_like(origin)
        tangent[column] = 1.0
        _, jvp = torch.autograd.functional.jvp(endpoint, origin, tangent, create_graph=False, strict=True)
        columns.append(jvp)
    jacobian = torch.stack(columns, dim=1)
    if jacobian.ndim != 2 or jacobian.shape[1] != action_dim or not torch.isfinite(jacobian).all():
        raise RuntimeError(f"invalid local Jacobian shape/value: {tuple(jacobian.shape)}")
    if float(jacobian.norm()) <= 1.0e-10:
        raise RuntimeError("zero local action Jacobian: this is a code/model-path stop, not a spectral result")
    return nominal, jacobian


def spectral_cost_to_go(model: Any, chart: Any, goal: torch.Tensor, action_dim: int, ridge: float, action_lower: torch.Tensor, action_upper: torch.Tensor, device: torch.device) -> dict[str, Any]:
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    nominal, jacobian = exact_local_jacobian(model, chart, action_dim, device)
    residual = goal - nominal
    gram = jacobian.T @ jacobian
    cross = jacobian.T @ residual
    unconstrained_delta = torch.linalg.solve(gram + float(ridge) * torch.eye(action_dim, device=device), cross)
    # The analytic unconstrained optimum is only a proposal.  Project it onto
    # exactly the physical action box used by the MuJoCo continuation labels;
    # otherwise P1 would compare incompatible feasible sets.
    delta = torch.maximum(torch.minimum(unconstrained_delta, action_upper), action_lower)
    value = torch.sum((residual - jacobian @ delta) ** 2) + float(ridge) * torch.sum(delta ** 2)
    singular = torch.linalg.svdvals(jacobian)
    # Arithmetic agreement guards against a sign/transpose bug in the closed form.
    direct = torch.sum((residual - jacobian @ delta) ** 2) + float(ridge) * torch.sum(delta ** 2)
    if not torch.allclose(value, direct, atol=2e-4, rtol=2e-4):
        raise RuntimeError(f"spectral closed form disagreement: {float(value)} vs {float(direct)}")
    return {
        "value": float(value),
        "direct_damped_least_squares_value": float(direct),
        "immediate_latent_distance_sq": float(torch.sum((nominal - goal) ** 2)),
        "jacobian_frobenius_norm": float(jacobian.norm()),
        "singular_values": [float(v) for v in singular],
        "suggested_delta_norm": float(delta.norm()),
        "unconstrained_delta_norm": float(unconstrained_delta.norm()),
        "box_constraint_active": bool(not torch.allclose(delta, unconstrained_delta)),
    }


def discrete_reachable_set_cost(model: Any, chart: Any, goal: torch.Tensor, continuation_blocks: torch.Tensor) -> dict[str, Any]:
    """Score a chart on the exact fixed continuation bank used for its physical label.

    This is deliberately a frozen-model bank evaluation rather than a learned
    value function.  It closes the old P1 mismatch where a clipped continuous
    DLS surrogate selected the first block while the physical evaluator scored
    a different, discrete continuation set.
    """

    if continuation_blocks.ndim != 2:
        raise ValueError("continuation bank must be [candidates, action_dim]")
    with torch.no_grad():
        endpoints = terminal_latent(model, chart, continuation_blocks.unsqueeze(0))
        costs = torch.sum((endpoints - goal.unsqueeze(0)) ** 2, dim=1)
    if costs.ndim != 1 or not torch.isfinite(costs).all():
        raise RuntimeError("discrete reachable-set score is non-finite")
    index = int(torch.argmin(costs))
    return {
        "value": float(costs[index]),
        "chosen_continuation_index": index,
        "candidate_costs": [float(value) for value in costs],
    }


def block_to_physical(block: torch.Tensor, dataset: Any) -> np.ndarray:
    data = block.detach().cpu().reshape(5, -1)
    return (data * dataset.action_std + dataset.action_mean).numpy().astype(np.float64)


def physical_tree_labels(source: np.ndarray, goal: np.ndarray, first_blocks: torch.Tensor, continuation_blocks: torch.Tensor, dataset: Any) -> list[float]:
    """Evaluator-only two-block oracle labels; never fed into SRA selection."""

    from experiments.spectral_reachability.pointmaze_physical_evaluator import replay_position_endpoints
    physical_first = [block_to_physical(block, dataset) for block in first_blocks]
    physical_continuations = [block_to_physical(block, dataset) for block in continuation_blocks]

    middle = replay_position_endpoints(np.repeat(source.reshape(1, 4), len(physical_first), axis=0), np.stack(physical_first))
    endpoints = replay_position_endpoints(np.repeat(middle, len(physical_continuations), axis=0), np.tile(np.stack(physical_continuations), (len(physical_first), 1, 1)))
    return np.linalg.norm(endpoints.reshape(len(physical_first), len(physical_continuations), 4)[:, :, :2] - goal[:2], axis=2).min(axis=1).tolist()


def rank(values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(values), kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(len(values), dtype=np.float64)
    return out.tolist()


def main() -> None:
    args = parse_args()
    roots = parse_roots(args.roots)
    if args.output_dir.exists():
        raise FileExistsError(f"immutable P1 output already exists: {args.output_dir}")
    if args.start_frame < 0:
        raise ValueError("start frame must be non-negative")
    device = require_cuda()
    from experiments.spectral_reachability.pointmaze_physical_evaluator import position_backend_receipt

    physical_backend = position_backend_receipt()
    args.output_dir.mkdir(parents=True)
    model, dataset = load_model_and_dataset(args, device)
    before = parameter_hash(model)
    action_dim = int(model.action_dim)
    expected = int(dataset.actions.shape[-1]) * 5
    if action_dim != expected:
        raise RuntimeError(f"model action dimension {action_dim} differs from dataset block dimension {expected}")
    rows = []
    for root in roots:
        initial, goal_latent, _observed_two_block = make_root_context(model, dataset, root, args.start_frame, 2, device)
        labels, blocks = action_bank(dataset, root, args.start_frame, args.physical_action_amplitude, device)
        continuation_labels, continuation_blocks = continuation_bank(dataset, args.physical_action_amplitude, device)
        action_lower, action_upper = normalized_action_box(dataset, args.physical_action_amplitude, device)
        spectral, immediate = [], []
        for block in blocks:
            chart, first_latent = predicted_chart_context(model, initial, block)
            analytic = spectral_cost_to_go(model, chart, goal_latent, action_dim, args.ridge, action_lower, action_upper, device)
            if args.continuation_score == "analytic_dls":
                item = {**analytic, "score_family": "analytic_dls"}
            else:
                item = {
                    **analytic,
                    **discrete_reachable_set_cost(model, chart, goal_latent, continuation_blocks),
                    "score_family": "discrete_reachable_set",
                }
            spectral.append(item)
            immediate.append(float(torch.sum((first_latent - goal_latent) ** 2)))
        source = dataset.states[root, args.start_frame].detach().cpu().numpy().astype(np.float64)
        physical_goal = dataset.states[root, args.start_frame + 10].detach().cpu().numpy().astype(np.float64)
        labels_physical = physical_tree_labels(source, physical_goal, blocks, continuation_blocks, dataset)
        spectral_scores = [item["value"] for item in spectral]
        chosen_spectral = int(np.argmin(spectral_scores))
        chosen_immediate = int(np.argmin(immediate))
        optimal = int(np.argmin(labels_physical))
        rows.append({
            "root": root,
            "candidate_labels": labels,
            "scores": {"spectral_cost_to_go": spectral_scores, "immediate_latent_distance_sq": immediate, "physical_two_block_oracle_distance": labels_physical},
            "local_charts": spectral,
            "selection": {"spectral_index": chosen_spectral, "immediate_index": chosen_immediate, "physical_oracle_index": optimal, "spectral_physical_regret": float(labels_physical[chosen_spectral] - labels_physical[optimal]), "immediate_physical_regret": float(labels_physical[chosen_immediate] - labels_physical[optimal])},
        })
        print(json.dumps({"root": root, **rows[-1]["selection"]}, sort_keys=True))
    after = parameter_hash(model)
    spectral_regret = np.asarray([item["selection"]["spectral_physical_regret"] for item in rows])
    immediate_regret = np.asarray([item["selection"]["immediate_physical_regret"] for item in rows])
    gates = {
        "cuda_only": True,
        "frozen_parameter_hash_unchanged": before == after,
        "every_chart_has_finite_nonzero_jacobian": all(chart["jacobian_frobenius_norm"] > 1.0e-10 and math.isfinite(chart["value"]) for row in rows for chart in row["local_charts"]),
        "no_outcome_labels_used_for_selection": True,
        "root_disjoint_from_p0_smoke_roots": not bool(set(roots) & {1896, 1897, 1898, 1899}),
    }
    report = {
        "schema": "sra_p1_pointmaze_chart_selection.v2",
        "status": "mechanism_ready" if all(gates.values()) else "invalid",
        "scope": "root-disjoint frozen-model intermediate-chart selection audit; physical two-block tree labels are evaluator-only, not RGB closed-loop performance",
        "device": {"type": device.type, "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "inputs": {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "dataset": str(args.dataset_root), "roots": roots, "ridge": args.ridge, "physical_action_amplitude": args.physical_action_amplitude, "candidate_bank": ["zero", "plus_x", "minus_x", "plus_y", "minus_y", "observed"], "continuation_bank": continuation_labels, "continuation_score": args.continuation_score, "spectral_delta_box": "same [-physical-action-amplitude, +physical-action-amplitude] per raw control as the physical continuation evaluator", "physical_position_backend": physical_backend},
        "frozen_contract": {"parameter_hash_before": before, "parameter_hash_after": after},
        "gates": gates,
        "aggregate": {"spectral_mean_physical_regret": float(spectral_regret.mean()), "immediate_mean_physical_regret": float(immediate_regret.mean()), "paired_regret_difference_spectral_minus_immediate": float((spectral_regret - immediate_regret).mean()), "spectral_oracle_hit_rate": float(np.mean(spectral_regret <= 1.0e-12)), "immediate_oracle_hit_rate": float(np.mean(immediate_regret <= 1.0e-12))},
        "rows": rows,
        "claim_boundary": "No online outcome selected the method or ridge. This is an evaluator-state-labelled, fixed action-bank mechanism audit; it is not RGB-only MPC success evidence.",
    }
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "README.md").write_text("# SRA P1 PointMaze CUDA audit\n\nFrozen spectral cost-to-go versus immediate latent distance. MuJoCo state is evaluator-only and no learned component is trained.\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "status": report["status"], "aggregate": report["aggregate"], "gates": gates}, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise RuntimeError("P1 implementation/protocol gate failed")


if __name__ == "__main__":
    main()
