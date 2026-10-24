"""CUDA-only SRA P2 prerequisite: verify discrete action-order primitives.

This is not a Lie-bracket claim.  It compares actual swapped finite action
blocks in the frozen PointMaze world model with swapped blocks in the MuJoCo
evaluator.  The world model runs on CUDA; simulator states are evaluator-only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    make_root_context, parameter_hash, require_cuda, sha256_file, terminal_latent, write_json,
)
from experiments.spectral_reachability.run_sra_p1_pointmaze_cuda import block_to_physical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    parser.add_argument("--roots", default="1880,1881,1882,1883,1884,1885,1886,1887")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--physical-action-amplitude", type=float, default=0.8)
    parser.add_argument(
        "--effect-score",
        choices=("magnitude", "goal_projection"),
        default="magnitude",
        help="Magnitude is the historical primitive check; goal_projection tests goal-directed order effects with an RGB-derived latent goal.",
    )
    return parser.parse_args()


def parse_roots(raw: str) -> list[int]:
    roots = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if len(roots) < 4 or len(roots) != len(set(roots)):
        raise ValueError("P2 needs at least four distinct roots")
    return roots


def primitive_bank(dataset: Any, amplitude: float, device: torch.device) -> tuple[list[str], torch.Tensor]:
    physical = torch.tensor([[amplitude, 0.0], [-amplitude, 0.0], [0.0, amplitude], [0.0, -amplitude]], device=device)
    normalized = (physical - dataset.action_mean.to(device)) / dataset.action_std.to(device)
    return ["plus_x", "minus_x", "plus_y", "minus_y"], normalized.unsqueeze(1).expand(-1, 5, -1).reshape(4, -1)


def physical_endpoint(source: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    from experiments.spectral_reachability.pointmaze_physical_evaluator import replay_position_endpoints

    return replay_position_endpoints(source.reshape(1, 4), np.concatenate((first, second), axis=0).reshape(1, -1, 2))[0]


def average_rank(values: np.ndarray) -> np.ndarray:
    """Average tied ranks; physical symmetric actions routinely create ties."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        reference = values[order[start]]
        while stop < len(values) and np.isclose(values[order[stop]], reference, rtol=1.0e-7, atol=1.0e-9):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return float("nan")
    rx = average_rank(x)
    ry = average_rank(y)
    if np.ptp(rx) <= 1e-12 or np.ptp(ry) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def goal_projected_effect(forward: np.ndarray | torch.Tensor, reverse: np.ndarray | torch.Tensor, goal: np.ndarray | torch.Tensor) -> float:
    """Magnitude of a swap-order effect along the local direction to a fixed goal."""

    if isinstance(forward, torch.Tensor):
        midpoint = 0.5 * (forward + reverse)
        residual = goal - midpoint
        denominator = torch.linalg.vector_norm(residual)
        if not torch.isfinite(denominator) or float(denominator) <= 1.0e-8:
            raise RuntimeError("goal-projection order effect has a degenerate latent residual")
        return float(torch.abs(torch.dot(forward - reverse, residual)) / denominator)
    midpoint = 0.5 * (forward + reverse)
    residual = goal - midpoint
    denominator = float(np.linalg.norm(residual))
    if not np.isfinite(denominator) or denominator <= 1.0e-8:
        raise RuntimeError("goal-projection order effect has a degenerate physical residual")
    return float(abs(np.dot(forward - reverse, residual)) / denominator)


def main() -> None:
    args = parse_args()
    roots = parse_roots(args.roots)
    if args.output_dir.exists():
        raise FileExistsError(f"immutable P2 output exists: {args.output_dir}")
    if set(roots) & set(range(1888, 1900)):
        raise ValueError("P2 roots must be disjoint from the P0/P1 root panels")
    device = require_cuda()
    from experiments.spectral_reachability.pointmaze_physical_evaluator import position_backend_receipt

    physical_backend = position_backend_receipt()
    args.output_dir.mkdir(parents=True)
    model, dataset = load_model_and_dataset(args, device)
    before = parameter_hash(model)
    labels, blocks = primitive_bank(dataset, args.physical_action_amplitude, device)
    if int(model.action_dim) != int(blocks.shape[-1]):
        raise RuntimeError("primitive block shape does not match published predictor action interface")
    physical_blocks = [block_to_physical(block, dataset) for block in blocks]
    rows: list[dict[str, Any]] = []
    all_pred, all_actual = [], []
    for root in roots:
        initial, goal_latent, _observed = make_root_context(model, dataset, root, args.start_frame, 2, device)
        sequences, pairs = [], []
        for left, right in itertools.combinations(range(len(labels)), 2):
            pairs.append((left, right))
            sequences.extend([torch.stack([blocks[left], blocks[right]]), torch.stack([blocks[right], blocks[left]])])
        suffix = torch.stack(sequences, dim=1)
        with torch.no_grad():
            endpoints = terminal_latent(model, initial, suffix)
        source = dataset.states[root, args.start_frame].detach().cpu().numpy().astype(np.float64)
        physical_goal = dataset.states[root, args.start_frame + 10].detach().cpu().numpy().astype(np.float64)
        root_pairs = []
        for pair_index, (left, right) in enumerate(pairs):
            predicted_forward, predicted_reverse = endpoints[2 * pair_index], endpoints[2 * pair_index + 1]
            forward = physical_endpoint(source, physical_blocks[left], physical_blocks[right])
            reverse = physical_endpoint(source, physical_blocks[right], physical_blocks[left])
            if args.effect_score == "magnitude":
                predicted_effect = float(torch.linalg.vector_norm(predicted_forward - predicted_reverse))
                actual_effect = float(np.linalg.norm(forward[:2] - reverse[:2]))
            else:
                predicted_effect = goal_projected_effect(predicted_forward, predicted_reverse, goal_latent)
                actual_effect = goal_projected_effect(forward[:2], reverse[:2], physical_goal[:2])
            root_pairs.append({"first": labels[left], "second": labels[right], "predicted_latent_order_effect_l2": predicted_effect, "physical_position_order_effect_l2": actual_effect})
            all_pred.append(predicted_effect)
            all_actual.append(actual_effect)
        rows.append({"root": root, "pairs": root_pairs, "root_spearman": spearman(np.asarray([item["predicted_latent_order_effect_l2"] for item in root_pairs]), np.asarray([item["physical_position_order_effect_l2"] for item in root_pairs]))})
        print(json.dumps({"root": root, "root_spearman": rows[-1]["root_spearman"], "min_predicted": min(item["predicted_latent_order_effect_l2"] for item in root_pairs), "min_physical": min(item["physical_position_order_effect_l2"] for item in root_pairs)}, sort_keys=True))
    after = parameter_hash(model)
    finite_root_correlations = [item["root_spearman"] for item in rows if math.isfinite(item["root_spearman"])]
    gates = {
        "cuda_only": True,
        "frozen_parameter_hash_unchanged": before == after,
        "model_order_effect_nonzero_every_root": all(max(pair["predicted_latent_order_effect_l2"] for pair in item["pairs"]) > 1e-7 for item in rows),
        "physical_order_effect_nonzero_every_root": all(max(pair["physical_position_order_effect_l2"] for pair in item["pairs"]) > 1e-8 for item in rows),
        "root_disjoint_from_p0_p1": True,
    }
    report = {
        "schema": "sra_p2_pointmaze_discrete_action_order.v2",
        "status": "mechanism_ready" if all(gates.values()) else "invalid",
        "scope": "necessary action-order primitive audit only; no multi-chart-search or RGB closed-loop performance claim",
        "device": {"type": device.type, "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "inputs": {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "roots": roots, "physical_action_amplitude": args.physical_action_amplitude, "effect_score": args.effect_score, "primitive_actions": labels, "physical_position_backend": physical_backend},
        "frozen_contract": {"parameter_hash_before": before, "parameter_hash_after": after},
        "gates": gates,
        "aggregate": {"pair_count": len(all_pred), "global_spearman_predicted_vs_physical_order_effect": spearman(np.asarray(all_pred), np.asarray(all_actual)), "mean_root_spearman": float(np.mean(finite_root_correlations)) if finite_root_correlations else float("nan")},
        "rows": rows,
        "claim_boundary": "This uses actual discrete swapped rollouts. It does not identify a smooth Lie bracket and cannot by itself justify atlas search or task improvement.",
    }
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"output": str(args.output_dir), "status": report["status"], "gates": gates, "aggregate": report["aggregate"]}, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise RuntimeError("P2 action-order prerequisite failed")


if __name__ == "__main__":
    main()
