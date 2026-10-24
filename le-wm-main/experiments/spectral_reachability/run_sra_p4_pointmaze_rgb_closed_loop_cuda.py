"""CUDA-only RGB-only closed-loop SRA P4 admission/evaluation runner.

The two controllers share a fixed five-arm physical action bank.  ``spectral``
selects a first model block by the best two-block frozen-model continuation;
``immediate`` scores exactly the same two-block candidate tensor at its first
predicted successor.  Thus both consume the same number of two-step model
rollouts at every replan.  Simulator state is used only to reset/step/evaluate
the environment; model inputs are current rendered RGB, goal RGB, and an RGB
calibration-only proprio proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.run_live_pointmaze_planning import (
    GOAL_STATE,
    DatasetMatchedRenderer,
    LegacyPointMazeRuntime,
    fit_visual_calibration,
    green_center,
    load_episode,
    load_episode_states,
)
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    ROOT,
    load_model_and_dataset,
    parameter_hash,
    require_cuda,
    sha256_file,
)
from experiments.spectral_reachability.run_sra_p1_pointmaze_cuda import action_bank, block_to_physical


SCHEMA = "sra_pointmaze_p4_rgb_closed_loop.v1"
MODEL_BLOCK_HIGH_LEVEL_ACTIONS = 5
FIXED_ACTION_AMPLITUDE = 0.8


def _parse_roots(value: str, label: str, minimum: int) -> list[int]:
    roots = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(roots) < minimum or len(roots) != len(set(roots)) or min(roots, default=0) < 0:
        raise ValueError(f"{label} requires at least {minimum} distinct non-negative roots")
    return roots


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_rgb_proxy_observation(
    image: np.ndarray,
    previous_center: tuple[float, float] | None,
    calibration: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], tuple[float, float], np.ndarray]:
    """Create model input solely from RGB and a calibration-only RGB proxy."""

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB input must be HWC uint8, got {rgb.shape}")
    center = green_center(rgb)
    delta = (0.0, 0.0) if previous_center is None else (center[0] - previous_center[0], center[1] - previous_center[1])
    proxy = np.r_[calibration.position(center), calibration.velocity(delta)].astype(np.float32)
    if proxy.shape != (4,) or not np.isfinite(proxy).all():
        raise RuntimeError("RGB proxy is non-finite or malformed")
    visual = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0).unsqueeze(0).to(device)
    proprio = torch.from_numpy(proxy).reshape(1, 1, 4).to(device)
    return {"visual": visual, "proprio": proprio}, center, proxy


def _encoded_context_and_goal(model: Any, observation: dict[str, torch.Tensor], goal_observation: dict[str, torch.Tensor]) -> tuple[Any, torch.Tensor]:
    from tensordict import TensorDict

    with torch.no_grad():
        encoded = model.encode(observation)
        encoded_goal = model.encode(goal_observation)
    context = TensorDict(
        {"visual": encoded["visual"], "proprio": encoded["proprio"]},
        batch_size=[1, 1], device=observation["visual"].device,
    )
    goal = encoded_goal["visual"][0, 0].mean(dim=(0, 1, 2)).detach()
    return context, goal


def _pooled_visual(rollout: Any, index: int) -> torch.Tensor:
    value = rollout["visual"][index]
    return value.mean(dim=(1, 2, 3))


def select_fixed_tree_action(model: Any, context: Any, goal: torch.Tensor, candidates: torch.Tensor) -> dict[str, Any]:
    """Select both arms from identical 25 two-step candidate computations."""

    if tuple(candidates.shape) != (5, 10):
        raise ValueError(f"fixed action bank must be [5,10], got {tuple(candidates.shape)}")
    suffix = torch.stack([
        torch.stack([first, continuation])
        for first in candidates
        for continuation in candidates
    ], dim=1)
    started = time.perf_counter()
    with torch.no_grad():
        rollout = model.unroll(context, act_suffix=suffix)
        immediate_costs = torch.sum((_pooled_visual(rollout, 0) - goal.unsqueeze(0)) ** 2, dim=1)
        spectral_costs = torch.sum((_pooled_visual(rollout, -1) - goal.unsqueeze(0)) ** 2, dim=1)
    elapsed = time.perf_counter() - started
    immediate_by_first = immediate_costs.reshape(5, 5).min(dim=1).values
    spectral_by_first = spectral_costs.reshape(5, 5).min(dim=1).values
    immediate_index, spectral_index = int(torch.argmin(immediate_by_first)), int(torch.argmin(spectral_by_first))
    return {
        "spectral_index": spectral_index,
        "immediate_index": immediate_index,
        "spectral_first_costs": [float(value) for value in spectral_by_first],
        "immediate_first_costs": [float(value) for value in immediate_by_first],
        "candidate_rollouts": int(suffix.shape[1]),
        "horizon": int(suffix.shape[0]),
        "planner_seconds": float(elapsed),
    }


def _execute_model_block(runtime: LegacyPointMazeRuntime, model_block: torch.Tensor, dataset: Any) -> None:
    raw_actions = block_to_physical(model_block, dataset)
    if raw_actions.shape != (MODEL_BLOCK_HIGH_LEVEL_ACTIONS, 2):
        raise RuntimeError(f"model block decoded to unexpected physical shape {raw_actions.shape}")
    for action in raw_actions:
        runtime.step_block(action)


def run_episode(
    *,
    root: int,
    method: str,
    state_root: Path,
    renderer: DatasetMatchedRenderer,
    calibration: Any,
    model: Any,
    dataset: Any,
    candidates: torch.Tensor,
    goal_observation: dict[str, torch.Tensor],
    max_blocks: int,
    success_radius: float,
    device: torch.device,
) -> dict[str, Any]:
    runtime = LegacyPointMazeRuntime(load_episode_states(state_root, root)[0])
    current_rgb = renderer.render(runtime.state)
    observation, center, proxy = _raw_rgb_proxy_observation(current_rgb, None, calibration, device)
    initial_distance = float(np.linalg.norm(runtime.state[:2] - GOAL_STATE))
    minimum_distance = initial_distance
    trace: list[dict[str, Any]] = []
    for block in range(max_blocks):
        context, goal = _encoded_context_and_goal(model, observation, goal_observation)
        selection = select_fixed_tree_action(model, context, goal, candidates)
        chosen = int(selection[f"{method}_index"])
        _execute_model_block(runtime, candidates[chosen], dataset)
        next_rgb = renderer.render(runtime.state)
        observation, center, proxy = _raw_rgb_proxy_observation(next_rgb, center, calibration, device)
        distance = float(np.linalg.norm(runtime.state[:2] - GOAL_STATE))
        minimum_distance = min(minimum_distance, distance)
        trace.append({
            "block": block,
            "chosen_index": chosen,
            "rgb_proxy": proxy.tolist(),
            "distance_for_evaluation": distance,
            **selection,
        })
    final_distance = float(np.linalg.norm(runtime.state[:2] - GOAL_STATE))
    return {
        "root": int(root), "method": method, "initial_distance": initial_distance,
        "final_distance": final_distance, "minimum_distance": minimum_distance,
        "success": bool(minimum_distance <= success_radius), "blocks_executed": len(trace),
        "total_candidate_rollouts": int(sum(item["candidate_rollouts"] for item in trace)),
        "total_planner_seconds": float(sum(item["planner_seconds"] for item in trace)),
        "trace": trace,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration-roots", required=True)
    parser.add_argument("--outcome-roots", required=True)
    parser.add_argument("--max-blocks", type=int, default=4)
    parser.add_argument("--success-radius", type=float, default=0.25)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_roots = _parse_roots(args.calibration_roots, "--calibration-roots", 4)
    outcome_roots = _parse_roots(args.outcome_roots, "--outcome-roots", 2)
    if set(calibration_roots) & set(outcome_roots):
        raise ValueError("calibration and outcome roots must be root-disjoint")
    if args.output_dir.exists() or args.max_blocks < 1 or args.success_radius <= 0:
        raise ValueError("output must be new; max-blocks and success-radius must be positive")
    device = require_cuda()
    args.output_dir.mkdir(parents=True)
    model, dataset = load_model_and_dataset(args, device)
    parameter_before = parameter_hash(model)
    calibration = fit_visual_calibration(args.dataset_root, calibration_roots, max_frames=1600)
    reference_images, _ = load_episode(args.dataset_root, calibration_roots[0])
    renderer = DatasetMatchedRenderer.from_episode(reference_images, calibration)
    goal_rgb = renderer.render(np.r_[GOAL_STATE, [0.0, 0.0]])
    goal_observation, _, _ = _raw_rgb_proxy_observation(goal_rgb, None, calibration, device)
    _, action_bank_tensor = action_bank(dataset, calibration_roots[0], 0, FIXED_ACTION_AMPLITUDE, device)
    candidates = action_bank_tensor[:5].detach()
    rows = [
        run_episode(
            root=root, method=method, state_root=args.dataset_root, renderer=renderer, calibration=calibration,
            model=model, dataset=dataset, candidates=candidates, goal_observation=goal_observation,
            max_blocks=args.max_blocks, success_radius=args.success_radius, device=device,
        )
        for root in outcome_roots for method in ("spectral", "immediate")
    ]
    parameter_after = parameter_hash(model)
    rgb_manifest = args.dataset_root / "rgb_calibration_manifest.json"
    state_manifest = (args.dataset_root / "states.pth").resolve().parent / "extension_manifest.json"
    if not rgb_manifest.is_file() or not state_manifest.is_file():
        raise FileNotFoundError("P4 dataset must retain both RGB and state-extension provenance manifests")
    paired = []
    for root in outcome_roots:
        values = {row["method"]: row for row in rows if row["root"] == root}
        paired.append({
            "root": root,
            "spectral_minus_immediate_final_distance": values["spectral"]["final_distance"] - values["immediate"]["final_distance"],
            "spectral_minus_immediate_minimum_distance": values["spectral"]["minimum_distance"] - values["immediate"]["minimum_distance"],
            "spectral_minus_immediate_planner_seconds": values["spectral"]["total_planner_seconds"] - values["immediate"]["total_planner_seconds"],
            "spectral_minus_immediate_candidate_rollouts": values["spectral"]["total_candidate_rollouts"] - values["immediate"]["total_candidate_rollouts"],
        })
    gates = {
        "cuda_only": True,
        "frozen_parameter_hash_unchanged": parameter_before == parameter_after,
        "root_disjoint_calibration_outcome": not bool(set(calibration_roots) & set(outcome_roots)),
        "matched_two_step_candidate_rollout_counts": all(item["spectral_minus_immediate_candidate_rollouts"] == 0 for item in paired),
        "rgb_only_policy_input": True,
    }
    report = {
        "schema": SCHEMA,
        "status": "admission_pass" if all(gates.values()) else "invalid",
        "scope": "RGB-only matched-budget SRA receding-horizon admission; small panel is not an efficacy conclusion.",
        "inputs": {
            "dataset_root": str(args.dataset_root.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
            "calibration_roots": calibration_roots, "outcome_roots": outcome_roots,
            "action_bank": "zero plus/minus x plus/minus y; five repeated high-level actions per model block",
            "max_blocks": args.max_blocks, "success_radius": args.success_radius,
            "runner_sha256": _sha256(Path(__file__)),
            "runtime_contract_sha256": _sha256(ROOT / "experiments/run_live_pointmaze_planning.py"),
            "rgb_calibration_manifest_sha256": _sha256(rgb_manifest),
            "state_extension_manifest_sha256": _sha256(state_manifest),
        },
        "proxy_calibration": {
            "position_rmse": calibration.position_rmse.tolist(), "velocity_rmse": calibration.velocity_rmse.tolist(),
            "roots": list(calibration.calibration_episodes),
        },
        "device": {"type": device.type, "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "gates": gates, "rows": rows, "paired": paired,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "paired": paired}, indent=2))
    if not all(gates.values()):
        raise RuntimeError("P4 admission gates failed")


if __name__ == "__main__":
    main()
