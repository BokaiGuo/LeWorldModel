"""Mode-stratified counterfactual action-effect audit for a frozen visual model.

For every rendered source state this evaluator executes a fixed fan of actions
in independent simulator resets.  The model sees only the source RGB frame(s),
the goal RGB frame, and a proposed action; physical state is retained solely
for offline labels and stratification.  Roots, action fans, and calibration
roots are seeded before checkpoint loading.

This produces causal *intervention* evidence for the declared simulator and
checkpoint interface.  It does not turn a local predictor into a universal
physical world model or establish online-planning success.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401: registers PushT
import torch

try:
    from .audit_causal_visual_checkpoint import load_checkpoint
    from .run_causal_visual_pusht_world_model import (
        balanced_source_cases,
        observable_pose_vector,
        reset_with_case,
        resize_image,
        tensor_images,
    )
except ImportError:  # direct script execution
    from audit_causal_visual_checkpoint import load_checkpoint  # type: ignore[no-redef]
    from run_causal_visual_pusht_world_model import (  # type: ignore[no-redef]
        balanced_source_cases,
        observable_pose_vector,
        reset_with_case,
        resize_image,
        tensor_images,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=81001)
    parser.add_argument("--roots", type=int, default=400)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--root-rollout-steps", type=int, default=8)
    parser.add_argument("--fan-size", type=int, default=8, help="non-zero action directions per visual source")
    parser.add_argument("--action-magnitude", type=float, default=0.85)
    parser.add_argument(
        "--action-support", choices=("isotropic", "heldout_diagonal"), default="isotropic",
        help="heldout_diagonal evaluates only diagonal directions and requires a cardinal-trained checkpoint",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def mode_label(source: np.ndarray, successor: np.ndarray) -> str:
    """Offline physical stratum; never consumed by the predictor/controller."""
    start_distance = float(np.linalg.norm(source[:2] - source[2:4]))
    end_distance = float(np.linalg.norm(successor[:2] - successor[2:4]))
    block_motion = float(np.linalg.norm(successor[2:4] - source[2:4]))
    if start_distance > 65.0:
        return "no_contact"
    if end_distance > 80.0:
        return "separation"
    if block_motion > 1.0:
        return "sliding"
    return "contact"


def action_fan(fan_size: int, magnitude: float, rng: np.random.Generator, support: str = "isotropic") -> np.ndarray:
    if fan_size < 4:
        raise ValueError("fan_size must be at least four to identify action ranking")
    if support == "isotropic":
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
    elif support == "heldout_diagonal":
        if fan_size != 4:
            raise ValueError("heldout_diagonal uses exactly four diagonal directions; set fan_size=4")
        phase = math.pi / 4.0
    else:
        raise ValueError(f"unknown action support: {support}")
    angles = phase + np.linspace(0.0, 2.0 * math.pi, fan_size, endpoint=False)
    directional = np.stack([np.cos(angles), np.sin(angles)], axis=1) * float(magnitude)
    return np.concatenate([directional, np.zeros((1, 2), dtype=np.float64)], axis=0).astype(np.float32)


@torch.inference_mode()
def predict_successor(
    model: torch.nn.Module, normalizer: Any, image: np.ndarray, previous_image: np.ndarray,
    action: np.ndarray, horizon: int, image_size: int, device: str,
) -> tuple[np.ndarray, np.ndarray]:
    mean, std = normalizer.tensors(device)
    current = tensor_images(resize_image(image, image_size)[None], device)
    previous = tensor_images(resize_image(previous_image, image_size)[None], device)
    state = model.encode_state(current, previous)
    decoded = (state * std + mean).cpu().numpy()[0]
    action_tensor = torch.as_tensor(action[None], dtype=torch.float32, device=device)
    for _ in range(horizon):
        state = model.predict_next(state, action_tensor)
    return decoded, (state * std + mean).cpu().numpy()[0]


def pose_cost(pose: np.ndarray, goal: np.ndarray) -> float:
    target = observable_pose_vector(goal)
    orientation = np.sum(
        (pose[4:6] / max(float(np.linalg.norm(pose[4:6])), 1e-6)
         - target[4:6] / max(float(np.linalg.norm(target[4:6])), 1e-6)) ** 2
    )
    return float(np.sum((pose[2:4] - target[2:4]) ** 2) + 2400.0 * orientation)


def _quantile(values: list[float], coverage: float = 0.9) -> float | None:
    if not values:
        return None
    # Finite-sample split-conformal order statistic.
    level = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(np.asarray(values, dtype=np.float64), level, method="higher"))


def summarize(rows: list[dict[str, Any]], calibration_fraction: float) -> dict[str, Any]:
    by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(int(row["horizon"]), str(row["mode"]))].append(row)
    report: dict[str, Any] = {}
    for (horizon, mode), values in sorted(by_key.items()):
        calibration = [item for item in values if item["split"] == "calibration"]
        heldout = [item for item in values if item["split"] == "heldout"]
        quantile = _quantile([item["effect_error"] for item in calibration])
        root_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in heldout:
            root_groups[int(item["root_id"])].append(item)
        ranking = []
        for group in root_groups.values():
            if len(group) < 2:
                continue
            truth = np.asarray([item["true_goal_cost"] for item in group])
            prediction = np.asarray([item["predicted_goal_cost"] for item in group])
            ranking.append(float(np.argmin(truth) == np.argmin(prediction)))
        effect = np.asarray([item["effect_error"] for item in heldout], dtype=np.float64)
        endpoint = np.asarray([item["endpoint_error"] for item in heldout], dtype=np.float64)
        coverage = None if quantile is None or not len(effect) else float(np.mean(effect <= quantile))
        report[f"h{horizon}/{mode}"] = {
            "horizon": horizon,
            "mode": mode,
            "calibration_rows": len(calibration),
            "heldout_rows": len(heldout),
            "heldout_effect_rmse": float(np.sqrt(np.mean(effect ** 2))) if len(effect) else None,
            "heldout_endpoint_rmse": float(np.sqrt(np.mean(endpoint ** 2))) if len(endpoint) else None,
            "counterfactual_goal_rank_top1": float(np.mean(ranking)) if ranking else None,
            "ranking_roots": len(ranking),
            "conformal_effect_radius_90": quantile,
            "conformal_effect_coverage_90": coverage,
        }
    return report


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    if any(horizon < 1 for horizon in args.horizons):
        raise ValueError("horizons must be positive")
    model, normalizer, saved = load_checkpoint(args.checkpoint, args.device)
    if args.action_support == "heldout_diagonal":
        if getattr(saved, "train_action_support", None) != "cardinal":
            raise ValueError("heldout_diagonal evaluation requires a checkpoint trained with --train-action-support cardinal")
        if args.action_magnitude <= float(getattr(saved, "train_action_max_magnitude", 1.0)):
            raise ValueError("held-out magnitude must exceed the checkpoint training action maximum")
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=max(args.horizons) * args.action_repeat)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    try:
        roots = balanced_source_cases(
            env, args.roots, args.root_rollout_steps, args.seed, float(saved.contact_root_fraction),
            int(saved.contact_approach_steps),
        )
        calibration_roots = int(round(args.roots * args.calibration_fraction))
        for root_id, (source, goal, image, previous_image, _goal_image, _contact) in enumerate(roots):
            split = "calibration" if root_id < calibration_roots else "heldout"
            fan = action_fan(args.fan_size, args.action_magnitude, rng, args.action_support)
            for action_index, action in enumerate(fan):
                for horizon in args.horizons:
                    observation, _ = reset_with_case(env, source, goal)
                    for _ in range(horizon * args.action_repeat):
                        observation, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            break
                    successor = np.asarray(observation["state"], dtype=np.float32)
                    decoded, predicted = predict_successor(
                        model, normalizer, image, previous_image, action, horizon, int(saved.image_size), args.device,
                    )
                    true_pose = observable_pose_vector(successor)
                    source_pose = observable_pose_vector(source)
                    # Effect error removes source-image state-readout error;
                    # endpoint error reports the full perception-plus-dynamics task.
                    effect_error = float(np.linalg.norm((predicted - decoded) - (true_pose - source_pose)))
                    rows.append({
                        "root_id": root_id,
                        "split": split,
                        "action_index": action_index,
                        "action": action.tolist(),
                        "horizon": horizon,
                        "mode": mode_label(source, successor),
                        "effect_error": effect_error,
                        "endpoint_error": float(np.linalg.norm(predicted - true_pose)),
                        "true_goal_cost": pose_cost(true_pose, goal),
                        "predicted_goal_cost": pose_cost(predicted, goal),
                    })
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "same_visual_source_multiple_interventions": True,
            "action_fan": int(args.fan_size) + 1,
            "action_support": args.action_support,
            "horizons": [int(value) for value in args.horizons],
            "root_calibration_heldout_disjoint": True,
            "runtime_input": "previous/current rendered image and proposed action only" if saved.temporal_context else "current rendered image and proposed action only",
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "physical_state_role": "offline intervention outcome and mode label only",
            "uncertainty": "split-conformal effect radius calibrated on disjoint intervention roots; this is calibrated coverage, not an intrinsic probabilistic model",
        },
        "summary": summarize(rows, args.calibration_fraction),
        "rows": rows,
        "claim_boundary": "Counterfactual intervention evidence for one frozen visual model in PushT. A positive result supports mode-scoped action-conditioned prediction, not universal physical causality, stable planning, or cross-environment generality.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "causal_intervention_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "causal_intervention_audit.md").write_text(
        "# Mode-stratified visual intervention audit\n\n"
        "Each source RGB state is paired with independently executed action interventions. "
        "Rows are split by root before conformal calibration/evaluation.\n\n"
        + "\n".join(
            f"- {name}: n={item['heldout_rows']}, effect RMSE={item['heldout_effect_rmse']}, "
            f"rank@1={item['counterfactual_goal_rank_top1']}, coverage={item['conformal_effect_coverage_90']}"
            for name, item in report["summary"].items()
        ) + "\n"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
