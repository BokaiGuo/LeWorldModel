"""Root-disjoint temporal action-effect gate audit for the frozen Cube model.

This experiment asks a narrower question than online planning: does a short
history of RGB/frozen-latent observations and *executed* actions predict the
next observed block effect better than the current frame and action alone?

The collection action is a predeclared approach/close schedule.  Simulator
state is used only to make evaluator labels after each transition.  The gate
is frozen from calibration roots only:

    base prediction + event(x history) * support(x history, action) * residual

When event or support evidence is insufficient, the method falls back to the
static predictor.  This is an offline mechanism audit, not online planning
evidence and not a causal-dynamics claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel  # noqa: F401 - registers environments
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_cube_rgb_observability import (  # noqa: E402
    _contact,
    _exploratory_action,
    _goal_position,
    _make_env,
    _vec,
    fit_ridge,
    image_transform,
    root_split,
)
from experiments.checkpoint_compat import load_compatible_pretrained  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--action-seed", type=int, default=20261017)
    parser.add_argument("--roots", type=int, default=80)
    parser.add_argument("--frames-per-root", type=int, default=24)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--event-quantile", type=float, default=0.75)
    parser.add_argument("--support-quantile", type=float, default=0.95)
    parser.add_argument(
        "--collection-mode", choices=("generic", "guided_intervention"),
        default="guided_intervention",
        help="guided_intervention is mechanism-only coverage, never runtime planning.",
    )
    return parser.parse_args()


def collect_temporal_frames(
    *, roots: int, frames_per_root: int, tasks: list[int], seed: int,
    action_seed: int, collection_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Collect consecutive current/previous frames and executed actions."""
    if roots < 8 or frames_per_root < 2:
        raise ValueError("roots must be >= 8 and frames-per-root must be >= 2")
    rng = np.random.default_rng(action_seed)
    current_images: list[np.ndarray] = []
    previous_images: list[np.ndarray] = []
    goal_images: list[np.ndarray] = []
    previous_actions: list[np.ndarray] = []
    labels: dict[str, list[Any]] = {
        "next_block_delta_xy": [],
        "next_block_goal_delta_xy": [],
        "next_effector_delta_xy": [],
        "physical_motion": [],
        "contact": [],
        "action": [],
        "previous_action": [],
        "history_valid": [],
        "root_id": [],
        "task_id": [],
        "frame_id": [],
    }
    env = _make_env()
    try:
        for root_id in range(roots):
            task_id = int(tasks[root_id % len(tasks)])
            obs, info = env.reset(
                seed=int(seed + root_id),
                options={"task_id": task_id, "render_goal": True},
            )
            goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
            guided_direction = None
            if collection_mode == "guided_intervention":
                guided_direction = (
                    _vec(info, "privileged/block_0_pos", 3)[:2]
                    - _vec(info, "proprio/effector_pos", 3)[:2]
                )
            previous_image = np.asarray(obs, dtype=np.uint8).copy()
            previous_action = np.zeros(5, dtype=np.float32)
            history_valid = 0.0
            for frame_id in range(frames_per_root):
                current = np.asarray(obs, dtype=np.uint8).copy()
                action = _exploratory_action(
                    root_id, frame_id, rng, direction=guided_direction,
                )
                block = _vec(info, "privileged/block_0_pos", 3)[:2]
                effector = _vec(info, "proprio/effector_pos", 3)[:2]
                current_images.append(current)
                previous_images.append(previous_image.copy())
                goal_images.append(goal_image.copy())
                previous_actions.append(previous_action.copy())
                labels["action"].append(action.copy())
                labels["previous_action"].append(previous_action.copy())
                labels["history_valid"].append(history_valid)
                labels["root_id"].append(root_id)
                labels["task_id"].append(task_id)
                labels["frame_id"].append(frame_id)
                labels["contact"].append(float(_contact(info) > 1.0e-5))

                next_obs, _, terminated, truncated, next_info = env.step(action)
                next_block = _vec(next_info, "privileged/block_0_pos", 3)[:2]
                next_effector = _vec(next_info, "proprio/effector_pos", 3)[:2]
                next_delta = next_block - block
                labels["next_block_delta_xy"].append(next_delta)
                labels["next_block_goal_delta_xy"].append(_goal_position(env)[:2] - next_block)
                labels["next_effector_delta_xy"].append(next_effector - effector)
                labels["physical_motion"].append(float(np.linalg.norm(next_delta) > 1.0e-5))
                previous_image = current
                previous_action = action
                history_valid = 1.0
                obs, info = next_obs, next_info
                if terminated or truncated:
                    break
    finally:
        env.close()

    arrays = {
        "next_block_delta_xy": np.asarray(labels["next_block_delta_xy"], dtype=np.float32),
        "next_block_goal_delta_xy": np.asarray(labels["next_block_goal_delta_xy"], dtype=np.float32),
        "next_effector_delta_xy": np.asarray(labels["next_effector_delta_xy"], dtype=np.float32),
        "physical_motion": np.asarray(labels["physical_motion"], dtype=np.float32),
        "contact": np.asarray(labels["contact"], dtype=np.float32),
        "action": np.asarray(labels["action"], dtype=np.float32),
        "previous_action": np.asarray(labels["previous_action"], dtype=np.float32),
        "history_valid": np.asarray(labels["history_valid"], dtype=np.float32),
        "root_id": np.asarray(labels["root_id"], dtype=np.int64),
        "task_id": np.asarray(labels["task_id"], dtype=np.int64),
        "frame_id": np.asarray(labels["frame_id"], dtype=np.int64),
    }
    if len(current_images) == 0 or len(current_images) != len(arrays["root_id"]):
        raise RuntimeError("temporal Cube collection produced misaligned frames and labels")
    return (
        np.asarray(current_images, dtype=np.uint8),
        np.asarray(previous_images, dtype=np.uint8),
        np.asarray(goal_images, dtype=np.uint8),
        np.asarray(previous_actions, dtype=np.float32),
        arrays,
    )


def _low_res(images: np.ndarray, device: str) -> torch.Tensor:
    values = torch.as_tensor(images, device=device).float().permute(0, 3, 1, 2) / 255.0
    return F.adaptive_avg_pool2d(
        F.interpolate(values, size=(32, 32), mode="bilinear", align_corners=False),
        (8, 8),
    ).flatten(1)


def _delta_action_interaction(delta: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Compress innovation before forming action-conditioned interactions.

    A full pixel/latent-by-action tensor is too high-dimensional for the
    available root count.  Fixed contiguous groups give a deterministic,
    low-variance summary: signed and absolute innovation per group, followed
    by its product with each executed action coordinate.
    """
    groups = torch.chunk(delta, chunks=12, dim=1)
    summary = torch.cat(
        [torch.cat([part.mean(dim=1, keepdim=True), part.abs().mean(dim=1, keepdim=True)], dim=1)
         for part in groups],
        dim=1,
    )
    interaction = (summary[:, :, None] * action[:, None, :]).flatten(1)
    return torch.cat([summary, interaction], dim=1)


def _compact_latent(latent: torch.Tensor, bins: int = 24) -> torch.Tensor:
    """Apply a fixed low-variance channel pooling to a frozen latent.

    The raw temporal family appends a full latent innovation to the current
    state and action features.  With a small number of root-disjoint contact
    branches this creates a high-dimensional, weakly regularized readout.  A
    fixed adaptive average pool is parameter-free, preserves signed trends,
    and makes the compact family a predeclared variance-control intervention
    rather than a test-set feature search.
    """
    if latent.ndim != 2 or bins < 1:
        raise ValueError("latent must be a matrix and bins must be positive")
    return F.adaptive_avg_pool1d(latent.unsqueeze(1), bins).squeeze(1)


@torch.inference_mode()
def _encode_images(
    model: torch.nn.Module, images: np.ndarray, device: str, batch_size: int,
) -> torch.Tensor:
    transform = image_transform()
    outputs: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([
            transform(torch.from_numpy(image).permute(2, 0, 1))
            for image in images[start:start + batch_size]
        ]).unsqueeze(1).to(device)
        outputs.append(model.encode({"pixels": batch})["emb"][:, 0])
    return torch.cat(outputs, dim=0)


def build_feature_families(
    current: np.ndarray, previous: np.ndarray, goals: np.ndarray,
    actions: np.ndarray, previous_actions: np.ndarray, history_valid: np.ndarray,
    *, device: str, model: torch.nn.Module | None = None, batch_size: int = 32,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Create static and temporal features plus a calibration-free event score."""
    current_rgb = _low_res(current, device)
    previous_rgb = _low_res(previous, device)
    goal_rgb = _low_res(goals, device)
    action = torch.as_tensor(actions, device=device).float()
    previous_action = torch.as_tensor(previous_actions, device=device).float()
    valid = torch.as_tensor(history_valid, device=device).float().reshape(-1, 1)
    rgb_delta = current_rgb - previous_rgb
    rgb_goal_delta = current_rgb - goal_rgb
    rgb_interaction = _delta_action_interaction(rgb_delta, action)
    rgb_static = torch.cat([current_rgb, goal_rgb, rgb_goal_delta, action], dim=1)
    rgb_temporal = torch.cat(
        [current_rgb, goal_rgb, rgb_goal_delta, rgb_delta,
         action, previous_action, valid], dim=1,
    )
    rgb_temporal_interaction = torch.cat(
        [current_rgb, goal_rgb, rgb_goal_delta, rgb_interaction,
         action, previous_action, valid], dim=1,
    )
    families: dict[str, torch.Tensor] = {
        "rgb_static": rgb_static,
        "rgb_temporal": rgb_temporal,
        "rgb_temporal_interaction": rgb_temporal_interaction,
    }
    # Mean absolute image innovation is an observable event score.  The first
    # frame is deliberately invalid, so it cannot activate the residual gate.
    event_score = rgb_delta.abs().mean(dim=1) * valid[:, 0]
    if model is not None:
        all_images = np.concatenate([current, previous, goals], axis=0)
        embeddings = _encode_images(model, all_images, device, batch_size)
        n = len(current)
        current_latent, previous_latent, goal_latent = embeddings.split(n, dim=0)
        latent_delta = current_latent - previous_latent
        latent_goal_delta = current_latent - goal_latent
        latent_interaction = _delta_action_interaction(latent_delta, action)
        latent_static = torch.cat([current_latent, goal_latent, latent_goal_delta, action], dim=1)
        latent_temporal = torch.cat(
            [current_latent, goal_latent, latent_goal_delta, latent_delta,
             action, previous_action, valid], dim=1,
        )
        latent_temporal_interaction = torch.cat(
            [current_latent, goal_latent, latent_goal_delta, latent_interaction,
             action, previous_action, valid], dim=1,
        )
        current_compact = _compact_latent(current_latent)
        goal_compact = _compact_latent(goal_latent)
        goal_delta_compact = _compact_latent(latent_goal_delta)
        delta_compact = _compact_latent(latent_delta)
        latent_static_compact = torch.cat(
            [current_compact, goal_compact, goal_delta_compact, action], dim=1,
        )
        latent_temporal_compact = torch.cat(
            [current_compact, goal_compact, goal_delta_compact, delta_compact,
             action, previous_action, valid], dim=1,
        )
        latent_residual_compact = torch.cat(
            [delta_compact, previous_action, valid], dim=1,
        )
        families["latent_static"] = latent_static
        families["latent_temporal"] = latent_temporal
        families["latent_temporal_interaction"] = latent_temporal_interaction
        families["latent_static_compact"] = latent_static_compact
        families["latent_temporal_compact"] = latent_temporal_compact
        families["latent_residual_compact"] = latent_residual_compact
        latent_event = latent_delta.abs().mean(dim=1) * valid[:, 0]
        # Keep the observable event score separate; latent event is recorded as
        # an auxiliary diagnostic rather than used to select the RGB gate.
        event_score = torch.stack([event_score, latent_event], dim=1).mean(dim=1)
    return families, event_score


def _regression_metrics(prediction: torch.Tensor, target: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    error = prediction.detach().float().cpu().numpy() - target
    held = error[test]
    return {
        "test_rmse": float(np.sqrt(np.mean(held ** 2))),
        "test_component_rmse": np.sqrt(np.mean(held ** 2, axis=0)).tolist(),
        "test_positive_effect_rmse": (
            float(np.sqrt(np.mean(held[np.linalg.norm(target[test], axis=1) > 1.0e-5] ** 2)))
            if np.any(np.linalg.norm(target[test], axis=1) > 1.0e-5) else None
        ),
    }


def _per_root_rmse(
    prediction: torch.Tensor, target: np.ndarray, root_ids: np.ndarray, test: np.ndarray,
) -> list[dict[str, Any]]:
    prediction_np = prediction.detach().float().cpu().numpy()
    rows = []
    for root in np.unique(root_ids[test]):
        mask = test & (root_ids == root)
        error = prediction_np[mask] - target[mask]
        rows.append({
            "root_id": int(root),
            "n_frames": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
        })
    return rows


def _binary_metrics(score: np.ndarray, target: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    actual = target[test] > 0.5
    predicted = score[test] >= 0.5
    tp = int(np.sum(predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    if not np.any(actual) or not np.any(~actual):
        balanced = None
    else:
        balanced = float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))
    return {
        "test_balanced_accuracy": balanced,
        "test_brier": float(np.mean((score[test] - target[test]) ** 2)),
        "test_positive_rate": float(np.mean(actual)) if len(actual) else None,
        "test_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _fit_family(
    features: torch.Tensor, target: torch.Tensor,
    calibration: np.ndarray, test: np.ndarray, ridge: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prediction, mean, scale, _ = fit_ridge(features, target, calibration, ridge)
    return prediction, {
        "feature_dim": int(features.shape[1]),
        "calibration_roots_only": True,
        "calibration_mean_norm": float(torch.linalg.vector_norm(mean).item()),
        "calibration_scale_min": float(scale.min().item()),
        "metrics": _regression_metrics(prediction, target.detach().cpu().numpy(), test),
    }


def _fit_gated_residual(
    base_features: torch.Tensor, temporal_features: torch.Tensor,
    event_score: torch.Tensor, target: torch.Tensor,
    calibration: np.ndarray, test: np.ndarray, ridge: float,
    event_quantile: float, support_quantile: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit a calibration-only residual and freeze event/support thresholds."""
    base_prediction, _, _, _ = fit_ridge(base_features, target, calibration, ridge)
    residual_target = target - base_prediction.detach()
    residual_prediction, temporal_mean, temporal_scale, _ = fit_ridge(
        temporal_features, residual_target, calibration, ridge,
    )
    cal = torch.as_tensor(calibration, dtype=torch.bool, device=event_score.device)
    event_threshold = float(torch.quantile(event_score[cal], event_quantile).item())
    standardized = (temporal_features - temporal_mean) / temporal_scale.clamp_min(1.0e-4)
    support_score = torch.sqrt(torch.mean(standardized.square(), dim=1))
    support_threshold = float(torch.quantile(support_score[cal], support_quantile).item())
    # Strict comparison is intentional: an all-zero calibration event score
    # must not activate the residual for every frame.  In that case the gate
    # is closed until an actually observed innovation exceeds the threshold.
    gate = (
        (event_score > event_threshold)
        & (support_score <= support_threshold)
        & torch.as_tensor(np.asarray(calibration | test), device=event_score.device)
    )
    gated_prediction = base_prediction + gate[:, None].to(base_prediction.dtype) * residual_prediction
    target_np = target.detach().cpu().numpy()
    test_gate = gate.detach().cpu().numpy() & test
    metrics = _regression_metrics(gated_prediction, target_np, test)
    if np.any(test_gate):
        accepted_error = gated_prediction.detach().cpu().numpy()[test_gate] - target_np[test_gate]
        accepted_rmse = float(np.sqrt(np.mean(accepted_error ** 2)))
    else:
        accepted_rmse = None
    metrics.update({
        "accepted_test_rmse": accepted_rmse,
        "test_coverage": float(np.mean(test_gate[test])) if np.any(test) else 0.0,
        "test_rejection_rate": float(1.0 - np.mean(test_gate[test])) if np.any(test) else 1.0,
    })
    return gated_prediction, {
        "event_quantile": event_quantile,
        "support_quantile": support_quantile,
        "event_threshold_calibration_only": event_threshold,
        "support_threshold_calibration_only": support_threshold,
        "support_score_test_mean": float(support_score[test].mean().item()),
        "gate_test_activation_rate": float(gate[test].float().mean().item()),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for temporal Cube audit but unavailable")
    if not 0.0 < args.event_quantile < 1.0 or not 0.0 < args.support_quantile < 1.0:
        raise ValueError("gate quantiles must lie in (0, 1)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "[purpose] Cube temporal action-effect gate audit; evaluator labels are offline only; "
        f"device={args.device} roots={args.roots} frames_per_root={args.frames_per_root}"
    )
    current, previous, goals, previous_actions, labels = collect_temporal_frames(
        roots=args.roots, frames_per_root=args.frames_per_root, tasks=args.tasks,
        seed=args.seed, action_seed=args.action_seed,
        collection_mode=args.collection_mode,
    )
    calibration, test = root_split(labels["root_id"], args.calibration_fraction, args.seed + 1)
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    families, event_score = build_feature_families(
        current, previous, goals, labels["action"], previous_actions,
        labels["history_valid"], device=args.device, model=model,
        batch_size=args.batch_size,
    )
    target = torch.as_tensor(labels["next_block_delta_xy"], device=args.device).float()
    report: dict[str, Any] = {
        "schema": "cube_temporal_action_effect_gate.v1",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "device": args.device,
            "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "root_disjoint_calibration_test": True,
            "collection_mode": args.collection_mode,
            "runtime_policy_state_access": False,
            "evaluator_state_role": "offline next-block displacement labels only",
            "history": "consecutive previous RGB/latent frame plus previous executed action",
            "primary_target": "next_block_delta_xy",
            "gate": "calibration-only event quantile AND feature-support quantile; fallback to static base",
            "event_score": "mean absolute current-previous RGB/latent innovation, first frame invalid",
            "event_quantile": args.event_quantile,
            "support_quantile": args.support_quantile,
            "ridge": args.ridge,
        },
        "data": {
            "frames": int(len(current)),
            "roots": int(len(np.unique(labels["root_id"]))),
            "history_valid_rate": float(labels["history_valid"].mean()),
            "physical_motion_rate": float(labels["physical_motion"].mean()),
            "contact_rate": float(labels["contact"].mean()),
            "task_counts": {str(task): int(np.sum(labels["task_id"] == task)) for task in np.unique(labels["task_id"])},
        },
        "metrics": {},
        "claim_boundary": (
            "Root-disjoint mechanism audit only. A temporal effect gain does not "
            "establish causal latent dynamics or online visual planning; the gate "
            "is eligible for a planner only after an independent paired consequence."
        ),
    }
    predictions: dict[str, torch.Tensor] = {}
    target_np = target.detach().cpu().numpy()
    for name in (
        "rgb_static", "rgb_temporal", "rgb_temporal_interaction",
        "latent_static", "latent_temporal", "latent_temporal_interaction",
    ):
        print(f"[fit] feature={name} dim={families[name].shape[-1]}")
        prediction, details = _fit_family(families[name], target, calibration, test, args.ridge)
        predictions[name] = prediction
        details["test_root_rmse"] = _per_root_rmse(
            prediction, target_np, labels["root_id"], test,
        )
        report["metrics"][name] = details

    gate_prediction, gate_details = _fit_gated_residual(
        families["rgb_static"], families["rgb_temporal_interaction"], event_score, target,
        calibration, test, args.ridge, args.event_quantile, args.support_quantile,
    )
    report["metrics"]["rgb_static_plus_event_support_interaction_gate"] = gate_details
    gate_details["test_root_rmse"] = _per_root_rmse(
        gate_prediction, target_np, labels["root_id"], test,
    )
    report["event_score"] = {
        "calibration_mean": float(event_score[calibration].mean().item()),
        "test_mean": float(event_score[test].mean().item()),
        "test_quantiles": {
            str(q): float(torch.quantile(event_score[test], q).item())
            for q in (0.5, 0.75, 0.9, 0.95)
        },
    }
    report["split"] = {
        "calibration_roots": np.unique(labels["root_id"][calibration]).astype(int).tolist(),
        "test_roots": np.unique(labels["root_id"][test]).astype(int).tolist(),
        "calibration_frames": int(calibration.sum()),
        "test_frames": int(test.sum()),
    }
    (args.output / "temporal_action_effect.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output / "temporal_action_effect.json"), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
