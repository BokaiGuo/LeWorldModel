"""Audit whether Cube RGB and the frozen LeWM latent preserve task geometry.

This is an offline observability diagnostic, not a planner.  The policy never
receives the simulator labels below.  Current and goal RGB frames plus a
predeclared candidate action are reduced to fixed features, and
calibration-only ridge readouts are evaluated on root-disjoint transitions.
The same readouts are then fitted to frozen LeWM
embeddings.  The comparison distinguishes a visual-information bottleneck
from an encoder/latent bottleneck before another controller intervention is
attempted.

The reported labels are evaluator-only: block XY, effector XY, relative
block-to-goal and effector-to-block vectors, gripper contact, and the next
block displacement after the predeclared action.  They are never used to
construct an input or a learned action.  The result is deliberately limited
to observability and one-step effect evidence; it does not establish causal
dynamics or planning performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from scipy import ndimage
import stable_pretraining as spt
import stable_worldmodel  # noqa: F401 - registers environments
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--roots", type=int, default=80)
    parser.add_argument("--frames-per-root", type=int, default=24)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--action-seed", type=int, default=20261016)
    parser.add_argument(
        "--collection-mode", choices=("generic", "guided_intervention"), default="generic",
        help="generic is label-free support probing; guided_intervention is offline mechanism-only coverage.",
    )
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _make_env() -> gym.Env:
    return gym.make(
        "swm/OGBCube-v0",
        ob_type="pixels",
        render_mode="rgb_array",
        terminate_at_goal=False,
    )


def _vec(info: dict[str, Any], key: str, width: int) -> np.ndarray:
    value = np.asarray(info[key], dtype=np.float32).reshape(-1)
    if value.size != width:
        raise ValueError(f"{key} expected {width} values, got {value.shape}")
    return value


def _goal_position(env: gym.Env) -> np.ndarray:
    raw = env.unwrapped
    target_block = int(getattr(raw, "_target_block", 0))
    return np.asarray(raw.cur_task_info["goal_xyzs"][target_block], dtype=np.float32)


def _contact(info: dict[str, Any]) -> float:
    return float(np.max(np.asarray(info["proprio/gripper_contact"], dtype=np.float32)))


def _largest_red_centroid(image: np.ndarray) -> np.ndarray:
    """Return normalized centroid and validity of the largest red object."""
    image = np.asarray(image, dtype=np.uint8)
    red = image[..., 0].astype(np.float32)
    green = image[..., 1].astype(np.float32)
    blue = image[..., 2].astype(np.float32)
    mask = (red > 80.0) & (red > 1.25 * green) & (red > 1.15 * blue)
    components, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(3, dtype=np.float32)
    sizes = np.bincount(components.reshape(-1))[1:]
    component_id = int(np.argmax(sizes)) + 1
    y, x = np.where(components == component_id)
    if len(x) < 3:
        return np.zeros(3, dtype=np.float32)
    height, width = image.shape[:2]
    return np.asarray(
        [2.0 * float(x.mean()) / max(width - 1, 1) - 1.0,
         2.0 * float(y.mean()) / max(height - 1, 1) - 1.0,
         1.0],
        dtype=np.float32,
    )


def _target_red_centroid(
    image: np.ndarray, *, reference_y: float | None = None,
) -> np.ndarray:
    """Return the RGB centroid of the physical Cube, not its goal marker.

    ``OGBCube-v0`` renders the physical red cube together with a smaller,
    semi-transparent red target marker.  The previous goal-y matching rule
    selected the marker in the current frame, so the visual servo aligned the
    effector with a non-physical overlay and never established contact on
    several tasks.  The physical object is the largest sufficiently visible
    red component in this renderer.  ``reference_y`` remains in the API for
    compatibility with older callers, but is deliberately not used to choose
    a component: it is exactly the signal that caused the correspondence
    failure.  This is a fixed RGB-only rule and does not read simulator state.
    """
    del reference_y
    image = np.asarray(image, dtype=np.uint8)
    red = image[..., 0].astype(np.float32)
    green = image[..., 1].astype(np.float32)
    blue = image[..., 2].astype(np.float32)
    mask = (red > 80.0) & (red > 1.25 * green) & (red > 1.15 * blue)
    components, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(3, dtype=np.float32)
    candidates: list[tuple[int, float, np.ndarray]] = []
    height, width = image.shape[:2]
    for component_id in range(1, count + 1):
        y, x = np.where(components == component_id)
        if len(x) < 3:
            continue
        candidates.append((int(len(x)), float((red - green)[components == component_id].mean()),
                          np.asarray([x.mean(), y.mean()])))
    if not candidates:
        return np.zeros(3, dtype=np.float32)
    _, _, centroid = max(candidates, key=lambda item: (item[0], item[1]))
    return np.asarray(
        [2.0 * float(centroid[0]) / max(width - 1, 1) - 1.0,
         2.0 * float(centroid[1]) / max(height - 1, 1) - 1.0,
         1.0],
        dtype=np.float32,
    )


def _largest_effector_centroid(image: np.ndarray) -> np.ndarray:
    """Return a normalized RGB chart for the visible robot working end.

    The previous implementation selected the lowest bright, low-saturation
    component.  In the OGB-Cube renderer that component is the vertical table
    seam/shadow, so it is nearly invariant to one action direction and makes
    the calibration action map effectively rank one.  The robot is rendered
    in a purple/magenta material.  We therefore aggregate all sufficiently
    large purple components in the declared working-region band instead of
    selecting a brightness artifact.  This remains RGB-only at runtime; the
    row band and color rule are frozen before calibration outcomes are used.
    """
    image = np.asarray(image, dtype=np.uint8)
    value = image.astype(np.float32)
    red, green, blue = value[..., 0], value[..., 1], value[..., 2]
    # Purple robot material, with a permissive threshold for shading.  The
    # background is blue/gray and the task blocks are red, so both are rejected
    # by the two channel-ratio tests.
    mask = (
        (red > 45.0)
        & (blue > 45.0)
        & (red > 1.15 * green)
        & (blue > 1.15 * green)
        & (blue > 0.90 * red)
    )
    components, count = ndimage.label(mask)
    height, width = image.shape[:2]
    lower = 0.13 * max(height - 1, 1)
    # The fixed descent prelude moves the working end into the lower half of
    # the camera view.  Keep this upper bound broad enough for that operating
    # height while still excluding the red blocks and table background by the
    # color rule above.
    upper = 0.75 * max(height - 1, 1)
    weighted_x = 0.0
    weighted_y = 0.0
    total = 0
    for component_id in range(1, count + 1):
        y, x = np.where(components == component_id)
        if len(x) < 5:
            continue
        component_y = float(y.mean())
        if not lower <= component_y <= upper:
            continue
        total += len(x)
        weighted_x += float(len(x)) * float(x.mean())
        weighted_y += float(len(x)) * float(y.mean())
    if total < 20:
        return np.zeros(3, dtype=np.float32)
    centroid = np.asarray([weighted_x / total, weighted_y / total])
    return np.asarray(
        [2.0 * float(centroid[0]) / max(width - 1, 1) - 1.0,
         2.0 * float(centroid[1]) / max(height - 1, 1) - 1.0,
         1.0],
        dtype=np.float32,
    )


def extract_color_geometry_features(
    images: np.ndarray, goals: np.ndarray, actions: np.ndarray,
    device: str,
) -> dict[str, torch.Tensor]:
    """Extract a fixed RGB object/goal chart without simulator labels."""
    current = np.asarray([_largest_red_centroid(image) for image in images], dtype=np.float32)
    goal = np.asarray([_largest_red_centroid(image) for image in goals], dtype=np.float32)
    current_tensor = torch.as_tensor(current, device=device)
    goal_tensor = torch.as_tensor(goal, device=device)
    pair = torch.cat([current_tensor, goal_tensor, current_tensor - goal_tensor], dim=1)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    return {
        "color_geometry": pair,
        "color_geometry_action": torch.cat([pair, action_tensor], dim=1),
    }


def extract_color_geometry_temporal_features(
    images: np.ndarray, previous_images: np.ndarray, goals: np.ndarray,
    actions: np.ndarray, previous_actions: np.ndarray, device: str,
) -> dict[str, torch.Tensor]:
    """Extract a compact RGB geometry state with one-step visual innovation.

    This feature family is fixed before outcome evaluation.  It augments the
    current/goal centroid chart with the previous centroid, its innovation,
    and the previous executed action, so a delayed contact effect is not
    forced into a static current-state/action readout.  It uses RGB only;
    simulator geometry is never read here.
    """
    current = torch.as_tensor(
        np.asarray([_largest_red_centroid(image) for image in images], dtype=np.float32),
        device=device,
    )
    previous = torch.as_tensor(
        np.asarray([_largest_red_centroid(image) for image in previous_images], dtype=np.float32),
        device=device,
    )
    goal = torch.as_tensor(
        np.asarray([_largest_red_centroid(image) for image in goals], dtype=np.float32),
        device=device,
    )
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    previous_action_tensor = torch.as_tensor(
        previous_actions, dtype=torch.float32, device=device,
    )
    return {
        "color_geometry_temporal_action": torch.cat(
            [current, goal, current - goal, current - previous,
             action_tensor, previous_action_tensor], dim=1,
        ),
    }


def _exploratory_action(
    root_id: int, frame_id: int, rng: np.random.Generator,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """Return a frozen, label-free approach schedule for the next frame."""
    angles = np.arange(16, dtype=np.float32) * (2.0 * np.pi / 16.0)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    if direction is None:
        direction = directions[int(root_id) % len(directions)]
    else:
        direction = np.asarray(direction, dtype=np.float32).reshape(2)
        direction = direction / max(float(np.linalg.norm(direction)), 1.0e-6)
    if frame_id < 5:
        action = np.zeros(5, dtype=np.float32)
        action[:2] = direction
        return action
    if frame_id < 20:
        action = np.zeros(5, dtype=np.float32)
        action[2] = -1.0
        return action
    if frame_id < 24:
        action = np.zeros(5, dtype=np.float32)
        action[4] = -1.0
        return action
    # Beyond the declared approach schedule, use only a small fixed random
    # perturbation; this branch is not used by the default 24-frame panel.
    return rng.uniform(-0.2, 0.2, size=5).astype(np.float32)


def collect_frames(
    *, roots: int, frames_per_root: int, tasks: list[int], seed: int, action_seed: int,
    collection_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if roots < 8 or frames_per_root < 1:
        raise ValueError("roots must be >= 8 and frames-per-root must be positive")
    rng = np.random.default_rng(action_seed)
    current_images: list[np.ndarray] = []
    goal_images: list[np.ndarray] = []
    labels: dict[str, list[Any]] = {
        "block_xy": [],
        "effector_xy": [],
        "block_goal_delta_xy": [],
        "effector_block_delta_xy": [],
        "next_block_delta_xy": [],
        "next_effector_delta_xy": [],
        "physical_motion": [],
        "action": [],
        "contact": [],
        "root_id": [],
        "task_id": [],
    }
    env = _make_env()
    try:
        for root_id in range(roots):
            task_id = int(tasks[root_id % len(tasks)])
            reset_seed = int(seed + root_id)
            obs, info = env.reset(
                seed=reset_seed,
                options={"task_id": task_id, "render_goal": True},
            )
            goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
            goal = _goal_position(env)[:2]
            guided_direction = None
            if collection_mode == "guided_intervention":
                initial_block = _vec(info, "privileged/block_0_pos", 3)[:2]
                initial_effector = _vec(info, "proprio/effector_pos", 3)[:2]
                guided_direction = initial_block - initial_effector
            for frame_id in range(frames_per_root):
                current = np.asarray(obs, dtype=np.uint8).copy()
                block = _vec(info, "privileged/block_0_pos", 3)[:2]
                effector = _vec(info, "proprio/effector_pos", 3)[:2]
                action = _exploratory_action(
                    root_id, frame_id, rng, direction=guided_direction
                )
                current_images.append(current)
                goal_images.append(goal_image.copy())
                labels["block_xy"].append(block)
                labels["effector_xy"].append(effector)
                labels["block_goal_delta_xy"].append(goal - block)
                labels["effector_block_delta_xy"].append(block - effector)
                labels["contact"].append(float(_contact(info) > 1.0e-5))
                labels["action"].append(action)
                labels["root_id"].append(root_id)
                labels["task_id"].append(task_id)
                # The action is selected only from the predeclared schedule.
                # The post-step state is evaluator output for this transition.
                next_obs, _, terminated, truncated, next_info = env.step(action)
                next_block = _vec(next_info, "privileged/block_0_pos", 3)[:2]
                next_effector = _vec(next_info, "proprio/effector_pos", 3)[:2]
                next_delta = next_block - block
                labels["next_block_delta_xy"].append(next_delta)
                labels["next_effector_delta_xy"].append(next_effector - effector)
                labels["physical_motion"].append(float(np.linalg.norm(next_delta) > 1.0e-5))
                obs, info = next_obs, next_info
                if terminated or truncated:
                    # Do not reset into a new trajectory under the same root.
                    break
    finally:
        env.close()
    image_arrays = np.asarray(current_images, dtype=np.uint8)
    goal_arrays = np.asarray(goal_images, dtype=np.uint8)
    label_arrays = {
        "block_xy": np.asarray(labels["block_xy"], dtype=np.float32),
        "effector_xy": np.asarray(labels["effector_xy"], dtype=np.float32),
        "block_goal_delta_xy": np.asarray(labels["block_goal_delta_xy"], dtype=np.float32),
        "effector_block_delta_xy": np.asarray(labels["effector_block_delta_xy"], dtype=np.float32),
        "next_block_delta_xy": np.asarray(labels["next_block_delta_xy"], dtype=np.float32),
        "next_effector_delta_xy": np.asarray(labels["next_effector_delta_xy"], dtype=np.float32),
        "physical_motion": np.asarray(labels["physical_motion"], dtype=np.float32),
        "action": np.asarray(labels["action"], dtype=np.float32),
        "contact": np.asarray(labels["contact"], dtype=np.float32),
        "root_id": np.asarray(labels["root_id"], dtype=np.int64),
        "task_id": np.asarray(labels["task_id"], dtype=np.int64),
    }
    if len(image_arrays) == 0:
        raise RuntimeError("Cube collection produced no frames")
    return image_arrays, goal_arrays, label_arrays


def image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


@torch.inference_mode()
def extract_image_features(
    images: np.ndarray, goals: np.ndarray, actions: np.ndarray,
    device: str, batch_size: int,
) -> dict[str, torch.Tensor]:
    """Return fixed low-resolution RGB features on the requested device."""
    features: dict[str, list[torch.Tensor]] = {
        "current": [], "pair": [], "current_action": [], "pair_action": [],
    }
    for start in range(0, len(images), batch_size):
        current = torch.as_tensor(images[start:start + batch_size], device=device).float()
        goal = torch.as_tensor(goals[start:start + batch_size], device=device).float()
        current = current.permute(0, 3, 1, 2) / 255.0
        goal = goal.permute(0, 3, 1, 2) / 255.0
        current = F.adaptive_avg_pool2d(F.interpolate(current, size=(32, 32), mode="bilinear", align_corners=False), (8, 8))
        goal = F.adaptive_avg_pool2d(F.interpolate(goal, size=(32, 32), mode="bilinear", align_corners=False), (8, 8))
        action = torch.as_tensor(actions[start:start + batch_size], device=device).float()
        current_flat = current.flatten(1)
        pair_flat = torch.cat([current, goal, (current - goal).abs()], dim=1).flatten(1)
        features["current"].append(current_flat)
        features["pair"].append(pair_flat)
        features["current_action"].append(torch.cat([current_flat, action], dim=1))
        features["pair_action"].append(torch.cat([pair_flat, action], dim=1))
    return {name: torch.cat(parts, dim=0) for name, parts in features.items()}


@torch.inference_mode()
def extract_latent_features(
    model: torch.nn.Module, images: np.ndarray, goals: np.ndarray,
    actions: np.ndarray, device: str, batch_size: int,
) -> dict[str, torch.Tensor]:
    transform = image_transform()
    features: dict[str, list[torch.Tensor]] = {
        "current": [], "pair": [], "current_action": [], "pair_action": [],
    }
    for start in range(0, len(images), batch_size):
        current = torch.stack([
            transform(torch.from_numpy(image).permute(2, 0, 1))
            for image in images[start:start + batch_size]
        ]).unsqueeze(1).to(device)
        goal = torch.stack([
            transform(torch.from_numpy(image).permute(2, 0, 1))
            for image in goals[start:start + batch_size]
        ]).unsqueeze(1).to(device)
        current_emb = model.encode({"pixels": current})["emb"][:, 0]
        goal_emb = model.encode({"pixels": goal})["emb"][:, 0]
        action = torch.as_tensor(actions[start:start + batch_size], device=device).float()
        pair_emb = torch.cat([current_emb, goal_emb, current_emb - goal_emb], dim=-1)
        features["current"].append(current_emb)
        features["pair"].append(pair_emb)
        features["current_action"].append(torch.cat([current_emb, action], dim=-1))
        features["pair_action"].append(torch.cat([pair_emb, action], dim=-1))
    return {name: torch.cat(parts, dim=0) for name, parts in features.items()}


def root_split(root_ids: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(root_ids)
    if len(unique) < 8 or not 0.1 <= fraction <= 0.9:
        raise ValueError("need at least eight roots and calibration fraction in [0.1, 0.9]")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_cal = int(round(len(unique) * fraction))
    n_cal = min(max(n_cal, 1), len(unique) - 1)
    calibration_roots = set(shuffled[:n_cal].tolist())
    calibration = np.asarray([root in calibration_roots for root in root_ids], dtype=bool)
    return calibration, ~calibration


def _standardize_fit(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x[mask].mean(dim=0)
    scale = x[mask].std(dim=0, unbiased=False).clamp_min(1.0e-4)
    return (x - mean) / scale, mean, scale


@torch.inference_mode()
def fit_ridge(
    x: torch.Tensor, y: torch.Tensor, calibration: np.ndarray, ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Float64 is intentional: high-dimensional low-resolution RGB features
    # are strongly collinear (especially for a static goal image), and the
    # Gram solve in float32 can produce numerically explosive readouts.
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    cal = torch.as_tensor(calibration, dtype=torch.bool, device=x.device)
    z, mean, scale = _standardize_fit(x, cal)
    x_cal = z[cal]
    y_cal = y[cal]
    gram = x_cal.T @ x_cal
    regularizer = ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=x.device)
    weight = torch.linalg.solve(gram + regularizer, x_cal.T @ y_cal)
    prediction = z @ weight
    return prediction, mean, scale, weight


def _vector_metrics(prediction: np.ndarray, target: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction - target, dtype=np.float64)
    frame_rmse = float(np.sqrt(np.mean(error[test] ** 2)))
    vector_rmse = np.sqrt(np.mean(error[test] ** 2, axis=0)).tolist()
    return {"test_frame_rmse": frame_rmse, "test_component_rmse": vector_rmse}


def _contact_metrics(prediction: np.ndarray, target: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    score = np.clip(np.asarray(prediction, dtype=np.float64).reshape(-1), 0.0, 1.0)
    actual = np.asarray(target, dtype=np.float64).reshape(-1)
    predicted = score >= 0.5
    test_actual = actual[test] > 0.5
    test_predicted = predicted[test]
    tp = int(np.sum(test_predicted & test_actual))
    tn = int(np.sum(~test_predicted & ~test_actual))
    fp = int(np.sum(test_predicted & ~test_actual))
    fn = int(np.sum(~test_predicted & test_actual))
    has_both_classes = bool(np.any(test_actual) and np.any(~test_actual))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "test_balanced_accuracy": float(0.5 * (tpr + tnr)) if has_both_classes else None,
        "test_class_coverage": has_both_classes,
        "test_brier": float(np.mean((score[test] - actual[test]) ** 2)),
        "test_positive_rate": float(np.mean(test_actual)) if len(test_actual) else float("nan"),
        "test_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate_feature_family(
    features: torch.Tensor, labels: dict[str, np.ndarray],
    calibration: np.ndarray, test: np.ndarray, ridge: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in (
        "block_xy", "effector_xy", "block_goal_delta_xy", "effector_block_delta_xy",
        "next_block_delta_xy", "next_effector_delta_xy",
    ):
        target = torch.as_tensor(labels[name], dtype=torch.float32, device=features.device)
        prediction, _, _, _ = fit_ridge(features, target, calibration, ridge)
        results[name] = _vector_metrics(
            prediction.detach().cpu().numpy(), labels[name], test
        )
    target = torch.as_tensor(labels["contact"][:, None], dtype=torch.float32, device=features.device)
    prediction, _, _, _ = fit_ridge(features, target, calibration, ridge)
    results["contact"] = _contact_metrics(
        prediction.detach().cpu().numpy(), labels["contact"], test
    )
    target = torch.as_tensor(labels["physical_motion"][:, None], dtype=torch.float32, device=features.device)
    prediction, _, _, _ = fit_ridge(features, target, calibration, ridge)
    results["physical_motion"] = _contact_metrics(
        prediction.detach().cpu().numpy(), labels["physical_motion"], test
    )
    return results


def _root_aggregate(values: np.ndarray, root_ids: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    per_root = []
    for root in np.unique(root_ids[test]):
        per_root.append(float(np.mean(values[test & (root_ids == root)])))
    if not per_root:
        return {"n_roots": 0, "mean": float("nan")}
    return {"n_roots": len(per_root), "mean": float(np.mean(per_root))}


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this audit refuses CPU fallback")
    if args.batch_size < 1 or args.ridge <= 0.0:
        raise ValueError("batch-size must be positive and ridge must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "[purpose] Cube RGB/latent observability audit; labels are evaluator-only; "
        f"device={args.device} roots={args.roots} frames_per_root={args.frames_per_root}"
    )
    images, goals, labels = collect_frames(
        roots=args.roots, frames_per_root=args.frames_per_root, tasks=args.tasks,
        seed=args.seed, action_seed=args.action_seed,
        collection_mode=args.collection_mode,
    )
    calibration, test = root_split(labels["root_id"], args.calibration_fraction, args.seed + 1)
    print(
        f"[data] frames={len(images)} roots={len(np.unique(labels['root_id']))} "
        f"calibration_frames={int(calibration.sum())} test_frames={int(test.sum())} "
        f"contact_rate={float(labels['contact'].mean()):.4f}"
    )
    image_features = extract_image_features(
        images, goals, labels["action"], args.device, args.batch_size
    )
    image_features.update(
        extract_color_geometry_features(images, goals, labels["action"], args.device)
    )
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    latent_features = extract_latent_features(
        model, images, goals, labels["action"], args.device, args.batch_size
    )

    report: dict[str, Any] = {
        "schema": "cube_rgb_observability_audit.v1",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "device": args.device,
            "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "root_disjoint_calibration_test": True,
            "calibration_fraction": args.calibration_fraction,
            "roots": int(len(np.unique(labels["root_id"]))),
            "frames": int(len(images)),
            "frames_per_root_requested": args.frames_per_root,
            "tasks": [int(task) for task in args.tasks],
            "runtime_policy_state_access": False,
            "evaluator_state_role": "offline labels only; never feature or action input",
            "rgb_feature_definition": "8x8 average-pooled current RGB, plus goal and absolute difference for pair; action concatenated in *_action families",
            "rgb_color_geometry_definition": "largest red connected-component centroids in current/goal RGB; no simulator labels",
            "latent_feature_definition": "frozen LeWM current embedding, plus goal and current-goal difference for pair; action concatenated in *_action families",
            "action_distribution": "frozen 16-direction lateral-align/descend/close schedule; exploratory collection only",
            "collection_mode": args.collection_mode,
            "action_is_privileged": args.collection_mode == "guided_intervention",
            "guided_intervention_claim_boundary": "offline mechanism-only coverage; never a runtime policy input or planning result",
            "contact_label_source": "simulator contact flag retained as a diagnostic; physical_motion is the primary effect label",
            "ridge": args.ridge,
        },
        "data": {
            "contact_rate": float(labels["contact"].mean()),
            "physical_motion_rate": float(labels["physical_motion"].mean()),
            "task_counts": {
                str(task): int(np.sum(labels["task_id"] == task)) for task in np.unique(labels["task_id"])
            },
        },
        "metrics": {},
        "claim_boundary": (
            "Offline root-disjoint observability/effect diagnostic. It can localize "
            "an RGB versus frozen-latent information bottleneck, but does not "
            "establish causal dynamics, transfer sufficiency, or planning success."
        ),
    }
    for family, feature_map in (("rgb", image_features), ("frozen_lewm_latent", latent_features)):
        report["metrics"][family] = {}
        for feature_name, features in feature_map.items():
            print(f"[fit] family={family} feature={feature_name} dim={features.shape[-1]}")
            report["metrics"][family][feature_name] = evaluate_feature_family(
                features, labels, calibration, test, args.ridge
            )
    report["split"] = {
        "calibration_roots": np.unique(labels["root_id"][calibration]).astype(int).tolist(),
        "test_roots": np.unique(labels["root_id"][test]).astype(int).tolist(),
        "calibration_frames": int(calibration.sum()),
        "test_frames": int(test.sum()),
    }
    # Explicitly store root-level contact frequencies to expose whether the
    # diagnostic was accidentally driven by a handful of correlated frames.
    root_contact = []
    for root in np.unique(labels["root_id"]):
        root_mask = labels["root_id"] == root
        root_contact.append({
            "root_id": int(root),
            "task_id": int(labels["task_id"][root_mask][0]),
            "contact_rate": float(labels["contact"][root_mask].mean()),
            "n_frames": int(root_mask.sum()),
        })
    report["root_contact_summary"] = root_contact
    (args.output / "observability.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output / "observability.json"), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
