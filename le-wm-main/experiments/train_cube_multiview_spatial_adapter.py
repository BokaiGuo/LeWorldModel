"""Train a staged RGB-only multi-view spatial/action-effect adapter.

This is the first learned transfer of the Evo-Depth idea into LeWorldModel.
It is deliberately smaller and more auditable than a VLA: the input is a
fixed front/side RGB geometry chart, and the target is a paired RGB successor
difference (candidate pulse minus zero pulse) on the same frozen root.  No
simulator state, contact label, goal distance, or planning outcome enters the
training loss.

Stages:
  1. geometry autoencoding on the joint front/side RGB chart;
  2. frozen geometry encoder, action-effect head only;
  3. low-learning-rate joint alignment.

The resulting checkpoint is a calibration artifact, not evidence of a
successful planner until an independent closed-loop consequence is run.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_cube_rgb_observability import (
    _largest_effector_centroid,
    _make_env,
    _target_red_centroid,
)
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_cube_rgb_visual_servo import _capture_initial, _reset
from experiments.run_live_cube_rgb_pick_place import _target_red_area


XY_ACTIONS = np.asarray(
    [[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]], dtype=np.float32,
)


def _view_features(image: np.ndarray) -> np.ndarray:
    block = _target_red_centroid(image)[:2]
    effector = _largest_effector_centroid(image)[:2]
    return np.asarray(
        [*block, *effector, float(_target_red_area(image))], dtype=np.float32,
    )


def _joint_features(env: gym.Env, front: np.ndarray) -> np.ndarray:
    side = np.asarray(env.unwrapped.render(camera="side_pixels"), dtype=np.uint8)
    return np.concatenate((_view_features(front), _view_features(side))).astype(np.float32)


def _step_pulse(env: gym.Env, action_xy: np.ndarray, steps: int) -> np.ndarray:
    action = np.zeros(5, dtype=np.float32)
    action[:2] = np.asarray(action_xy, dtype=np.float32)
    obs = None
    for _ in range(int(steps)):
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    if obs is None:
        raise RuntimeError("pulse produced no observation")
    return np.asarray(obs, dtype=np.uint8)


def _paired_row(
    initial: dict[str, Any], *, descent_steps: int, pulse_steps: int,
    action_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect one root-paired RGB action-effect row."""
    env = _make_env()
    try:
        source, _info, goal_image = _reset(env, initial)
        for _ in range(int(descent_steps)):
            source, _reward, terminated, truncated, _info = env.step(
                np.asarray([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            )
            if terminated or truncated:
                break
        source = np.asarray(source, dtype=np.uint8)
        source_features = _joint_features(env, source)
        goal_xy = _target_red_centroid(goal_image)[:2]
        desired_delta = (goal_xy - source_features[:2]).astype(np.float32)
        zero_successor = _step_pulse(env, np.zeros(2, dtype=np.float32), pulse_steps)

        candidate_env = _make_env()
        try:
            candidate, _info, _goal = _reset(candidate_env, initial)
            for _ in range(int(descent_steps)):
                candidate, _reward, terminated, truncated, _info = candidate_env.step(
                    np.asarray([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
                )
                if terminated or truncated:
                    break
            candidate_successor = _step_pulse(candidate_env, action_xy, pulse_steps)
            zero_features = _joint_features(env, zero_successor)
            candidate_features = _joint_features(candidate_env, candidate_successor)
        finally:
            candidate_env.close()
        # Only the object centroid channels are action-effect targets.  The
        # zero pulse removes passive drift and preserves root pairing.
        zero_object = np.concatenate((zero_features[:2], zero_features[6:8]))
        candidate_object = np.concatenate((candidate_features[:2], candidate_features[6:8]))
        target = candidate_object - zero_object
        features = np.concatenate((source_features, np.asarray(action_xy, dtype=np.float32)))
        return features.astype(np.float32), target.astype(np.float32), desired_delta
    finally:
        env.close()


def _collect(
    initials: list[dict[str, Any]], *, descent_steps: int, pulse_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    desired: list[np.ndarray] = []
    for initial in initials:
        for action in XY_ACTIONS:
            feature, target, goal_delta = _paired_row(
                initial, descent_steps=descent_steps, pulse_steps=pulse_steps,
                action_xy=action,
            )
            features.append(feature)
            targets.append(target)
            desired.append(goal_delta)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(desired, dtype=np.float32),
    )


class SpatialAdapter(nn.Module):
    def __init__(self, input_dim: int = 10, latent_dim: int = 32, effect_dim: int = 4):
        super().__init__()
        self.geometry = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(),
        )
        self.reconstruction = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, input_dim),
        )
        self.effect = nn.Sequential(
            nn.Linear(latent_dim + 2, 64), nn.GELU(),
            nn.Linear(64, effect_dim),
        )

    def forward(self, geometry: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.geometry(geometry)
        return self.reconstruction(latent), self.effect(torch.cat((latent, action), dim=-1))


@dataclass
class Split:
    features: torch.Tensor
    targets: torch.Tensor


def _fit_stage(
    model: SpatialAdapter, split: Split, *, stage: int, epochs: int,
    lr: float, batch_size: int, device: torch.device,
    target_scale: torch.Tensor,
) -> dict[str, float]:
    if stage == 2:
        for parameter in model.geometry.parameters():
            parameter.requires_grad = False
        for parameter in model.reconstruction.parameters():
            parameter.requires_grad = False
    else:
        for parameter in model.geometry.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(lr), weight_decay=1.0e-4,
    )
    n = split.features.shape[0]
    model.train()
    for _epoch in range(int(epochs)):
        order = torch.randperm(n, device=device)
        for start in range(0, n, int(batch_size)):
            index = order[start : start + int(batch_size)]
            geometry = split.features[index, :-2]
            action = split.features[index, -2:]
            reconstruction, effect = model(geometry, action)
            reconstruction_loss = torch.mean((reconstruction - geometry) ** 2)
            effect_loss = torch.mean((effect - split.targets[index]) ** 2)
            loss = reconstruction_loss + effect_loss if stage == 1 else effect_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        reconstruction, effect = model(split.features[:, :-2], split.features[:, -2:])
        return {
            "geometry_rmse": float(torch.sqrt(torch.mean((reconstruction - split.features[:, :-2]) ** 2)).item()),
            "effect_rmse": float(torch.sqrt(torch.mean(((effect - split.targets) * target_scale) ** 2)).item()),
        }


def _evaluate(model: SpatialAdapter, split: Split, target_mean: torch.Tensor, target_scale: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        _reconstruction, effect = model(split.features[:, :-2], split.features[:, -2:])
        error = (effect - split.targets) * target_scale
        raw_effect = effect * target_scale + target_mean
        raw_target = split.targets * target_scale + target_mean
        return {
            "effect_rmse": float(torch.sqrt(torch.mean(error ** 2)).item()),
            "effect_mae": float(torch.mean(torch.abs(error)).item()),
            "sign_accuracy": float(torch.mean((raw_effect * raw_target > 0).float()).item()),
        }


def _evaluate_zero_baseline(train_targets: np.ndarray, test_targets: np.ndarray) -> dict[str, float]:
    """Evaluate a calibration-only constant predictor on held-out roots.

    The mean is fit on train roots only.  This is the minimum baseline needed
    to establish that a learned action-effect head transfers beyond the most
    common RGB delta in the calibration set.
    """
    mean_target = np.mean(train_targets, axis=0, keepdims=True)
    error = test_targets - mean_target
    nonzero = np.abs(test_targets) > 1.0e-8
    sign_accuracy = float(
        np.mean((mean_target * test_targets > 0)[nonzero])
        if np.any(nonzero) else 0.0
    )
    return {
        "effect_rmse": float(np.sqrt(np.mean(error ** 2))),
        "effect_mae": float(np.mean(np.abs(error))),
        "sign_accuracy": sign_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="reuse a previously collected root-disjoint npz panel")
    parser.add_argument("--seed", type=int, default=20261920)
    parser.add_argument("--train-roots", type=int, default=24)
    parser.add_argument("--test-roots", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument("--pulse-steps", type=int, default=4)
    parser.add_argument("--epochs-stage1", type=int, default=80)
    parser.add_argument("--epochs-stage2", type=int, default=160)
    parser.add_argument("--epochs-stage3", type=int, default=120)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        require_cuda_allocation("cuda")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if args.data is not None:
        panel = np.load(args.data)
        train_features = np.asarray(panel["train_features"], dtype=np.float32)
        train_targets = np.asarray(panel["train_targets"], dtype=np.float32)
        test_features = np.asarray(panel["test_features"], dtype=np.float32)
        test_targets = np.asarray(panel["test_targets"], dtype=np.float32)
        train_desired = np.asarray(panel.get("train_desired", np.zeros((len(train_features), 2))), dtype=np.float32)
        test_desired = np.asarray(panel.get("test_desired", np.zeros((len(test_features), 2))), dtype=np.float32)
        train_initials = []
        test_initials = []
    else:
        all_initials = [
            _capture_initial(args.seed + i, int(args.tasks[i % len(args.tasks)]))
            for i in range(int(args.train_roots) + int(args.test_roots))
        ]
        train_initials = all_initials[: int(args.train_roots)]
        test_initials = all_initials[int(args.train_roots) :]
        train_features, train_targets, train_desired = _collect(
            train_initials, descent_steps=args.descent_steps, pulse_steps=args.pulse_steps,
        )
        test_features, test_targets, test_desired = _collect(
            test_initials, descent_steps=args.descent_steps, pulse_steps=args.pulse_steps,
        )
    target_mean_np = np.mean(train_targets, axis=0, keepdims=True).astype(np.float32)
    target_scale_np = np.std(train_targets, axis=0, keepdims=True).astype(np.float32)
    target_scale_np = np.maximum(target_scale_np, 1.0e-4)
    train_targets_scaled = (train_targets - target_mean_np) / target_scale_np
    test_targets_scaled = (test_targets - target_mean_np) / target_scale_np
    target_mean = torch.as_tensor(target_mean_np, device=device)
    target_scale = torch.as_tensor(target_scale_np, device=device)
    train = Split(torch.as_tensor(train_features, device=device), torch.as_tensor(train_targets_scaled, device=device))
    test = Split(torch.as_tensor(test_features, device=device), torch.as_tensor(test_targets_scaled, device=device))
    model = SpatialAdapter(input_dim=train.features.shape[1] - 2).to(device)
    stages = {
        "stage1_geometry": _fit_stage(model, train, stage=1, epochs=args.epochs_stage1, lr=2.0e-3, batch_size=32, device=device, target_scale=target_scale),
        "stage2_action_alignment": _fit_stage(model, train, stage=2, epochs=args.epochs_stage2, lr=2.0e-3, batch_size=32, device=device, target_scale=target_scale),
        "stage3_joint_alignment": _fit_stage(model, train, stage=3, epochs=args.epochs_stage3, lr=3.0e-4, batch_size=32, device=device, target_scale=target_scale),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    held_out_test = _evaluate(model, test, target_mean, target_scale)
    zero_baseline = _evaluate_zero_baseline(train_targets, test_targets)
    npz_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
        train_desired=train_desired,
        test_desired=test_desired,
    )
    payload = {
        "schema": "cube_multiview_spatial_adapter.v1",
        "claim_boundary": "staged RGB spatial/action-effect training artifact; no closed-loop planning claim",
        "device": str(device),
        "protocol": {
            "train_roots": int(args.train_roots), "test_roots": int(args.test_roots),
            "root_disjoint": True, "descent_steps": int(args.descent_steps),
            "pulse_steps": int(args.pulse_steps), "target": "candidate_rgb_object_delta_minus_zero_rgb_object_delta",
            "input": "front_and_side_rgb_geometry_plus_action", "tasks": args.tasks,
            "data_reused": str(args.data) if args.data is not None else None,
            "target_normalization": "train_root_mean_and_std_per_effect_channel",
            "target_mean": target_mean_np.reshape(-1).tolist(),
            "target_scale": target_scale_np.reshape(-1).tolist(),
        },
        "stages": stages,
        "held_out_test": held_out_test,
        "held_out_zero_baseline": zero_baseline,
        "held_out_delta_vs_zero": {
            key: float(zero_baseline[key] - held_out_test[key])
            for key in ("effect_rmse", "effect_mae", "sign_accuracy")
        },
        "data_artifact": str(npz_path),
        "train_rows": int(train_features.shape[0]), "test_rows": int(test_features.shape[0]),
        "train_root_seeds": [int(row["seed"]) for row in train_initials],
        "test_root_seeds": [int(row["seed"]) for row in test_initials],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    checkpoint = args.output.with_suffix(".pt")
    torch.save({"model": model.state_dict(), "protocol": payload["protocol"]}, checkpoint)
    print(json.dumps({
        "device": str(device), "stages": stages,
        "held_out_test": payload["held_out_test"],
        "held_out_zero_baseline": payload["held_out_zero_baseline"],
        "held_out_delta_vs_zero": payload["held_out_delta_vs_zero"],
    }, indent=2))


if __name__ == "__main__":
    main()
