"""Train an RGB action-effect adapter on post-pulse states.

The single-pulse adapter is not trained on observations after an earlier
action.  This script creates a root-disjoint two-pulse panel and trains the
same SpatialAdapter on the *second* pulse conditioned on the RGB chart after
the first pulse.  It is a targeted repair for the distribution-shift failure
observed in the support-preserving two-pulse audit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_cube_rgb_observability import _make_env, _target_red_centroid
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_cube_rgb_visual_servo import _capture_initial, _reset
from experiments.train_cube_multiview_spatial_adapter import (
    SpatialAdapter,
    Split,
    _fit_stage,
    _joint_features,
    _prepare_contact,
    _step_pulse,
)


AXIS_ACTIONS = np.asarray(
    [[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]], dtype=np.float32,
)


def _restore_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _restore_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        array = np.asarray(value)
        if array.dtype != object and array.ndim > 0:
            return array
        return [_restore_arrays(item) for item in value]
    return value


def _branch_row(
    initial: dict[str, Any], first_action: np.ndarray, second_action: np.ndarray,
    *, descent_steps: int, approach_steps: int, close_steps: int,
    approach_gain: float, pulse_steps: int, pixel_mapping: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect a paired second-pulse row with identical first-pulse history."""
    candidate_env = _make_env()
    zero_env = _make_env()
    try:
        candidate, _info, goal = _reset(candidate_env, initial)
        zero, _info, _goal = _reset(zero_env, initial)
        candidate = _prepare_contact(
            candidate_env, candidate, descent_steps=descent_steps,
            approach_steps=approach_steps, close_steps=close_steps,
            approach_gain=approach_gain, pixel_action_mapping=pixel_mapping,
        )
        zero = _prepare_contact(
            zero_env, zero, descent_steps=descent_steps,
            approach_steps=approach_steps, close_steps=close_steps,
            approach_gain=approach_gain, pixel_action_mapping=pixel_mapping,
        )
        candidate = _step_pulse(candidate_env, first_action, pulse_steps)
        zero = _step_pulse(zero_env, first_action, pulse_steps)
        mid_features = _joint_features(candidate_env, candidate)
        candidate = _step_pulse(candidate_env, second_action, pulse_steps)
        zero = _step_pulse(zero_env, np.zeros(2, dtype=np.float32), pulse_steps)
        candidate_features = _joint_features(candidate_env, candidate)
        zero_features = _joint_features(zero_env, zero)
        candidate_object = np.concatenate((candidate_features[:2], candidate_features[6:8]))
        zero_object = np.concatenate((zero_features[:2], zero_features[6:8]))
        target = candidate_object - zero_object
        desired = (_target_red_centroid(goal)[:2] - mid_features[:2]).astype(np.float32)
        features = np.concatenate((mid_features, np.asarray(second_action, dtype=np.float32)))
        return features.astype(np.float32), target.astype(np.float32), desired
    finally:
        candidate_env.close()
        zero_env.close()


def _collect(
    initials: list[dict[str, Any]], *, descent_steps: int, approach_steps: int,
    close_steps: int, approach_gain: float, pulse_steps: int,
    pixel_mapping: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    desired: list[np.ndarray] = []
    for initial in initials:
        for first_action in AXIS_ACTIONS:
            for second_action in AXIS_ACTIONS:
                row, target, goal_delta = _branch_row(
                    initial, first_action, second_action,
                    descent_steps=descent_steps, approach_steps=approach_steps,
                    close_steps=close_steps, approach_gain=approach_gain,
                    pulse_steps=pulse_steps, pixel_mapping=pixel_mapping,
                )
                features.append(row)
                targets.append(target)
                desired.append(goal_delta)
    return np.asarray(features), np.asarray(targets), np.asarray(desired)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--initials-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261980)
    parser.add_argument("--train-roots", type=int, default=8)
    parser.add_argument("--test-roots", type=int, default=4)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs-stage1", type=int, default=80)
    parser.add_argument("--epochs-stage2", type=int, default=160)
    parser.add_argument("--epochs-stage3", type=int, default=120)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        require_cuda_allocation("cuda")
    device = torch.device(args.device)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    protocol = base["protocol"]
    mapping_data = (protocol.get("approach_mapping") or {}).get("mapping")
    pixel_mapping = np.asarray(mapping_data, dtype=np.float32) if mapping_data is not None else None
    total = int(args.train_roots) + int(args.test_roots)
    if args.initials_json is None:
        initials = [
            _capture_initial(int(args.seed) + index, int(args.tasks[index % len(args.tasks)]))
            for index in range(total)
        ]
    else:
        manifest = json.loads(args.initials_json.read_text())
        initials = [_restore_arrays(root) for root in manifest["roots"]]
        if len(initials) != total:
            raise ValueError(f"initials manifest has {len(initials)} roots, expected {total}")
    train_initials = initials[: int(args.train_roots)]
    test_initials = initials[int(args.train_roots):]
    common = dict(
        descent_steps=int(protocol["descent_steps"]),
        approach_steps=int(protocol["approach_steps"]),
        close_steps=int(protocol["close_steps"]),
        approach_gain=float(protocol["approach_gain"]),
        pulse_steps=int(protocol["pulse_steps"]),
        pixel_mapping=pixel_mapping,
    )
    train_features, train_targets, train_desired = _collect(train_initials, **common)
    test_features, test_targets, test_desired = _collect(test_initials, **common)
    # Persist the expensive collection before fitting so an artifact-writing
    # failure cannot discard the root-disjoint data panel.
    np.savez_compressed(
        args.output.with_suffix(".collected.npz"),
        train_features=train_features, train_targets=train_targets,
        test_features=test_features, test_targets=test_targets,
        train_desired=train_desired, test_desired=test_desired,
        candidate_actions=AXIS_ACTIONS,
    )
    target_mean_np = np.mean(train_targets, axis=0, keepdims=True).astype(np.float32)
    target_scale_np = np.maximum(np.std(train_targets, axis=0, keepdims=True), 1.0e-4).astype(np.float32)
    train_scaled = (train_targets - target_mean_np) / target_scale_np
    test_scaled = (test_targets - target_mean_np) / target_scale_np
    target_scale = torch.as_tensor(target_scale_np, device=device)
    train = Split(torch.as_tensor(train_features, device=device), torch.as_tensor(train_scaled, device=device))
    test = Split(torch.as_tensor(test_features, device=device), torch.as_tensor(test_scaled, device=device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    model = SpatialAdapter(input_dim=10).to(device)
    stages = {
        "stage1_geometry": _fit_stage(model, train, stage=1, epochs=args.epochs_stage1, lr=2.0e-3, batch_size=32, device=device, target_scale=target_scale),
        "stage2_action_alignment": _fit_stage(model, train, stage=2, epochs=args.epochs_stage2, lr=2.0e-3, batch_size=32, device=device, target_scale=target_scale),
        "stage3_joint_alignment": _fit_stage(model, train, stage=3, epochs=args.epochs_stage3, lr=3.0e-4, batch_size=32, device=device, target_scale=target_scale),
    }
    model.eval()
    with torch.no_grad():
        _reconstruction, prediction = model(test.features[:, :-2], test.features[:, -2:])
    prediction = prediction.detach().cpu().numpy() * target_scale_np + target_mean_np
    error = prediction - test_targets
    common_json = dict(common)
    common_json["pixel_mapping"] = (
        pixel_mapping.tolist() if pixel_mapping is not None else None
    )
    payload = {
        "schema": "cube_sequence_effect_adapter.v1",
        "claim_boundary": "RGB second-pulse action-effect training artifact; no closed-loop claim",
        "device": str(device),
        "protocol": {
            **common_json,
            "root_disjoint": True,
            "first_action_support": AXIS_ACTIONS.tolist(),
            "second_action_support": AXIS_ACTIONS.tolist(),
            "target": "second-pulse candidate RGB object delta minus matched zero successor",
            "base_checkpoint": str(args.base_checkpoint),
            "target_mean": target_mean_np.reshape(-1).tolist(),
            "target_scale": target_scale_np.reshape(-1).tolist(),
        },
        "stages": stages,
        "held_out_test": {
            "effect_rmse": float(np.sqrt(np.mean(error ** 2))),
            "effect_mae": float(np.mean(np.abs(error))),
        },
        "data_artifact": str(args.output.with_suffix(".npz")),
        "train_root_seeds": [int(row["seed"]) for row in train_initials],
        "test_root_seeds": [int(row["seed"]) for row in test_initials],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        train_features=train_features, train_targets=train_targets,
        test_features=test_features, test_targets=test_targets,
        train_desired=train_desired, test_desired=test_desired,
        candidate_actions=AXIS_ACTIONS,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "protocol": payload["protocol"]}, args.output.with_suffix(".pt"))
    print(json.dumps(payload["held_out_test"], indent=2))


if __name__ == "__main__":
    main()
