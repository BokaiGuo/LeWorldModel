"""Evaluate support-preserving two-pulse RGB action ranking.

This is a held-root consequence audit for the cube spatial adapter.  It
compares an open-loop diagonal macro against two axis-aligned pulses with an
RGB re-observation between pulses.  Both candidates use the same total number
of environment steps.  The predictor is queried only on actions present in
the four-direction training support; no simulator contact/state is read by
the controller or the predictor.

The true successor is used only after ranking, to compute held-out ranking
agreement.  The intermediate RGB observation is allowed because the method
under test is explicitly receding-horizon.
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
    _joint_features,
    _prepare_contact,
    _step_pulse,
)


def _pairwise_agreement(true_scores: np.ndarray, predicted_scores: np.ndarray) -> float:
    values: list[float] = []
    for i in range(len(true_scores)):
        for j in range(i + 1, len(true_scores)):
            if true_scores[i] == true_scores[j]:
                continue
            values.append(float(
                (true_scores[i] - true_scores[j])
                * (predicted_scores[i] - predicted_scores[j]) > 0
            ))
    return float(np.mean(values)) if values else 0.0


def _restore_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _restore_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        array = np.asarray(value)
        if array.dtype != object and array.ndim > 0:
            return array
        return [_restore_arrays(item) for item in value]
    return value


def _macro_bank() -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal open-loop and cardinal two-pulse candidates."""
    signs = np.asarray([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], dtype=np.float32)
    diagonals = signs * (0.5 / np.sqrt(2.0))
    axis_sequences = np.stack(
        [np.stack([np.asarray([sx * 0.5, 0.0]), np.asarray([0.0, sy * 0.5])])
         for sx, sy in signs],
        axis=0,
    ).astype(np.float32)
    return diagonals.astype(np.float32), axis_sequences


def _predict_effect(
    model: SpatialAdapter,
    features: np.ndarray,
    action: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    geometry = torch.from_numpy(np.asarray(features, dtype=np.float32)[None]).to(device)
    action_tensor = torch.from_numpy(np.asarray(action, dtype=np.float32)[None]).to(device)
    with torch.no_grad():
        _reconstruction, effect = model(geometry, action_tensor)
    return (effect.detach().cpu().numpy()[0] * target_scale + target_mean).astype(np.float32)


def _prepare_root(
    seed: int,
    task: int,
    *,
    descent_steps: int,
    approach_steps: int,
    close_steps: int,
    approach_gain: float,
    pixel_mapping: np.ndarray | None,
    initial: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    env = _make_env()
    if initial is None:
        initial = _capture_initial(seed, task)
    source, _info, goal = _reset(env, initial)
    source = _prepare_contact(
        env, source, descent_steps=descent_steps, approach_steps=approach_steps,
        close_steps=close_steps, approach_gain=approach_gain,
        pixel_action_mapping=pixel_mapping,
    )
    source_features = _joint_features(env, source)
    desired = (_target_red_centroid(goal)[:2] - source_features[:2]).astype(np.float32)
    return env, initial, source_features, desired


def _run_sequence(
    initial: dict[str, Any],
    sequence: np.ndarray,
    *,
    descent_steps: int,
    pulse_steps: int,
    approach_steps: int,
    close_steps: int,
    approach_gain: float,
    pixel_mapping: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    env = _make_env()
    try:
        source, _info, _goal = _reset(env, initial)
        source = _prepare_contact(
            env, source, descent_steps=descent_steps, approach_steps=approach_steps,
            close_steps=close_steps, approach_gain=approach_gain,
            pixel_action_mapping=pixel_mapping,
        )
        source_features = _joint_features(env, source)
        intermediate: list[np.ndarray] = []
        for index, action in enumerate(np.asarray(sequence, dtype=np.float32)):
            source = _step_pulse(env, action, pulse_steps)
            if index < len(sequence) - 1:
                intermediate.append(_joint_features(env, source))
        successor_features = _joint_features(env, source)
        return source_features, successor_features, intermediate
    finally:
        env.close()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    protocol = checkpoint["protocol"]
    model = SpatialAdapter(input_dim=10).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    target_mean = np.asarray(protocol["target_mean"], dtype=np.float32)
    target_scale = np.asarray(protocol["target_scale"], dtype=np.float32)
    sequence_model = None
    sequence_mean = target_mean
    sequence_scale = target_scale
    if args.sequence_checkpoint is not None:
        sequence_checkpoint = torch.load(args.sequence_checkpoint, map_location=device, weights_only=True)
        sequence_model = SpatialAdapter(input_dim=10).to(device)
        sequence_model.load_state_dict(sequence_checkpoint["model"])
        sequence_model.eval()
        sequence_protocol = sequence_checkpoint["protocol"]
        sequence_mean = np.asarray(sequence_protocol["target_mean"], dtype=np.float32)
        sequence_scale = np.asarray(sequence_protocol["target_scale"], dtype=np.float32)
    mapping_data = (protocol.get("approach_mapping") or {}).get("mapping")
    pixel_mapping = np.asarray(mapping_data, dtype=np.float32) if mapping_data is not None else None
    diagonals, axis_sequences = _macro_bank()
    summary = json.loads(Path(args.summary_json).read_text())
    if args.initials_json is None:
        test_seeds = [int(value) for value in summary["test_root_seeds"]]
        initials = [
            _capture_initial(seed, int(args.tasks[index % len(args.tasks)]))
            for index, seed in enumerate(test_seeds)
        ]
    else:
        manifest = json.loads(args.initials_json.read_text())
        initials = [_restore_arrays(root) for root in manifest["roots"]]
        test_seeds = [int(root["seed"]) for root in initials]
    tasks = [int(root["task_id"]) for root in initials]
    direct_true: list[float] = []
    direct_pred: list[float] = []
    composed_true: list[float] = []
    composed_pred: list[float] = []
    rows: list[dict[str, Any]] = []
    for seed, task, initial in zip(test_seeds, tasks, initials):
        env, initial, source_features, desired = _prepare_root(
            seed, task, descent_steps=int(protocol["descent_steps"]),
            approach_steps=int(protocol["approach_steps"]),
            close_steps=int(protocol["close_steps"]),
            approach_gain=float(protocol["approach_gain"]),
            pixel_mapping=pixel_mapping,
            initial=initial,
        )
        try:
            zero = _step_pulse(env, np.zeros(2, dtype=np.float32), 2 * args.pulse_steps)
            zero_features = _joint_features(env, zero)
        finally:
            env.close()
        direct_true_scores: list[float] = []
        direct_pred_scores: list[float] = []
        composed_true_scores: list[float] = []
        composed_pred_scores: list[float] = []
        for diagonal, axes in zip(diagonals, axis_sequences):
            direct_source, direct_successor, _ = _run_sequence(
                initial, np.stack([diagonal, diagonal]),
                descent_steps=int(protocol["descent_steps"]), pulse_steps=args.pulse_steps,
                approach_steps=int(protocol["approach_steps"]), close_steps=int(protocol["close_steps"]),
                approach_gain=float(protocol["approach_gain"]), pixel_mapping=pixel_mapping,
            )
            composed_source, composed_successor, intermediate = _run_sequence(
                initial, axes,
                descent_steps=int(protocol["descent_steps"]), pulse_steps=args.pulse_steps,
                approach_steps=int(protocol["approach_steps"]), close_steps=int(protocol["close_steps"]),
                approach_gain=float(protocol["approach_gain"]), pixel_mapping=pixel_mapping,
            )
            direct_true_effect = direct_successor[:2] - zero_features[:2]
            composed_true_effect = composed_successor[:2] - zero_features[:2]
            direct_pred_effect = 2.0 * _predict_effect(
                model, direct_source, diagonal, target_mean, target_scale, device,
            )[:2]
            composed_pred_effect = _predict_effect(
                model, composed_source, axes[0], target_mean, target_scale, device,
            )[:2]
            if intermediate:
                second_model = sequence_model if sequence_model is not None else model
                composed_pred_effect += _predict_effect(
                    second_model, intermediate[0], axes[1],
                    sequence_mean if sequence_model is not None else target_mean,
                    sequence_scale if sequence_model is not None else target_scale,
                    device,
                )[:2]
            direct_true_scores.append(float(direct_true_effect @ desired))
            direct_pred_scores.append(float(direct_pred_effect @ desired))
            composed_true_scores.append(float(composed_true_effect @ desired))
            composed_pred_scores.append(float(composed_pred_effect @ desired))
        direct_true_array = np.asarray(direct_true_scores)
        direct_pred_array = np.asarray(direct_pred_scores)
        composed_true_array = np.asarray(composed_true_scores)
        composed_pred_array = np.asarray(composed_pred_scores)
        direct_true.extend(direct_true_scores)
        direct_pred.extend(direct_pred_scores)
        composed_true.extend(composed_true_scores)
        composed_pred.extend(composed_pred_scores)
        rows.append({
            "seed": seed,
            "task": task,
            "direct_top1": int(np.argmax(direct_true_array) == np.argmax(direct_pred_array)),
            "direct_pairwise": _pairwise_agreement(direct_true_array, direct_pred_array),
            "composed_top1": int(np.argmax(composed_true_array) == np.argmax(composed_pred_array)),
            "composed_pairwise": _pairwise_agreement(composed_true_array, composed_pred_array),
            "direct_true_scores": direct_true_array.tolist(),
            "direct_pred_scores": direct_pred_array.tolist(),
            "composed_true_scores": composed_true_array.tolist(),
            "composed_pred_scores": composed_pred_array.tolist(),
        })
    return {
        "schema": "cube_support_preserving_two_pulse.v1",
        "protocol": {
            "root_disjoint": True,
            "candidate_count": int(len(diagonals)),
            "total_pulse_steps": int(2 * args.pulse_steps),
            "direct_candidate": "same diagonal pulse twice without intermediate re-observation",
            "composed_candidate": "axis pulse, RGB re-observation, orthogonal axis pulse",
            "privileged_state_used_for_ranking": False,
            "second_pulse_checkpoint": (
                str(args.sequence_checkpoint) if args.sequence_checkpoint is not None else None
            ),
        },
        "direct_open_loop": {
            "groups": len(rows),
            "top1_agreement": float(np.mean([row["direct_top1"] for row in rows])),
            "pairwise_agreement": float(np.mean([row["direct_pairwise"] for row in rows])),
        },
        "support_preserving_receding": {
            "groups": len(rows),
            "top1_agreement": float(np.mean([row["composed_top1"] for row in rows])),
            "pairwise_agreement": float(np.mean([row["composed_pairwise"] for row in rows])),
        },
        "paired_rows": rows,
        "delta_receding_minus_direct": {
            "top1_agreement": float(np.mean([row["composed_top1"] for row in rows]) - np.mean([row["direct_top1"] for row in rows])),
            "pairwise_agreement": float(np.mean([row["composed_pairwise"] for row in rows]) - np.mean([row["direct_pairwise"] for row in rows])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--initials-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-checkpoint", type=Path, default=None)
    parser.add_argument("--pulse-steps", type=int, default=4)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        require_cuda_allocation("cuda")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
