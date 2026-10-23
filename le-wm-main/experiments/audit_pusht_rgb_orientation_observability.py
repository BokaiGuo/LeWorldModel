"""Root-disjoint RGB-only PushT orientation-observability audit.

At runtime the two decoders consume only two rendered ``rgb_array`` frames:
the current frame and the immediately preceding frame.  Simulator state is
used *offline* to construct labels and is never passed into either decoder.

The two labels are deliberately narrower than a planning target:

* ``history_angle_direction`` asks whether RGB history identifies the signed
  block rotation that actually occurred between its two frames.
* ``mirrored_pulse_effect_preference`` asks which member of a fixed,
  globally predeclared ``+y``/``-y`` pulse pair produces the larger
  zero-drift-corrected absolute angular response.

The latter is a representation-observability diagnostic for within-contact
action ranking.  It does not certify a visual controller: the action bank is
fixed, labels are privileged offline measurements, and the audit neither
plans nor executes decoder-selected actions.

PushT currently exposes only one ``rgb_array`` renderer; unlike Cube it has
no side-camera API.  The artifact records this as an explicit interface
boundary rather than presenting temporal frames as multi-view observations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - environment registration
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gpu_preflight import require_cuda_allocation  # noqa: E402
from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
    mode_for_state,
)


@dataclass(frozen=True)
class RGBOrientationExample:
    """One root-level RGB history with offline-only binary labels."""

    previous_rgb: np.ndarray
    current_rgb: np.ndarray
    history_angle_direction: int | None
    mirrored_pulse_effect_preference: int | None
    root_id: int
    contact_gap: float
    preference_margin: float | None


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    """Map an angle or array of angles into [-pi, pi)."""
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def preference_label(
    before_angle: float,
    zero_after_angle: float,
    plus_after_angle: float,
    minus_after_angle: float,
    minimum_margin: float,
) -> tuple[int | None, float]:
    """Return the offline winner of a fixed mirrored pulse-effect pair.

    ``1`` denotes ``+y`` and ``0`` denotes ``-y``.  The zero pulse estimates
    passive contact drift; the label compares the *absolute action response*
    after subtracting that drift.  It intentionally contains no task goal:
    the decoder must only answer whether the contact configuration makes one
    fixed mirrored pulse more orientation-effective than the other.

    Ties and barely separated pairs are intentionally ineligible: forcing a
    label in the stiction/noise regime would turn a visual observability
    question into label noise.
    """
    zero_delta = float(wrap_angle(float(zero_after_angle) - float(before_angle)))
    plus_response = abs(float(wrap_angle(float(plus_after_angle) - float(before_angle))) - zero_delta)
    minus_response = abs(float(wrap_angle(float(minus_after_angle) - float(before_angle))) - zero_delta)
    margin = float(plus_response - minus_response)
    if abs(margin) < float(minimum_margin):
        return None, margin
    return int(margin > 0.0), margin


def rgb_history_features(previous_rgb: np.ndarray, current_rgb: np.ndarray, size: int) -> np.ndarray:
    """Runtime feature map: downsampled RGB history only, with no state path."""
    previous = np.asarray(previous_rgb, dtype=np.uint8)
    current = np.asarray(current_rgb, dtype=np.uint8)
    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[-1] != 3:
        raise ValueError("RGB history must contain same-shaped HWC RGB frames")
    frames = torch.as_tensor(np.stack([previous, current]), dtype=torch.float32)
    frames = frames.permute(0, 3, 1, 2) / 255.0
    frames = F.interpolate(frames, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    # Current, previous, and temporal difference are all RGB-derived.  The
    # explicit difference gives a linear readout access to contact motion
    # without leaking the action or simulator state.
    return torch.cat([frames[1], frames[0], frames[1] - frames[0]], dim=0).flatten().numpy().astype(np.float32)


def feature_matrix(examples: Iterable[RGBOrientationExample], size: int) -> np.ndarray:
    rows = [rgb_history_features(row.previous_rgb, row.current_rgb, size) for row in examples]
    if not rows:
        raise ValueError("cannot build an RGB feature matrix without examples")
    return np.stack(rows).astype(np.float32)


def train_binary_readout(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a small GPU/CPU linear readout on frozen RGB features.

    This is intentionally a decoder, not a world model or a planner.  Feature
    standardization is fit on the training roots only.
    """
    if train_x.ndim != 2 or eval_x.ndim != 2 or train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("feature matrices must be aligned two-dimensional arrays")
    if len(train_x) < 4 or len(np.unique(train_y)) != 2:
        raise ValueError("binary readout needs at least four training rows from both classes")
    torch.manual_seed(int(seed))
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.maximum(train_x.std(axis=0, keepdims=True), 1.0e-5)
    x = torch.as_tensor((train_x - mean) / std, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    x_eval = torch.as_tensor((eval_x - mean) / std, dtype=torch.float32, device=device)
    model = torch.nn.Linear(x.shape[1], 1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        logits = model(x).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        probability = torch.sigmoid(model(x_eval).reshape(-1)).detach().cpu().numpy()
        train_loss = float(F.binary_cross_entropy_with_logits(model(x).reshape(-1), y).item())
    return probability.astype(np.float64), {"train_loss": train_loss, "feature_dim": float(x.shape[1])}


def binary_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    if probability.shape != target.shape or target.size == 0:
        raise ValueError("probability and target must be aligned non-empty vectors")
    predicted = (probability >= 0.5).astype(np.int64)
    accuracy = float(np.mean(predicted == target))
    positive = target == 1
    negative = ~positive
    tpr = float(np.mean(predicted[positive] == 1)) if bool(positive.any()) else None
    tnr = float(np.mean(predicted[negative] == 0)) if bool(negative.any()) else None
    balanced = None if tpr is None or tnr is None else float((tpr + tnr) / 2.0)
    majority = float(max(np.mean(target == 0), np.mean(target == 1)))
    return {
        "n": int(target.size),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "majority_accuracy": majority,
        "positive_rate": float(np.mean(target == 1)),
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
    }


def format_metric(value: float | None) -> str:
    """Format possibly-unidentified held-out metrics without inventing zero."""
    return "n/a" if value is None else f"{float(value):.4g}"


def has_binary_support(target: np.ndarray) -> bool:
    """Return whether both binary labels occur in a declared split."""
    return bool(np.asarray(target, dtype=np.int64).size >= 2 and len(np.unique(target)) == 2)


def _render_goal_for_state(state: np.ndarray, root_id: int) -> np.ndarray:
    """Provide a visually separated environment goal without encoding labels.

    PushT requires a goal state for reset/render.  This goal is never passed
    to the decoder and is deliberately independent of both audit labels.
    """
    rng = np.random.default_rng(7_100_000 + int(root_id))
    state = np.asarray(state, dtype=np.float64)
    goal = state.copy()
    # Keep the goal visible but avoid exact overlap with the movable block.
    displacement = rng.normal(size=2)
    displacement = 70.0 * displacement / max(float(np.linalg.norm(displacement)), 1.0e-8)
    goal[2:4] = np.clip(state[2:4] + displacement, 70.0, 442.0)
    goal[4] = float(rng.uniform(-np.pi, np.pi))
    goal[5:] = 0.0
    return goal


def _history_action(root_id: int, magnitude: float) -> np.ndarray:
    """A predeclared global pulse; it is not a state-derived runtime feature."""
    directions = np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)), dtype=np.float32)
    return directions[int(root_id) % len(directions)] * float(magnitude)


def _roll_one(env: gym.Env, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observation, _, _terminated, _truncated, _info = env.step(np.asarray(action, dtype=np.float32))
    return np.asarray(observation["state"], dtype=np.float64).copy(), np.asarray(env.render(), dtype=np.uint8).copy()


def collect_examples(
    env: gym.Env,
    *,
    count: int,
    pool_transitions: int,
    seed: int,
    history_magnitude: float,
    pulse_magnitude: float,
    max_contact_gap: float,
    minimum_angle_motion: float,
    minimum_preference_margin: float,
) -> tuple[list[RGBOrientationExample], dict[str, Any]]:
    """Collect root-disjoint RGB histories and privileged offline labels."""
    pool = collect_trajectory_data(env, int(pool_transitions), int(seed))
    candidates = [
        np.asarray(state, dtype=np.float64).copy()
        for state in np.asarray(pool["states"])
        if mode_for_state(state) == "contact"
        and float(np.linalg.norm(np.asarray(state)[2:4] - np.asarray(state)[:2])) <= float(max_contact_gap)
    ]
    examples: list[RGBOrientationExample] = []
    ineligible_angle_motion = 0
    ineligible_preference = 0
    for source_index, source_state in enumerate(candidates):
        if len(examples) >= int(count):
            break
        root_id = int(seed) * 10_000 + source_index
        goal = _render_goal_for_state(source_state, root_id)
        # State enters only through the offline simulator reset.  Both frames
        # supplied to the decoder are rendered after the reset/step.
        env.reset(seed=root_id, options={"state": source_state.copy(), "goal_state": goal.copy()})
        previous_rgb = np.asarray(env.render(), dtype=np.uint8).copy()
        current_state, current_rgb = _roll_one(env, _history_action(root_id, history_magnitude))
        if mode_for_state(current_state) != "contact":
            continue
        history_delta = float(wrap_angle(float(current_state[4]) - float(source_state[4])))
        history_direction = None if abs(history_delta) < float(minimum_angle_motion) else int(history_delta > 0.0)
        ineligible_angle_motion += int(history_direction is None)
        # The pair is globally fixed and known before data collection.  No
        # contact normal, geometry, state coordinate, or oracle successor is
        # provided to the decoder at runtime.
        plus = np.asarray([0.0, pulse_magnitude], dtype=np.float32)
        minus = -plus
        env.reset(seed=root_id + 1, options={"state": current_state.copy(), "goal_state": goal.copy()})
        zero_state, _ = _roll_one(env, np.zeros(2, dtype=np.float32))
        env.reset(seed=root_id + 1, options={"state": current_state.copy(), "goal_state": goal.copy()})
        plus_state, _ = _roll_one(env, plus)
        env.reset(seed=root_id + 1, options={"state": current_state.copy(), "goal_state": goal.copy()})
        minus_state, _ = _roll_one(env, minus)
        preference, margin = preference_label(
            float(current_state[4]), float(zero_state[4]), float(plus_state[4]), float(minus_state[4]),
            minimum_preference_margin,
        )
        ineligible_preference += int(preference is None)
        examples.append(RGBOrientationExample(
            previous_rgb=previous_rgb,
            current_rgb=current_rgb,
            history_angle_direction=history_direction,
            mirrored_pulse_effect_preference=preference,
            root_id=root_id,
            contact_gap=float(np.linalg.norm(current_state[:2] - current_state[2:4])),
            preference_margin=margin,
        ))
    if len(examples) < int(count):
        raise RuntimeError(
            f"only collected {len(examples)}/{count} contact RGB roots; "
            f"increase --pool-transitions or relax --max-contact-gap"
        )
    return examples, {
        "requested_roots": int(count),
        "pool_transitions": int(pool_transitions),
        "contact_candidates": int(len(candidates)),
        "angle_motion_ineligible_roots": int(ineligible_angle_motion),
        "preference_ineligible_roots": int(ineligible_preference),
    }


def evaluate_label(
    train: list[RGBOrientationExample],
    held: list[RGBOrientationExample],
    label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selector = (
        (lambda item: item.history_angle_direction)
        if label == "history_angle_direction"
        else (lambda item: item.mirrored_pulse_effect_preference)
    )
    train_rows = [item for item in train if selector(item) is not None]
    held_rows = [item for item in held if selector(item) is not None]
    if len(train_rows) < 4 or len(held_rows) < 2:
        return {"status": "insufficient_eligible_rows", "train_rows": len(train_rows), "held_rows": len(held_rows)}
    train_y = np.asarray([int(selector(item)) for item in train_rows], dtype=np.int64)
    held_y = np.asarray([int(selector(item)) for item in held_rows], dtype=np.int64)
    if not has_binary_support(train_y):
        return {"status": "single_class_training", "train_rows": len(train_rows), "held_rows": len(held_rows)}
    if not has_binary_support(held_y):
        return {
            "status": "single_class_held_out",
            "train_rows": len(train_rows),
            "held_rows": len(held_rows),
            "held_positive_rate": float(np.mean(held_y == 1)),
            "boundary": "A one-class held-out split cannot estimate balanced accuracy or demonstrate RGB decodability.",
        }
    probability, fit = train_binary_readout(
        feature_matrix(train_rows, args.image_size), train_y,
        feature_matrix(held_rows, args.image_size), device=args.device,
        epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, seed=args.seed + (11 if label == "history_angle_direction" else 29),
    )
    return {
        "status": "evaluated",
        "train_rows": len(train_rows),
        "held_rows": len(held_rows),
        "fit": fit,
        "held_out": binary_metrics(probability, held_y),
        "runtime_input": "two rendered PushT rgb_array frames only; no state, goal, contact geometry, action vector, or successor",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train-roots", type=int, default=96)
    parser.add_argument("--held-roots", type=int, default=32)
    parser.add_argument("--pool-transitions", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=202610211)
    parser.add_argument("--history-magnitude", type=float, default=0.35)
    parser.add_argument("--pulse-magnitude", type=float, default=0.55)
    parser.add_argument("--max-contact-gap", type=float, default=65.0)
    parser.add_argument("--minimum-angle-motion", type=float, default=0.002)
    parser.add_argument("--minimum-preference-margin", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_roots < 4 or args.held_roots < 2 or args.pool_transitions < args.train_roots + args.held_roots:
        raise ValueError("need disjoint train/held roots and a sufficiently large transition pool")
    if not 0.0 < args.pulse_magnitude <= 1.0 or not 0.0 <= args.history_magnitude <= 1.0:
        raise ValueError("pulse magnitudes must be in the PushT action range")
    runtime = require_cuda_allocation(args.device)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=128)
    try:
        train, train_audit = collect_examples(
            env, count=args.train_roots, pool_transitions=args.pool_transitions, seed=args.seed,
            history_magnitude=args.history_magnitude, pulse_magnitude=args.pulse_magnitude,
            max_contact_gap=args.max_contact_gap,
            minimum_angle_motion=args.minimum_angle_motion,
            minimum_preference_margin=args.minimum_preference_margin,
        )
        held, held_audit = collect_examples(
            env, count=args.held_roots, pool_transitions=args.pool_transitions,
            seed=args.seed + 1_000_000, history_magnitude=args.history_magnitude,
            pulse_magnitude=args.pulse_magnitude,
            max_contact_gap=args.max_contact_gap, minimum_angle_motion=args.minimum_angle_motion,
            minimum_preference_margin=args.minimum_preference_margin,
        )
    finally:
        env.close()
    if set(item.root_id for item in train).intersection(item.root_id for item in held):
        raise RuntimeError("root-disjointness violated")
    report = {
        "schema": "pusht_rgb_orientation_observability.v1",
        "runtime": runtime,
        "protocol": {
            "environment": "swm/PushT-v1",
            "split": "disjoint behavior-trajectory contact roots generated from non-overlapping seeds",
            "runtime_views": ["rgb_array previous", "rgb_array current"],
            "side_view_available": False,
            "side_view_boundary": "PushT.render() has no camera argument; this is a single-view temporal RGB audit, not front/side multi-view evidence.",
            "runtime_input": "rendered RGB history only",
            "offline_labels": "simulator angle states, zero successor, and paired mirrored-pulse successors only",
            "mirrored_pulse_pair": {"plus_y": [0.0, args.pulse_magnitude], "minus_y": [0.0, -args.pulse_magnitude]},
            "decoder": "linear RGB-history readout fitted only on train roots",
        },
        "collection": {"train": train_audit, "held_out": held_audit},
        "results": {
            "history_angle_direction": evaluate_label(train, held, "history_angle_direction", args),
            "mirrored_pulse_effect_preference": evaluate_label(train, held, "mirrored_pulse_effect_preference", args),
        },
        "sanity_checks": {
            "root_disjoint": True,
            "all_rgb_hwc": bool(all(item.current_rgb.ndim == 3 and item.current_rgb.shape[-1] == 3 for item in train + held)),
            "runtime_features_have_no_privileged_state": True,
            "torch_cuda_device": runtime["device"],
        },
        "claim_boundary": (
            "Representation observability only. It does not establish a learned world-model action ranking, "
            "counterfactual prediction, closed-loop planning gain, contact retention, or PushT pose success."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "rgb_orientation_observability.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# PushT RGB orientation observability audit", "",
        "This is a root-disjoint, RGB-history decoder audit. Offline simulator state provides labels only.", "",
        "| label | status | held roots | accuracy | balanced accuracy | majority baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, result in report["results"].items():
        metrics = result.get("held_out", {})
        lines.append(
            f"| {name} | {result['status']} | {result.get('held_rows', 0)} | "
            f"{format_metric(metrics.get('accuracy'))} | {format_metric(metrics.get('balanced_accuracy'))} | "
            f"{format_metric(metrics.get('majority_accuracy'))} |"
        )
    lines += ["", f"Claim boundary: {report['claim_boundary']}", ""]
    (args.output / "rgb_orientation_observability.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": report["results"], "claim_boundary": report["claim_boundary"]}, indent=2))


if __name__ == "__main__":
    main()
