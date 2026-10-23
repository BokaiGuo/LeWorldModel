"""Root-disjoint temporal RGB action-effect representation audit for PushT.

This standalone audit compares two *fixed* runtime feature maps for a narrow
within-contact question: which member of a globally declared mirrored torque
pulse pair has the larger offline-measured angular response?

``raw_two_frame`` consumes the final two RGB frames, their RGB difference,
and the final declared history action.  ``temporal_difference_history``
consumes the current RGB frame, every adjacent RGB difference in a fixed
multi-frame history, and every declared history action.  The only runtime
inputs to either map are rendered RGB and declared actions.  Simulator state,
zero-pulse drift, and mirrored-pulse successors exist solely to construct and
evaluate offline labels.

This is a representation-observability audit, not a world-model, successor
predictor, action executor, or planning evaluation.  A positive held-out
balanced accuracy would show only that the specified RGB/action history
contains information about this fixed offline label under this split.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm/PushT-v1
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

gym.logger.min_level = 40


@dataclass(frozen=True)
class TemporalTorqueExample:
    """Rendered temporal context plus one offline-only mirrored-pulse label."""

    root_id: int
    rgb_frames: np.ndarray
    declared_history_actions: np.ndarray
    offline_mirrored_preference: int | None
    offline_preference_margin: float


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    """Map an angle or array of angles into ``[-pi, pi)``."""

    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def offline_mirrored_preference_label(
    *,
    before_angle: float,
    zero_after_angle: float,
    plus_after_angle: float,
    minus_after_angle: float,
    minimum_margin: float,
) -> tuple[int | None, float]:
    """Build a privileged offline label; it is never a feature or input.

    Label ``1`` denotes the globally declared ``+y`` pulse and ``0`` denotes
    ``-y``.  A matched zero pulse removes passive angular drift.  Near ties are
    intentionally ineligible because they are not a stable preference target.
    """

    zero_delta = float(wrap_angle(float(zero_after_angle) - float(before_angle)))
    plus_response = abs(float(wrap_angle(float(plus_after_angle) - float(before_angle))) - zero_delta)
    minus_response = abs(float(wrap_angle(float(minus_after_angle) - float(before_angle))) - zero_delta)
    margin = float(plus_response - minus_response)
    return (None if abs(margin) < float(minimum_margin) else int(margin > 0.0), margin)


def _validate_rgb_action_history(rgb_frames: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(rgb_frames, dtype=np.uint8)
    action_history = np.asarray(actions, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 2:
        raise ValueError("RGB history must be [frames>=2, height, width, 3]")
    if action_history.ndim != 2 or action_history.shape[1] != 2:
        raise ValueError("declared action history must be [steps, 2]")
    if action_history.shape[0] != frames.shape[0] - 1:
        raise ValueError("temporal representation requires one action per RGB transition")
    if not np.isfinite(action_history).all():
        raise ValueError("declared actions must be finite")
    return frames, action_history


def _downsample_rgb(frames: np.ndarray, image_size: int) -> torch.Tensor:
    if int(image_size) < 2:
        raise ValueError("image_size must be at least 2")
    tensor = torch.as_tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return F.interpolate(tensor, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)


def raw_two_frame_effect_features(rgb_frames: np.ndarray, actions: np.ndarray, *, image_size: int) -> np.ndarray:
    """Final two RGB frames/difference plus their declared action.

    This is deliberately the raw two-frame reference.  It does not have
    access to earlier images or actions, even if they were collected to build
    the matched multi-frame comparator.
    """

    frames, action_history = _validate_rgb_action_history(rgb_frames, actions)
    pair = _downsample_rgb(frames[-2:], image_size)
    rgb = torch.cat((pair[0], pair[1], pair[1] - pair[0]), dim=0).flatten().numpy()
    return np.concatenate((rgb.astype(np.float32), action_history[-1].astype(np.float32)), axis=0)


def temporal_difference_history_features(rgb_frames: np.ndarray, actions: np.ndarray, *, image_size: int) -> np.ndarray:
    """Fixed multi-frame RGB-difference/history map with declared actions.

    The representation has no learned selector, state estimate, successor
    observation, or offline label path: ``current RGB || adjacent RGB
    differences || declared action sequence``.
    """

    frames, action_history = _validate_rgb_action_history(rgb_frames, actions)
    resized = _downsample_rgb(frames, image_size)
    differences = resized[1:] - resized[:-1]
    rgb_history = torch.cat((resized[-1:], differences), dim=0).flatten().numpy()
    return np.concatenate((rgb_history.astype(np.float32), action_history.reshape(-1).astype(np.float32)), axis=0)


FEATURE_MAPS: dict[str, Callable[..., np.ndarray]] = {
    "raw_two_frame": raw_two_frame_effect_features,
    "temporal_difference_history": temporal_difference_history_features,
}


def build_feature_matrix(
    examples: Iterable[TemporalTorqueExample], *, representation: str, image_size: int
) -> np.ndarray:
    """Build runtime-only feature rows without reading offline labels."""

    if representation not in FEATURE_MAPS:
        raise ValueError(f"unknown representation: {representation}")
    feature_map = FEATURE_MAPS[representation]
    rows = [feature_map(item.rgb_frames, item.declared_history_actions, image_size=int(image_size)) for item in examples]
    if not rows:
        raise ValueError("cannot create a feature matrix from zero rows")
    matrix = np.stack(rows).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("runtime RGB/action feature matrix must be finite")
    return matrix


def assert_root_disjoint(train: Iterable[TemporalTorqueExample], held: Iterable[TemporalTorqueExample]) -> None:
    """Reject any root reuse before feature fitting or evaluation."""

    train_ids = {int(item.root_id) for item in train}
    held_ids = {int(item.root_id) for item in held}
    overlap = train_ids.intersection(held_ids)
    if overlap:
        raise ValueError(f"root-disjoint split required; overlapping roots: {sorted(overlap)[:5]}")


def _has_binary_support(target: np.ndarray) -> bool:
    target = np.asarray(target, dtype=np.int64)
    return bool(len(target) >= 2 and len(np.unique(target)) == 2)


def _binary_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if probability.shape != target.shape or target.size == 0:
        raise ValueError("probabilities and binary targets must be aligned and nonempty")
    predicted = (probability >= 0.5).astype(np.int64)
    positive = target == 1
    negative = ~positive
    tpr = None if not bool(positive.any()) else float(np.mean(predicted[positive] == 1))
    tnr = None if not bool(negative.any()) else float(np.mean(predicted[negative] == 0))
    return {
        "n": int(target.size),
        "coverage": float(target.size),
        "accuracy": float(np.mean(predicted == target)),
        "balanced_accuracy": None if tpr is None or tnr is None else float((tpr + tnr) / 2.0),
        "majority_accuracy": float(max(np.mean(target == 0), np.mean(target == 1))),
        "positive_rate": float(np.mean(target == 1)),
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
    }


def train_linear_readout(
    train_x: np.ndarray,
    train_y: np.ndarray,
    held_x: np.ndarray,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit the same train-root-only linear decoder for either representation."""

    train_x = np.asarray(train_x, dtype=np.float32)
    held_x = np.asarray(held_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64).reshape(-1)
    if train_x.ndim != 2 or held_x.ndim != 2 or train_x.shape[1] != held_x.shape[1]:
        raise ValueError("readout matrices must be aligned two-dimensional arrays")
    if len(train_x) != len(train_y) or len(train_x) < 4 or not _has_binary_support(train_y):
        raise ValueError("readout needs at least four train roots containing both labels")
    torch.manual_seed(int(seed))
    mean = train_x.mean(axis=0, keepdims=True)
    scale = np.maximum(train_x.std(axis=0, keepdims=True), 1.0e-5)
    x_train = torch.as_tensor((train_x - mean) / scale, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    x_held = torch.as_tensor((held_x - mean) / scale, dtype=torch.float32, device=device)
    model = torch.nn.Linear(x_train.shape[1], 1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        logits = model(x_train).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        train_loss = float(F.binary_cross_entropy_with_logits(model(x_train).reshape(-1), y_train).item())
        probability = torch.sigmoid(model(x_held).reshape(-1)).detach().cpu().numpy()
    return probability.astype(np.float64), {"feature_dim": int(x_train.shape[1]), "train_loss": train_loss}


def _render_goal_for_state(state: np.ndarray, root_id: int) -> np.ndarray:
    """Create a rendered goal independently of any preference label."""

    rng = np.random.default_rng(9_150_000 + int(root_id))
    state = np.asarray(state, dtype=np.float64)
    goal = state.copy()
    direction = rng.normal(size=2)
    direction *= 70.0 / max(float(np.linalg.norm(direction)), 1.0e-8)
    goal[2:4] = np.clip(state[2:4] + direction, 70.0, 442.0)
    goal[4] = float(rng.uniform(-np.pi, np.pi))
    goal[5:] = 0.0
    return goal


def _fixed_history_actions(root_id: int, steps: int, magnitude: float) -> np.ndarray:
    """Globally fixed cyclic pulse schedule; root ID chooses phase, not action by state."""

    axes = np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)), dtype=np.float32)
    indices = (np.arange(int(steps), dtype=np.int64) + int(root_id)) % len(axes)
    return (axes[indices] * float(magnitude)).astype(np.float32)


def _roll(env: gym.Env, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observation, _reward, _terminated, _truncated, _info = env.step(np.asarray(action, dtype=np.float32))
    return np.asarray(observation["state"], dtype=np.float64).copy(), np.asarray(env.render(), dtype=np.uint8).copy()


def collect_examples(
    env: gym.Env,
    *,
    count: int,
    pool_transitions: int,
    seed: int,
    history_steps: int,
    history_magnitude: float,
    pulse_magnitude: float,
    max_contact_gap: float,
    minimum_preference_margin: float,
) -> tuple[list[TemporalTorqueExample], dict[str, int]]:
    """Collect root-disjoint rendered histories and privileged offline labels."""

    pool = collect_trajectory_data(env, int(pool_transitions), int(seed))
    source_states = [
        np.asarray(state, dtype=np.float64).copy()
        for state in np.asarray(pool["states"])
        if mode_for_state(state) == "contact"
        and float(np.linalg.norm(np.asarray(state)[2:4] - np.asarray(state)[:2])) <= float(max_contact_gap)
    ]
    rows: list[TemporalTorqueExample] = []
    left_contact = 0
    ineligible_margin = 0
    plus = np.asarray([0.0, float(pulse_magnitude)], dtype=np.float32)
    minus = -plus
    for source_index, source in enumerate(source_states):
        if len(rows) >= int(count):
            break
        root_id = int(seed) * 10_000 + int(source_index)
        goal = _render_goal_for_state(source, root_id)
        actions = _fixed_history_actions(root_id, int(history_steps), float(history_magnitude))
        # Source state is only an offline reset handle.  The feature path below
        # starts after rendering and sees RGB/action values alone.
        env.reset(seed=root_id, options={"state": source.copy(), "goal_state": goal.copy()})
        frames = [np.asarray(env.render(), dtype=np.uint8).copy()]
        current_state = source.copy()
        for action in actions:
            current_state, rgb = _roll(env, action)
            frames.append(rgb)
        if mode_for_state(current_state) != "contact":
            left_contact += 1
            continue
        before_angle = float(current_state[4])
        env.reset(seed=root_id + 1, options={"state": current_state.copy(), "goal_state": goal.copy()})
        zero_state, _ = _roll(env, np.zeros(2, dtype=np.float32))
        env.reset(seed=root_id + 1, options={"state": current_state.copy(), "goal_state": goal.copy()})
        plus_state, _ = _roll(env, plus)
        env.reset(seed=root_id + 1, options={"state": current_state.copy(), "goal_state": goal.copy()})
        minus_state, _ = _roll(env, minus)
        preference, margin = offline_mirrored_preference_label(
            before_angle=before_angle,
            zero_after_angle=float(zero_state[4]),
            plus_after_angle=float(plus_state[4]),
            minus_after_angle=float(minus_state[4]),
            minimum_margin=float(minimum_preference_margin),
        )
        ineligible_margin += int(preference is None)
        rows.append(TemporalTorqueExample(
            root_id=root_id,
            rgb_frames=np.stack(frames).astype(np.uint8),
            declared_history_actions=actions,
            offline_mirrored_preference=preference,
            offline_preference_margin=float(margin),
        ))
    if len(rows) < int(count):
        raise RuntimeError(
            f"only collected {len(rows)}/{count} contact roots; increase --pool-transitions or relax --max-contact-gap"
        )
    return rows, {
        "requested_roots": int(count),
        "collected_roots": int(len(rows)),
        "contact_source_candidates": int(len(source_states)),
        "left_contact_during_history": int(left_contact),
        "ineligible_preference_margin": int(ineligible_margin),
    }


def evaluate_representation(
    train: list[TemporalTorqueExample],
    held: list[TemporalTorqueExample],
    *,
    representation: str,
    image_size: int,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    """Compare one fixed representation under the common root split/label."""

    train_rows = [row for row in train if row.offline_mirrored_preference is not None]
    held_rows = [row for row in held if row.offline_mirrored_preference is not None]
    outcome: dict[str, Any] = {
        "representation": representation,
        "train_roots": len(train),
        "held_roots": len(held),
        "train_eligible": len(train_rows),
        "held_eligible": len(held_rows),
        "held_label_coverage": float(len(held_rows) / max(len(held), 1)),
        "runtime_input": "rendered RGB frames plus declared history actions only",
    }
    if len(train_rows) < 4 or len(held_rows) < 2:
        outcome.update({"status": "insufficient_eligible_roots"})
        return outcome
    train_y = np.asarray([int(row.offline_mirrored_preference) for row in train_rows], dtype=np.int64)
    held_y = np.asarray([int(row.offline_mirrored_preference) for row in held_rows], dtype=np.int64)
    if not _has_binary_support(train_y):
        outcome.update({"status": "single_class_training", "train_positive_rate": float(np.mean(train_y == 1))})
        return outcome
    if not _has_binary_support(held_y):
        outcome.update({
            "status": "single_class_held_out",
            "held_positive_rate": float(np.mean(held_y == 1)),
            "boundary": "Balanced accuracy is unidentified when held-out labels have one class.",
        })
        return outcome
    probability, fit = train_linear_readout(
        build_feature_matrix(train_rows, representation=representation, image_size=image_size),
        train_y,
        build_feature_matrix(held_rows, representation=representation, image_size=image_size),
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    outcome.update({"status": "evaluated", "fit": fit, "held_out": _binary_metrics(probability, held_y)})
    return outcome


def _format(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train-roots", type=int, default=96)
    parser.add_argument("--held-roots", type=int, default=32)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=202610213)
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--history-magnitude", type=float, default=0.30)
    parser.add_argument("--pulse-magnitude", type=float, default=0.55)
    parser.add_argument("--max-contact-gap", type=float, default=65.0)
    parser.add_argument("--minimum-preference-margin", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_roots < 4 or args.held_roots < 2 or args.history_steps < 2:
        raise ValueError("need train/held roots and at least two temporal history steps")
    if args.pool_transitions < args.train_roots + args.held_roots:
        raise ValueError("pool-transitions must cover the requested root counts")
    if not 0.0 < args.history_magnitude <= 1.0 or not 0.0 < args.pulse_magnitude <= 1.0:
        raise ValueError("PushT action magnitudes must be in (0, 1]")
    runtime = require_cuda_allocation(args.device)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=128)
    try:
        train, train_collection = collect_examples(
            env, count=args.train_roots, pool_transitions=args.pool_transitions, seed=args.seed,
            history_steps=args.history_steps, history_magnitude=args.history_magnitude,
            pulse_magnitude=args.pulse_magnitude, max_contact_gap=args.max_contact_gap,
            minimum_preference_margin=args.minimum_preference_margin,
        )
        held, held_collection = collect_examples(
            env, count=args.held_roots, pool_transitions=args.pool_transitions, seed=args.seed + 1_000_000,
            history_steps=args.history_steps, history_magnitude=args.history_magnitude,
            pulse_magnitude=args.pulse_magnitude, max_contact_gap=args.max_contact_gap,
            minimum_preference_margin=args.minimum_preference_margin,
        )
    finally:
        env.close()
    assert_root_disjoint(train, held)
    results = {
        name: evaluate_representation(
            train, held, representation=name, image_size=args.image_size, device=args.device,
            epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            seed=args.seed + offset,
        )
        for offset, name in enumerate(("raw_two_frame", "temporal_difference_history"), start=1)
    }
    report = {
        "schema": "pusht.temporal_rgb_action_effect_representation.v1",
        "runtime": runtime,
        "protocol": {
            "environment": "swm/PushT-v1",
            "split": "root-disjoint contact roots from non-overlapping behavior-trajectory seed bands",
            "runtime_input": "RGB history plus declared actions only",
            "runtime_forbidden": ["simulator state", "contact geometry", "goal", "offline label", "successor RGB"],
            "offline_labels_only": "simulator angle plus zero/+y/-y real successors",
            "mirrored_action_pair": {"plus_y": [0.0, args.pulse_magnitude], "minus_y": [0.0, -args.pulse_magnitude]},
            "history": {"steps": args.history_steps, "fixed_action_schedule": "cyclic cardinal pulses indexed only by root id"},
            "readout": "same train-root-only standardized linear binary readout for both fixed feature maps",
        },
        "collection": {"train": train_collection, "held_out": held_collection},
        "sanity_checks": {
            "root_disjoint": True,
            "feature_label_separation": True,
            "same_examples_and_labels_for_both_representations": True,
            "same_runtime_action_history_for_both_representations": True,
            "cuda_device": runtime["device"],
        },
        "results": results,
        "claim_boundary": (
            "This offline root-disjoint representation audit neither executes a decoder-selected action nor tests a "
            "world-model prediction. Balanced accuracy/coverage quantify only separability of a fixed privileged "
            "mirrored-pulse preference label from declared RGB/action histories; they do not establish causal dynamics, "
            "action ranking in deployment, contact retention, pose closure, or planning benefit."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "temporal_rgb_action_effect_representation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PushT temporal RGB action-effect representation audit", "",
        "Both rows use identical root-disjoint examples, offline labels, action schedule, and linear decoder. "
        "Only the fixed RGB/action representation differs.", "",
        "| representation | status | held label coverage | held n | accuracy | balanced accuracy | majority baseline |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        metrics = result.get("held_out", {})
        lines.append(
            f"| {name} | {result['status']} | {_format(result['held_label_coverage'])} | "
            f"{metrics.get('n', result['held_eligible'])} | {_format(metrics.get('accuracy'))} | "
            f"{_format(metrics.get('balanced_accuracy'))} | {_format(metrics.get('majority_accuracy'))} |"
        )
    lines += ["", f"Claim boundary: {report['claim_boundary']}", ""]
    (args.output / "temporal_rgb_action_effect_representation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": results, "claim_boundary": report["claim_boundary"]}, indent=2))


if __name__ == "__main__":
    main()
