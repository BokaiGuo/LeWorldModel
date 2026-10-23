"""Root-disjoint RGB observed-effect local contact ranker for PushT.

The prototype asks a deliberately narrow, deployable question:

    Can a chart fitted from *previously observed RGB probe effects* reuse a
    nearby RGB contact history to rank a fixed, declared ``+y``/``-y`` pulse?

At runtime, every chart input is a rendered RGB history plus the declared
action identity.  The chart target is the RGB difference observed *after a
past probe*.  It has no simulator state, contact geometry, true successor, or
task cost access.  Privileged angles are constructed only after collection to
score the held-out effect-ranking label.

This is an offline root-disjoint action-effect diagnostic, not a controller:
the evaluator never executes the ranker's chosen action and reports no
planning, pose-success, or causal world-model claim.
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
class ProbeObservation:
    """One root's RGB history, RGB successor observations, and offline label."""

    root_id: int
    previous_rgb: np.ndarray
    current_rgb: np.ndarray
    plus_successor_rgb: np.ndarray
    minus_successor_rgb: np.ndarray
    offline_preference: int | None
    offline_preference_margin: float


def runtime_interface_boundary() -> dict[str, Any]:
    """Explicitly freeze the permissible runtime information flow."""

    return {
        "runtime_input": "RGB history and declared candidate actions only",
        "runtime_privileged_state_access": False,
        "runtime_successor_access": False,
        "runtime_task_cost_access": False,
        "offline_privileged_labels_only": True,
    }


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    """Map scalar/array angles into [-pi, pi)."""

    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def offline_orientation_preference(
    *,
    before_angle: float,
    zero_after_angle: float,
    plus_after_angle: float,
    minus_after_angle: float,
    minimum_margin: float,
) -> tuple[int | None, float]:
    """Offline-only label for the stronger mirrored angular response.

    ``1`` is ``+y`` and ``0`` is ``-y``.  A zero pulse removes common drift;
    state angles are intentionally confined to this evaluator function.
    """

    zero_delta = float(wrap_angle(float(zero_after_angle) - float(before_angle)))
    plus = abs(float(wrap_angle(float(plus_after_angle) - float(before_angle))) - zero_delta)
    minus = abs(float(wrap_angle(float(minus_after_angle) - float(before_angle))) - zero_delta)
    margin = float(plus - minus)
    return (None if abs(margin) < float(minimum_margin) else int(margin > 0.0), margin)


def _rgb_pair_tensor(previous_rgb: np.ndarray, current_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    previous = np.asarray(previous_rgb, dtype=np.uint8)
    current = np.asarray(current_rgb, dtype=np.uint8)
    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[-1] != 3:
        raise ValueError("RGB history must contain same-shaped HWC RGB frames")
    frames = torch.as_tensor(np.stack((previous, current)), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return F.interpolate(frames, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)


def rgb_history_descriptor(previous_rgb: np.ndarray, current_rgb: np.ndarray, *, image_size: int) -> np.ndarray:
    """Return current, previous, and temporal RGB components; no state path."""

    frames = _rgb_pair_tensor(previous_rgb, current_rgb, image_size)
    return torch.cat((frames[1], frames[0], frames[1] - frames[0]), dim=0).flatten().numpy().astype(np.float32)


def rgb_observed_effect(current_rgb: np.ndarray, successor_rgb: np.ndarray, *, image_size: int) -> np.ndarray:
    """Return a recorded RGB successor-minus-current effect vector."""

    frames = _rgb_pair_tensor(current_rgb, successor_rgb, image_size)
    return (frames[1] - frames[0]).flatten().numpy().astype(np.float32)


def fit_local_probe_chart(contexts: np.ndarray, plus_effects: np.ndarray, minus_effects: np.ndarray) -> dict[str, Any]:
    """Freeze a local chart whose targets are observed RGB probe effects."""

    contexts = np.asarray(contexts, dtype=np.float64)
    plus_effects = np.asarray(plus_effects, dtype=np.float64)
    minus_effects = np.asarray(minus_effects, dtype=np.float64)
    if contexts.ndim != 2 or plus_effects.ndim != 2 or minus_effects.ndim != 2:
        raise ValueError("chart arrays must be two dimensional")
    if len(contexts) < 1 or plus_effects.shape != minus_effects.shape or len(contexts) != len(plus_effects):
        raise ValueError("chart contexts and paired RGB effects must have the same nonzero row count")
    if not (np.isfinite(contexts).all() and np.isfinite(plus_effects).all() and np.isfinite(minus_effects).all()):
        raise ValueError("chart arrays must be finite")
    # Context standardization is fit on train roots only, then stored in the
    # chart.  This makes nearest-neighbor support comparable across roots.
    mean = contexts.mean(axis=0, keepdims=True)
    # Descriptors are normalized RGB values.  A per-pixel scale smaller than
    # five percent of that physical range is dominated by renderer quantization
    # and static-background coincidence in a small root panel; using it would
    # manufacture an out-of-support verdict from imperceptible RGB noise.
    scale = np.maximum(contexts.std(axis=0, keepdims=True), 5.0e-2)
    return {
        "contexts": (contexts - mean) / scale,
        "plus_effects": plus_effects,
        "minus_effects": minus_effects,
        "context_mean": mean,
        "context_scale": scale,
        **runtime_interface_boundary(),
    }


def predict_local_pair_effects(chart: dict[str, Any], context: np.ndarray, *, neighbors: int) -> dict[str, Any]:
    """Rank declared mirrored actions from train-root RGB chart entries only."""

    contexts = np.asarray(chart["contexts"], dtype=np.float64)
    plus = np.asarray(chart["plus_effects"], dtype=np.float64)
    minus = np.asarray(chart["minus_effects"], dtype=np.float64)
    query = np.asarray(context, dtype=np.float64).reshape(1, -1)
    if query.shape[1] != contexts.shape[1]:
        raise ValueError("query RGB descriptor dimension does not match chart")
    if int(neighbors) < 1:
        raise ValueError("neighbors must be positive")
    normalized = (query - np.asarray(chart["context_mean"])) / np.asarray(chart["context_scale"])
    distances = np.linalg.norm(contexts - normalized, axis=1) / math.sqrt(float(contexts.shape[1]))
    count = min(int(neighbors), len(distances))
    selected = np.argsort(distances, kind="stable")[:count]
    weights = 1.0 / np.maximum(distances[selected], 1.0e-6)
    weights /= float(weights.sum())
    plus_prediction = np.sum(plus[selected] * weights[:, None], axis=0)
    minus_prediction = np.sum(minus[selected] * weights[:, None], axis=0)
    plus_response = float(np.linalg.norm(plus_prediction))
    minus_response = float(np.linalg.norm(minus_prediction))
    return {
        "preferred_action": "plus" if plus_response >= minus_response else "minus",
        "plus_response": plus_response,
        "minus_response": minus_response,
        "predicted_contrast": abs(plus_response - minus_response),
        "support_distance": float(distances[selected[0]]),
        "neighbors_used": int(count),
        "runtime_privileged_state_access": False,
        "runtime_successor_access": False,
    }


def _finite_upper_conformal(values: np.ndarray, alpha: float) -> float | None:
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rank = int(math.ceil((len(values) + 1) * (1.0 - float(alpha))))
    return None if len(values) == 0 or rank > len(values) else float(values[rank - 1])


def calibrate_probe_gate(
    *, support_distances: np.ndarray,
    observed_contrasts: np.ndarray,
    alpha: float,
    minimum_roots: int,
) -> dict[str, Any]:
    """Calibrate a fixed support/contrast gate only from calibration roots.

    The support threshold uses finite-sample split-conformal rank.  The
    contrast floor is a predeclared lower empirical quantile, not a coverage
    certificate.  Either a missing finite threshold or too few calibration
    roots produces an explicit fail-closed result.
    """

    support = np.asarray(support_distances, dtype=np.float64).reshape(-1)
    contrast = np.asarray(observed_contrasts, dtype=np.float64).reshape(-1)
    if support.shape != contrast.shape or not (np.isfinite(support).all() and np.isfinite(contrast).all()):
        raise ValueError("calibration support/contrast arrays must be finite and aligned")
    threshold = _finite_upper_conformal(support, alpha)
    if len(support) < int(minimum_roots) or threshold is None:
        return {
            "status": "insufficient_calibration",
            "eligible_roots": int(len(support)),
            "minimum_roots": int(minimum_roots),
            "support_threshold": None,
            "contrast_floor": None,
        }
    lower_index = min(len(contrast) - 1, int(math.floor(float(alpha) * len(contrast))))
    return {
        "status": "calibrated",
        "eligible_roots": int(len(support)),
        "minimum_roots": int(minimum_roots),
        "support_threshold": float(threshold),
        "contrast_floor": float(np.sort(contrast)[lower_index]),
        "alpha": float(alpha),
        "contrast_floor_note": "fixed lower empirical calibration quantile; not a coverage guarantee",
    }


def gate_local_prediction(prediction: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """Release only when calibration is finite and support/contrast both pass."""

    if calibration.get("status") != "calibrated":
        return {"release": False, "reason": "insufficient_calibration"}
    if float(prediction["support_distance"]) > float(calibration["support_threshold"]):
        return {"release": False, "reason": "insufficient_support"}
    if float(prediction["predicted_contrast"]) < float(calibration["contrast_floor"]):
        return {"release": False, "reason": "insufficient_probe_contrast"}
    return {"release": True, "reason": "calibrated_rgb_local_chart"}


def _render_goal_for_state(state: np.ndarray, root_id: int) -> np.ndarray:
    """Render a visible but label-independent goal for a contact root."""

    rng = np.random.default_rng(8_700_000 + int(root_id))
    state = np.asarray(state, dtype=np.float64)
    goal = state.copy()
    displacement = rng.normal(size=2)
    displacement *= 70.0 / max(float(np.linalg.norm(displacement)), 1.0e-8)
    goal[2:4] = np.clip(state[2:4] + displacement, 70.0, 442.0)
    goal[4] = float(rng.uniform(-np.pi, np.pi))
    goal[5:] = 0.0
    return goal


def _history_action(root_id: int, magnitude: float) -> np.ndarray:
    directions = np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)), dtype=np.float32)
    return directions[int(root_id) % len(directions)] * float(magnitude)


def _one_step(env: gym.Env, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observation, _reward, _terminated, _truncated, _info = env.step(np.asarray(action, dtype=np.float32))
    return np.asarray(observation["state"], dtype=np.float64).copy(), np.asarray(env.render(), dtype=np.uint8).copy()


def collect_probe_observations(
    env: gym.Env,
    *,
    count: int,
    pool_transitions: int,
    seed: int,
    history_magnitude: float,
    pulse_magnitude: float,
    max_contact_gap: float,
    minimum_label_margin: float,
) -> tuple[list[ProbeObservation], dict[str, int]]:
    """Collect root-disjoint rendered histories and fixed-pulse RGB probes."""

    pool = collect_trajectory_data(env, int(pool_transitions), int(seed))
    candidates = [
        np.asarray(state, dtype=np.float64).copy()
        for state in np.asarray(pool["states"])
        if mode_for_state(state) == "contact"
        and float(np.linalg.norm(np.asarray(state)[2:4] - np.asarray(state)[:2])) <= float(max_contact_gap)
    ]
    rows: list[ProbeObservation] = []
    ineligible_label = 0
    plus = np.asarray([0.0, float(pulse_magnitude)], dtype=np.float32)
    minus = -plus
    for source_index, source in enumerate(candidates):
        if len(rows) >= int(count):
            break
        root_id = int(seed) * 10_000 + int(source_index)
        goal = _render_goal_for_state(source, root_id)
        env.reset(seed=root_id, options={"state": source.copy(), "goal_state": goal.copy()})
        previous = np.asarray(env.render(), dtype=np.uint8).copy()
        current_state, current = _one_step(env, _history_action(root_id, history_magnitude))
        if mode_for_state(current_state) != "contact":
            continue
        # These three reset/step branches are real past probes used to build
        # the offline chart.  No branch is available while scoring a test root.
        env.reset(seed=root_id + 1, options={"state": current_state.copy(), "goal_state": goal.copy()})
        zero_state, _zero_rgb = _one_step(env, np.zeros(2, dtype=np.float32))
        env.reset(seed=root_id + 1, options={"state": current_state.copy(), "goal_state": goal.copy()})
        plus_state, plus_rgb = _one_step(env, plus)
        env.reset(seed=root_id + 1, options={"state": current_state.copy(), "goal_state": goal.copy()})
        minus_state, minus_rgb = _one_step(env, minus)
        label, margin = offline_orientation_preference(
            before_angle=float(current_state[4]), zero_after_angle=float(zero_state[4]),
            plus_after_angle=float(plus_state[4]), minus_after_angle=float(minus_state[4]),
            minimum_margin=float(minimum_label_margin),
        )
        ineligible_label += int(label is None)
        rows.append(ProbeObservation(
            root_id=root_id,
            previous_rgb=previous,
            current_rgb=current,
            plus_successor_rgb=plus_rgb,
            minus_successor_rgb=minus_rgb,
            offline_preference=label,
            offline_preference_margin=float(margin),
        ))
    if len(rows) < int(count):
        raise RuntimeError(f"only collected {len(rows)}/{count} contact roots; increase --pool-transitions")
    return rows, {
        "requested_roots": int(count),
        "contact_candidates": int(len(candidates)),
        "offline_label_ineligible": int(ineligible_label),
    }


def _arrays(rows: Iterable[ProbeObservation], image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = list(rows)
    return (
        np.stack([rgb_history_descriptor(row.previous_rgb, row.current_rgb, image_size=image_size) for row in values]),
        np.stack([rgb_observed_effect(row.current_rgb, row.plus_successor_rgb, image_size=image_size) for row in values]),
        np.stack([rgb_observed_effect(row.current_rgb, row.minus_successor_rgb, image_size=image_size) for row in values]),
    )


def _calibration_predictions(chart: dict[str, Any], contexts: np.ndarray, plus: np.ndarray, minus: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Return calibration scores on the exact support/contrast scale deployed.

    ``plus`` and ``minus`` are retained as aligned audit inputs: their raw
    observed RGB effects belong in the collection record, but not in this gate
    score.  Comparing an un-smoothed raw calibration response to a smoothed
    KNN prediction would create a threshold-scale mismatch and reject chart
    predictions by construction.
    """

    support: list[float] = []
    contrast: list[float] = []
    for context, _plus_effect, _minus_effect in zip(contexts, plus, minus, strict=True):
        prediction = predict_local_pair_effects(chart, context, neighbors=neighbors)
        support.append(float(prediction["support_distance"]))
        contrast.append(float(prediction["predicted_contrast"]))
    return np.asarray(support), np.asarray(contrast)


def evaluate_test_rows(chart: dict[str, Any], calibration: dict[str, Any], rows: list[ProbeObservation], *, image_size: int, neighbors: int) -> dict[str, Any]:
    """Apply only RGB/local-chart values to held roots, then score offline labels."""

    records: list[dict[str, Any]] = []
    eligible = 0
    correct = 0
    released = 0
    release_correct = 0
    for row in rows:
        context = rgb_history_descriptor(row.previous_rgb, row.current_rgb, image_size=image_size)
        prediction = predict_local_pair_effects(chart, context, neighbors=neighbors)
        gate = gate_local_prediction(prediction, calibration)
        predicted_label = 1 if prediction["preferred_action"] == "plus" else 0
        label_eligible = row.offline_preference is not None
        eligible += int(label_eligible)
        correct += int(label_eligible and predicted_label == int(row.offline_preference))
        released += int(gate["release"])
        release_correct += int(gate["release"] and label_eligible and predicted_label == int(row.offline_preference))
        records.append({
            "root_id": int(row.root_id),
            "gate": gate,
            "predicted_action": str(prediction["preferred_action"]),
            "support_distance": float(prediction["support_distance"]),
            "predicted_contrast": float(prediction["predicted_contrast"]),
            # Offline evaluator fields; they do not enter chart fitting/ranking.
            "offline_privileged_preference": row.offline_preference,
            "offline_privileged_margin": float(row.offline_preference_margin),
        })
    return {
        "test_roots": int(len(rows)),
        "offline_label_eligible_roots": int(eligible),
        "all_eligible_accuracy": None if eligible == 0 else float(correct / eligible),
        "release_count": int(released),
        "release_rate": float(released / max(len(rows), 1)),
        "released_label_accuracy": None if released == 0 else float(release_correct / released),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train-roots", type=int, default=48)
    parser.add_argument("--calibration-roots", type=int, default=24)
    parser.add_argument("--test-roots", type=int, default=32)
    parser.add_argument("--pool-transitions", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026102111)
    parser.add_argument("--history-magnitude", type=float, default=0.35)
    parser.add_argument("--pulse-magnitude", type=float, default=0.55)
    parser.add_argument("--max-contact-gap", type=float, default=65.0)
    parser.add_argument("--minimum-label-margin", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--minimum-calibration-roots", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.train_roots, args.calibration_roots, args.test_roots) < 1:
        raise ValueError("each explicit train/calibration/test partition needs at least one root")
    if not 0.0 < args.pulse_magnitude <= 1.0 or not 0.0 <= args.history_magnitude <= 1.0:
        raise ValueError("PushT action magnitudes must be in [0, 1]")
    runtime = require_cuda_allocation(args.device)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=128)
    try:
        train, train_collection = collect_probe_observations(
            env, count=args.train_roots, pool_transitions=args.pool_transitions, seed=args.seed,
            history_magnitude=args.history_magnitude, pulse_magnitude=args.pulse_magnitude,
            max_contact_gap=args.max_contact_gap, minimum_label_margin=args.minimum_label_margin,
        )
        calibration, calibration_collection = collect_probe_observations(
            env, count=args.calibration_roots, pool_transitions=args.pool_transitions, seed=args.seed + 1_000_000,
            history_magnitude=args.history_magnitude, pulse_magnitude=args.pulse_magnitude,
            max_contact_gap=args.max_contact_gap, minimum_label_margin=args.minimum_label_margin,
        )
        test, test_collection = collect_probe_observations(
            env, count=args.test_roots, pool_transitions=args.pool_transitions, seed=args.seed + 2_000_000,
            history_magnitude=args.history_magnitude, pulse_magnitude=args.pulse_magnitude,
            max_contact_gap=args.max_contact_gap, minimum_label_margin=args.minimum_label_margin,
        )
    finally:
        env.close()
    root_sets = [set(row.root_id for row in split) for split in (train, calibration, test)]
    if root_sets[0] & root_sets[1] or root_sets[0] & root_sets[2] or root_sets[1] & root_sets[2]:
        raise RuntimeError("root-disjoint train/calibration/test contract violated")
    train_contexts, train_plus, train_minus = _arrays(train, args.image_size)
    calibration_contexts, calibration_plus, calibration_minus = _arrays(calibration, args.image_size)
    chart = fit_local_probe_chart(train_contexts, train_plus, train_minus)
    support, contrast = _calibration_predictions(chart, calibration_contexts, calibration_plus, calibration_minus, args.neighbors)
    gate = calibrate_probe_gate(
        support_distances=support, observed_contrasts=contrast, alpha=args.alpha,
        minimum_roots=args.minimum_calibration_roots,
    )
    evaluation = evaluate_test_rows(chart, gate, test, image_size=args.image_size, neighbors=args.neighbors)
    report = {
        "schema": "pusht_observed_effect_local_ranker.v1",
        "runtime": runtime,
        "runtime_interface": runtime_interface_boundary(),
        "protocol": {
            "environment": "swm/PushT-v1",
            "partitions": "root-disjoint train / calibration / test using non-overlapping collection seed bands",
            "declared_actions": {"plus_y": [0.0, args.pulse_magnitude], "minus_y": [0.0, -args.pulse_magnitude]},
            "chart_target": "past observed RGB successor-minus-current effect for each declared action",
            "gate": "finite conformal support threshold plus calibration-only local-chart predicted-contrast floor",
            "offline_label": "zero-drift-corrected privileged angular-response preference; evaluation only",
        },
        "collection": {"train": train_collection, "calibration": calibration_collection, "test": test_collection},
        "gate": gate,
        "evaluation": evaluation,
        "sanity_checks": {
            "root_disjoint": True,
            "test_labels_not_in_chart": True,
            "all_chart_inputs_rgb_only": True,
            "declared_actions_fixed_before_collection": True,
        },
        "claim_boundary": (
            "Offline action-effect ranking diagnostic only. It does not execute ranker-selected test actions and does not establish "
            "task-conditioned ranking, closed-loop planning gain, pose closure, or a world-model causal benefit."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "observed_effect_local_ranker.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = [
        "# PushT observed-effect local ranker", "",
        "RGB-history + declared-action local-chart audit. Privileged angular labels are evaluation-only.", "",
        f"- gate: `{gate['status']}`",
        f"- test roots: `{evaluation['test_roots']}`",
        f"- released: `{evaluation['release_count']}` ({evaluation['release_rate']:.3f})",
        f"- all eligible accuracy: `{evaluation['all_eligible_accuracy']}`",
        f"- released-label accuracy: `{evaluation['released_label_accuracy']}`", "",
        f"Claim boundary: {report['claim_boundary']}", "",
    ]
    (args.output / "observed_effect_local_ranker.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": gate, "evaluation": evaluation, "claim_boundary": report["claim_boundary"]}, indent=2))


if __name__ == "__main__":
    main()
