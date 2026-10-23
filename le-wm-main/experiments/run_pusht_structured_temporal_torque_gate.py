"""Prospective, reject-capable PushT torque-preference confirmation.

This is a fresh-root successor to the failed high-dimensional temporal
confirmation.  The previous panel established that a 3,848-dimensional linear
readout can interpolate 48 training roots (near-zero training loss) without
transferring to calibration roots.  This protocol changes the *train-side
representation and sample budget*, never the old held-out panel:

* a fixed 4x4 RGB motion chart replaces full-resolution pixels;
* train-only PCA limits the supervised readout to 24 directions;
* a confidence threshold is selected on a separate calibration cohort;
* insufficient calibration support fails closed rather than selecting a
  torque preference.

All train/calibration/outcome roots are new high seed bands.  The fixed label
is still the privileged zero-drift-corrected response to a two-step +/-y
pulse.  Thus a positive result would only establish selective *offline*
visual torque-preference transfer; it does not establish a world model,
counterfactual planning, or pose closure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from experiments.run_pusht_balanced_temporal_torque_confirmation import (  # noqa: E402
    DEFAULT_SEED_BANDS,
    RootExample,
    assert_root_separation,
    bootstrap_balanced_accuracy,
    collect_band,
    labels,
    resize_rgb,
    validate_history,
    validate_seed_bands,
)

gym.logger.min_level = 40

SCHEMA = "pusht.structured_temporal_torque_gate.v1"
FRESH_SEED_BANDS = {
    "train": DEFAULT_SEED_BANDS["train"] + 300_000,
    "calibration": DEFAULT_SEED_BANDS["calibration"] + 300_000,
    "test": DEFAULT_SEED_BANDS["test"] + 300_000,
}
CONFIDENCE_GRID = tuple(round(value, 2) for value in np.arange(0.0, 0.91, 0.05))


def structured_temporal_feature(frames: np.ndarray, actions: np.ndarray, grid_size: int) -> np.ndarray:
    """Fixed, low-dimensional RGB motion chart plus declared action history.

    The last RGB frame conveys current contact geometry.  Each preceding
    frame-to-frame difference preserves the direction of recent observed
    motion.  The grid is intentionally tiny: unlike the failed full-pixel
    readout, a 4x4 chart has 248 dimensions for a four-action history.
    """

    frames, actions = validate_history(frames, actions)
    if int(grid_size) < 2 or int(grid_size) > 8:
        raise ValueError("structured grid_size must be within [2, 8]")
    chart = resize_rgb(frames, int(grid_size))
    values = torch.cat((chart[-1:], chart[1:] - chart[:-1]), dim=0).flatten().numpy()
    return np.concatenate((values.astype(np.float32), actions.reshape(-1).astype(np.float32)))


def feature_matrix(rows: Iterable[RootExample], grid_size: int) -> np.ndarray:
    matrix = np.stack(
        [structured_temporal_feature(row.rgb_frames, row.history_actions, int(grid_size)) for row in rows]
    ).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("structured RGB/action feature matrix is not finite")
    return matrix


def fit_pca_linear_readout(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
    *,
    device: str,
    pca_rank: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a train-only standardized PCA chart then a regularized classifier."""

    if len(np.unique(train_y)) != 2:
        raise ValueError("train cohort must include both preference classes")
    if not 1 <= int(pca_rank) < min(train_x.shape):
        raise ValueError("pca_rank must be positive and strictly below train sample/feature count")
    torch.manual_seed(int(seed))
    mean = train_x.mean(axis=0, keepdims=True)
    scale = np.maximum(train_x.std(axis=0, keepdims=True), 1.0e-5)
    x_train = torch.as_tensor((train_x - mean) / scale, dtype=torch.float32, device=device)
    x_evaluation = torch.as_tensor((evaluation_x - mean) / scale, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    # The singular vectors are estimated entirely from train roots.  No held
    # calibration/outcome RGB, labels, or candidate effects enter this map.
    _u, singular_values, right = torch.linalg.svd(x_train, full_matrices=False)
    basis = right[: int(pca_rank)].transpose(0, 1).contiguous()
    train_chart = x_train @ basis
    evaluation_chart = x_evaluation @ basis
    model = torch.nn.Linear(int(pca_rank), 1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        loss = F.binary_cross_entropy_with_logits(model(train_chart).reshape(-1), y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        train_loss = float(F.binary_cross_entropy_with_logits(model(train_chart).reshape(-1), y_train).item())
        probability = torch.sigmoid(model(evaluation_chart).reshape(-1)).detach().cpu().numpy()
    metadata = {
        "input_feature_dimension": int(x_train.shape[1]),
        "pca_rank": int(pca_rank),
        "train_loss": train_loss,
        "top_singular_value": float(singular_values[0].item()),
        "retained_variance_fraction": float(
            torch.square(singular_values[: int(pca_rank)]).sum().item()
            / max(torch.square(singular_values).sum().item(), 1.0e-12)
        ),
    }
    del x_train, x_evaluation, y_train, _u, singular_values, right, basis, train_chart, evaluation_chart
    torch.cuda.synchronize(torch.device(device))
    return probability.astype(np.float64), metadata


def balanced_accuracy(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    prediction = (probability >= 0.5).astype(np.int64)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("balanced accuracy needs both preference classes")
    return float((np.mean(prediction[target == 0] == 0) + np.mean(prediction[target == 1] == 1)) / 2.0)


def confidence(probability: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(probability, dtype=np.float64) - 0.5) * 2.0


def select_confidence_gate(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    minimum_calibration_ba: float,
    minimum_coverage: float,
    minimum_per_class: int,
) -> dict[str, Any]:
    """Freeze the largest supported threshold meeting calibration quality.

    Calibration chooses only a confidence gate.  It never changes features,
    PCA rank, model weights, pulse magnitude, label construction, or the
    outcome roots.  Returning ``authorized=False`` is an intended safe result.
    """

    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    candidates: list[dict[str, Any]] = []
    for threshold in CONFIDENCE_GRID:
        retained = confidence(probability) >= float(threshold)
        per_class = {str(label): int(np.sum(retained & (target == label))) for label in (0, 1)}
        coverage = float(np.mean(retained))
        supported = min(per_class.values()) >= int(minimum_per_class)
        score = balanced_accuracy(probability[retained], target[retained]) if supported else None
        candidates.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "retained_per_class": per_class,
                "balanced_accuracy": score,
                "eligible": bool(
                    supported
                    and coverage >= float(minimum_coverage)
                    and float(score) >= float(minimum_calibration_ba)
                ),
            }
        )
    authorized = [entry for entry in candidates if entry["eligible"]]
    if not authorized:
        return {
            "authorized": False,
            "reason": "no fixed confidence threshold has sufficient calibration support and quality",
            "threshold": None,
            "candidates": candidates,
        }
    # Prefer coverage, then accuracy, then the more conservative threshold.
    chosen = max(
        authorized,
        key=lambda entry: (float(entry["coverage"]), float(entry["balanced_accuracy"]), float(entry["threshold"])),
    )
    return {"authorized": True, "reason": "calibration quality/support gate passed", "threshold": chosen["threshold"], "candidates": candidates}


def evaluate_gate(
    probability: np.ndarray,
    target: np.ndarray,
    gate: dict[str, Any],
    *,
    bootstrap: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Apply an already frozen gate to outcome roots without refitting."""

    if not gate["authorized"]:
        return {"executed": False, "reason": str(gate["reason"])}
    retained = confidence(probability) >= float(gate["threshold"])
    per_class = {str(label): int(np.sum(retained & (target == label))) for label in (0, 1)}
    if min(per_class.values()) < 2:
        return {
            "executed": False,
            "reason": "frozen calibration gate leaves fewer than two outcome roots in a preference class",
            "threshold": float(gate["threshold"]),
            "retained_per_class": per_class,
        }
    selected_probability = np.asarray(probability, dtype=np.float64)[retained]
    selected_target = np.asarray(target, dtype=np.int64)[retained]
    return {
        "executed": True,
        "threshold": float(gate["threshold"]),
        "coverage": float(np.mean(retained)),
        "retained_roots": int(np.sum(retained)),
        "retained_per_class": per_class,
        "balanced_accuracy": bootstrap_balanced_accuracy(
            selected_probability, selected_target, draws=int(bootstrap), seed=int(seed), device=device
        ),
    }


def confirmation_rule(gate: dict[str, Any], outcome: dict[str, Any], *, minimum_outcome_ba: float,
                      minimum_outcome_ci_lower: float, minimum_outcome_coverage: float) -> dict[str, Any]:
    """A fail-closed decision independent of held-out outcome labels."""

    passed = bool(
        gate["authorized"]
        and outcome.get("executed", False)
        and float(outcome["coverage"]) >= float(minimum_outcome_coverage)
        and float(outcome["balanced_accuracy"]["estimate"]) >= float(minimum_outcome_ba)
        and float(outcome["balanced_accuracy"]["ci95_low"]) >= float(minimum_outcome_ci_lower)
    )
    observed = (
        {
            "coverage": float(outcome["coverage"]),
            "balanced_accuracy": float(outcome["balanced_accuracy"]["estimate"]),
            "balanced_accuracy_ci95_low": float(outcome["balanced_accuracy"]["ci95_low"]),
        }
        if outcome.get("executed", False)
        else {"reason": str(outcome["reason"])}
    )
    return {
        "status": "confirmed_selective_offline_transfer" if passed else "not_confirmed",
        "predeclared_rule": {
            "calibration_gate": "largest fixed confidence threshold with support and BA",
            "outcome_minimum_balanced_accuracy": float(minimum_outcome_ba),
            "outcome_minimum_ci95_lower": float(minimum_outcome_ci_lower),
            "outcome_minimum_coverage": float(minimum_outcome_coverage),
        },
        "observed": observed,
        "interpretation": (
            "The frozen low-dimensional temporal chart passed a selective offline torque-preference transfer test. "
            "It still does not execute a readout-selected action or establish planning/pose closure."
            if passed else
            "The fresh selective offline temporal transfer test was not confirmed; retain the fail-closed gate and do not tune on these outcome roots."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train-seed-band", type=int, default=FRESH_SEED_BANDS["train"])
    parser.add_argument("--calibration-seed-band", type=int, default=FRESH_SEED_BANDS["calibration"])
    parser.add_argument("--test-seed-band", type=int, default=FRESH_SEED_BANDS["test"])
    parser.add_argument("--train-per-class", type=int, default=128)
    parser.add_argument("--calibration-per-class", type=int, default=64)
    parser.add_argument("--test-per-class", type=int, default=128)
    parser.add_argument("--pool-transitions", type=int, default=8192)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--history-magnitude", type=float, default=0.30)
    parser.add_argument("--max-contact-gap", type=float, default=65.0)
    parser.add_argument("--minimum-preference-margin", type=float, default=2.0e-4)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--pca-rank", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--minimum-calibration-ba", type=float, default=0.65)
    parser.add_argument("--minimum-calibration-coverage", type=float, default=0.25)
    parser.add_argument("--minimum-calibration-per-class", type=int, default=12)
    parser.add_argument("--minimum-outcome-ba", type=float, default=0.70)
    parser.add_argument("--minimum-outcome-ci-lower", type=float, default=0.60)
    parser.add_argument("--minimum-outcome-coverage", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite an existing immutable protocol output: {args.output}")
    if args.history_steps < 2 or args.pool_transitions < 100:
        raise ValueError("protocol needs multi-frame history and a substantial root pool")
    if not (0.0 < args.history_magnitude <= 1.0 and 0.0 < args.minimum_preference_margin):
        raise ValueError("history magnitude and label margin must be positive")
    if not (0.5 < args.minimum_calibration_ba <= 1.0 and 0.0 < args.minimum_calibration_coverage <= 1.0):
        raise ValueError("calibration gate thresholds are invalid")
    if not (0.5 < args.minimum_outcome_ba <= 1.0 and 0.5 <= args.minimum_outcome_ci_lower <= 1.0):
        raise ValueError("outcome confirmation thresholds are invalid")
    bands = {"train": args.train_seed_band, "calibration": args.calibration_seed_band, "test": args.test_seed_band}
    validate_seed_bands(bands)
    runtime = require_cuda_allocation(args.device)
    started = time.perf_counter()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=128)
    try:
        train, train_info = collect_band(env, split="train", band_seed=int(args.train_seed_band), per_class=int(args.train_per_class), pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps), history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap), minimum_margin=float(args.minimum_preference_margin))
        calibration, calibration_info = collect_band(env, split="calibration", band_seed=int(args.calibration_seed_band), per_class=int(args.calibration_per_class), pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps), history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap), minimum_margin=float(args.minimum_preference_margin))
        test, test_info = collect_band(env, split="test", band_seed=int(args.test_seed_band), per_class=int(args.test_per_class), pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps), history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap), minimum_margin=float(args.minimum_preference_margin))
    finally:
        env.close()
    assert_root_separation(train, calibration, test)
    train_x, calibration_x, test_x = (feature_matrix(rows, int(args.grid_size)) for rows in (train, calibration, test))
    train_y, calibration_y, test_y = (labels(rows) for rows in (train, calibration, test))
    combined_x = np.concatenate((calibration_x, test_x), axis=0)
    combined_probability, fit = fit_pca_linear_readout(train_x, train_y, combined_x, device=args.device, pca_rank=int(args.pca_rank), epochs=int(args.epochs), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay), seed=int(args.train_seed_band + 31))
    calibration_probability = combined_probability[: len(calibration)]
    test_probability = combined_probability[len(calibration):]
    gate = select_confidence_gate(calibration_probability, calibration_y, minimum_calibration_ba=float(args.minimum_calibration_ba), minimum_coverage=float(args.minimum_calibration_coverage), minimum_per_class=int(args.minimum_calibration_per_class))
    outcome = evaluate_gate(test_probability, test_y, gate, bootstrap=int(args.bootstrap), seed=int(args.train_seed_band + 71), device=args.device)
    report = {
        "schema": SCHEMA,
        "runtime": runtime,
        "protocol": {
            "environment": "swm/PushT-v1",
            "mechanism": "fixed privileged zero-drift-corrected +/-y torque preference label",
            "inputs": "rendered RGB history and declared history actions only",
            "representation": "last 4x4 RGB frame plus 4x4 frame differences plus history actions",
            "readout": "train-only standardization + train-only PCA + regularized linear classifier",
            "calibration": "fixed confidence-grid abstention gate; no refit or representation selection",
            "outcome": "fresh root-disjoint balanced labels, untouched until frozen gate application",
        },
        "seed_bands": bands,
        "collection": {"train": train_info, "calibration": calibration_info, "test": test_info},
        "fit": fit,
        "calibration": {"ungated_balanced_accuracy": balanced_accuracy(calibration_probability, calibration_y), "gate": gate},
        "outcome": outcome,
        "confirmation": confirmation_rule(gate, outcome, minimum_outcome_ba=float(args.minimum_outcome_ba), minimum_outcome_ci_lower=float(args.minimum_outcome_ci_lower), minimum_outcome_coverage=float(args.minimum_outcome_coverage)),
        "sanity_checks": {
            "root_disjoint": True,
            "source_state_digest_disjoint": True,
            "all_cohorts_collected_before_fitting": True,
            "pca_train_only": True,
            "calibration_gate_frozen_before_outcome_evaluation": True,
            "no_runtime_action_selection": True,
            "immutable_output": True,
        },
        "claim_boundary": "This is a selective offline visual torque-preference experiment. It does not execute selected actions and does not establish planning, pose closure, causal dynamics, or a universal world model.",
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True)
    (args.output / "structured_temporal_torque_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["confirmation"]["status"], "output": str(args.output), "fit": fit, "outcome": outcome}, sort_keys=True))


if __name__ == "__main__":
    main()
