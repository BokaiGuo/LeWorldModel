"""Candidate-bank matched RGB confirmation for signed PushT torque response.

This is the deployment-compatible successor to the radial contact-frame
confirmation.  The earlier experiment constructed its own block-centred
radial/tangent pulse pair.  The closed-loop controller, however, acts through
the nearest-polygon-boundary candidate bank.  On an asymmetric T these are not
interchangeable action semantics.  This protocol therefore extracts its two
opposite-moment pulses from that exact declared bank before forming a label.

It remains an offline identifiability experiment: simulator successors form
labels only and no decoded action is executed online.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm/PushT-v1

from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_pusht_balanced_temporal_torque_confirmation import (
    BalancedCohortUnavailable,
    RootExample,
    TORQUE_DURATION,
    assert_root_separation,
    balanced_accuracy,
    collect_band,
    confirmation_rule,
    evaluate,
    feature_matrix,
    fit_linear_readout,
    labels,
    paired_bootstrap_delta,
    signed_offline_preference,
    validate_seed_bands,
)
from experiments.run_pusht_mode_aware_shared_candidate import (
    action_moment_sign,
    boundary_inward_normal,
    candidate_actions,
)
from experiments.run_pusht_signed_temporal_torque_confirmation import write_pre_outcome_rejection


SCHEMA = "pusht.candidate_bank_signed_temporal_torque_confirmation.v3_boundary_frame"
SIGNED_PAIR_CONFIDENCE_GRID = tuple(float(value) for value in np.arange(0.0, 1.0, 0.1))


def signed_pair_task_ranking(
    probability_positive_moment_more_positive: np.ndarray,
    rows: list[RootExample],
    *,
    release_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Transduce signed response probabilities into a declared two-action task choice.

    Each root is evaluated under both externally supplied desired-turn signs.
    This prevents a one-sided turn convention from being mistaken for action
    ranking.  The task utility is the observed zero-drift-corrected angular
    response in the requested direction.  It is intentionally narrower than
    goal-pose cost: no claim about translation, contact retention, multi-pulse
    composition, or full-bank regret follows from this bridge.
    """

    probability = np.asarray(probability_positive_moment_more_positive, dtype=np.float64).reshape(-1)
    if probability.shape != (len(rows),):
        raise ValueError("one signed-response probability is required for every root")
    if not np.isfinite(probability).all() or np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("signed-response probabilities must be finite values in [0, 1]")
    released = np.ones(len(rows), dtype=bool) if release_mask is None else np.asarray(release_mask, dtype=bool)
    if released.shape != probability.shape:
        raise ValueError("release mask must contain one decision for every root")
    contexts: list[dict[str, Any]] = []
    for item, row, is_released in zip(probability, rows, released):
        if not bool(is_released):
            continue
        predicted_positive_better = bool(item >= 0.5)
        observed_positive_better = bool(int(row.preference) == 1)
        response_gap = abs(float(row.zero_corrected_plus_response) - float(row.zero_corrected_minus_response))
        for desired_turn_sign in (1, -1):
            predicted_action = "plus" if predicted_positive_better == (desired_turn_sign > 0) else "minus"
            observed_action = "plus" if observed_positive_better == (desired_turn_sign > 0) else "minus"
            contexts.append({
                "root_id": int(row.root_id),
                "desired_turn_sign": int(desired_turn_sign),
                "predicted_member": predicted_action,
                "observed_best_member": observed_action,
                "correct": bool(predicted_action == observed_action),
                "signed_effect_regret": float(0.0 if predicted_action == observed_action else response_gap),
            })
    correct = np.asarray([context["correct"] for context in contexts], dtype=np.float64)
    regrets = np.asarray([context["signed_effect_regret"] for context in contexts], dtype=np.float64)
    by_turn = {
        str(sign): (
            float(np.mean([context["correct"] for context in contexts if int(context["desired_turn_sign"]) == sign]))
            if contexts else None
        )
        for sign in (1, -1)
    }
    return {
        "protocol": "each held root is evaluated for both declared desired-turn signs; the same frozen sign readout is only transduced to plus/minus selection",
        "candidate_pair_ranking_accuracy": float(correct.mean()) if len(correct) else None,
        "candidate_pair_ranking_accuracy_by_desired_turn_sign": by_turn,
        "mean_signed_effect_regret": float(regrets.mean()) if len(regrets) else None,
        "max_signed_effect_regret": float(regrets.max(initial=0.0)),
        "contexts": int(len(contexts)),
        "roots": int(len(rows)),
        "released_roots": int(np.sum(released)),
        "coverage": float(np.mean(released)),
        "abstained_roots": int(len(rows) - np.sum(released)),
        "claim_boundary": "A two-member local action-effect ranking transduction, not full-bank task regret, pose success, contact retention, or closed-loop planning.",
    }


def signed_pair_confidence(probability: np.ndarray) -> np.ndarray:
    """Distance from the fixed binary decision boundary, in [0, 1]."""

    probability = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(probability).all() or np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("signed-response probabilities must be finite values in [0, 1]")
    return 2.0 * np.abs(probability - 0.5)


def select_signed_pair_release_gate(
    calibration_probability: np.ndarray,
    calibration_target: np.ndarray,
    *, minimum_balanced_accuracy: float, minimum_coverage: float, minimum_per_class: int,
) -> dict[str, Any]:
    """Freeze an abstention threshold using calibration roots only."""

    probability = np.asarray(calibration_probability, dtype=np.float64)
    target = np.asarray(calibration_target, dtype=np.int64)
    if probability.shape != target.shape or set(np.unique(target)) != {0, 1}:
        raise ValueError("calibration gate needs aligned probabilities and both signed-response classes")
    candidates: list[dict[str, Any]] = []
    for threshold in SIGNED_PAIR_CONFIDENCE_GRID:
        retain = signed_pair_confidence(probability) >= float(threshold)
        per_class = {str(label): int(np.sum(retain & (target == label))) for label in (0, 1)}
        coverage = float(np.mean(retain))
        supported = min(per_class.values()) >= int(minimum_per_class)
        # Keep this definition shared with the primary confirmation statistic:
        # a selective-release threshold must not silently optimize a different
        # scoring convention than the fixed 0.5 temporal readout.
        accuracy = balanced_accuracy(probability[retain], target[retain]) if supported else None
        candidates.append({
            "threshold": float(threshold), "coverage": coverage, "retained_per_class": per_class,
            "balanced_accuracy": accuracy,
            "eligible": bool(supported and coverage >= float(minimum_coverage) and float(accuracy) >= float(minimum_balanced_accuracy)),
        })
    authorized = [candidate for candidate in candidates if candidate["eligible"]]
    if not authorized:
        return {
            "authorized": False,
            "reason": "no fixed signed-pair confidence threshold has sufficient calibration support and quality",
            "threshold": None,
            "candidates": candidates,
        }
    # After calibration has established adequate support and quality, freeze
    # the most conservative threshold.  Coverage is used only to break a tie,
    # never to select the unconditional (threshold-zero) policy by default.
    chosen = max(
        authorized,
        key=lambda candidate: (float(candidate["threshold"]), float(candidate["balanced_accuracy"]), float(candidate["coverage"])),
    )
    return {
        "authorized": True,
        "reason": "calibration quality/support gate passed",
        "threshold": float(chosen["threshold"]),
        "candidates": candidates,
    }


def apply_signed_pair_release_gate(probability: np.ndarray, gate: dict[str, Any]) -> np.ndarray:
    """Release only decisions approved by a previously frozen calibration gate."""

    probability = np.asarray(probability, dtype=np.float64)
    if not bool(gate.get("authorized", False)):
        return np.zeros(len(probability), dtype=bool)
    threshold = gate.get("threshold")
    if threshold is None:
        raise ValueError("authorized signed-pair gate requires a threshold")
    return signed_pair_confidence(probability) >= float(threshold)


def signed_pair_calibration_and_test_probabilities(
    train: list[RootExample], calibration: list[RootExample], test: list[RootExample], *, image_size: int,
    device: str, epochs: int, learning_rate: float, weight_decay: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get calibration/test probabilities from the same train-only frozen fit."""

    train_x = feature_matrix(train, "temporal_history", int(image_size))
    evaluation_x = np.concatenate((
        feature_matrix(calibration, "temporal_history", int(image_size)),
        feature_matrix(test, "temporal_history", int(image_size)),
    ), axis=0)
    probability, _ = fit_linear_readout(
        train_x, labels(train), evaluation_x, device=device, epochs=int(epochs),
        learning_rate=float(learning_rate), weight_decay=float(weight_decay), seed=int(seed),
    )
    return probability[: len(calibration)], probability[len(calibration):]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train-seed-band", type=int, default=1_702_210_000)
    parser.add_argument("--calibration-seed-band", type=int, default=1_702_220_000)
    parser.add_argument("--test-seed-band", type=int, default=1_702_230_000)
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--calibration-per-class", type=int, default=32)
    parser.add_argument("--test-per-class", type=int, default=64)
    parser.add_argument("--pool-transitions", type=int, default=8192)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--history-magnitude", type=float, default=0.30)
    parser.add_argument("--max-contact-gap", type=float, default=65.0)
    parser.add_argument("--minimum-preference-margin", type=float, default=2.0e-4)
    # These defaults are the exact boundary-frame candidate bank authorized by
    # the matched preflight launcher.  The selected pair is its largest
    # contact-preserving mirror, not a separately tuned diagnostic action.
    parser.add_argument("--contact-inward-magnitude", type=float, default=0.08)
    parser.add_argument("--contact-tangent-magnitude", type=float, default=0.16)
    parser.add_argument("--deployed-tangent-magnitudes", type=float, nargs="+", default=[0.06, 0.10, 0.16])
    parser.add_argument("--deployed-contact-normal-magnitudes", type=float, nargs="+", default=[0.02, 0.04, 0.08])
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-ci-lower", type=float, default=0.60)
    parser.add_argument("--minimum-gate-calibration-ba", type=float, default=0.70)
    parser.add_argument("--minimum-gate-coverage", type=float, default=0.25)
    parser.add_argument("--minimum-gate-per-class", type=int, default=8)
    return parser.parse_args()


def candidate_bank_action_pair(
    env: gym.Env,
    state: np.ndarray,
    *,
    inward_magnitude: float,
    tangent_magnitude: float,
    deployed_tangent_magnitudes: tuple[float, ...] = (0.06, 0.10, 0.16),
    deployed_contact_normal_magnitudes: tuple[float, ...] = (0.02, 0.04, 0.08),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract one equal-norm opposite-moment pair from the deployed bank.

    The output names are canonical *geometric moment* signs, never the
    implementation's tangent enumeration order.  A changed candidate builder,
    missing mirror, or mismatched normal component rejects the root before a
    label can be formed.
    """

    state = np.asarray(state, dtype=np.float64)
    normal = np.asarray(boundary_inward_normal(env, state), dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    if not any(np.isclose(float(value), float(inward_magnitude), atol=1.0e-8) for value in deployed_contact_normal_magnitudes):
        raise ValueError("selected inward magnitude must be a member of the deployed candidate bank")
    if not any(np.isclose(float(value), float(tangent_magnitude), atol=1.0e-8) for value in deployed_tangent_magnitudes):
        raise ValueError("selected tangent magnitude must be a member of the deployed candidate bank")
    bank = np.asarray(
        candidate_actions(
            env, state, tuple(float(value) for value in deployed_tangent_magnitudes), float(inward_magnitude),
            boundary_action_frame=True,
            contact_normal_magnitudes=tuple(float(value) for value in deployed_contact_normal_magnitudes),
        ),
        dtype=np.float64,
    )
    matching = [
        int(index)
        for index, action in enumerate(bank)
        if np.isclose(float(np.dot(action, normal)), float(inward_magnitude), atol=1.0e-6)
        and np.isclose(abs(float(np.dot(action, tangent))), float(tangent_magnitude), atol=1.0e-6)
    ]
    audit: dict[str, Any] = {
        "family": "deployed_boundary_candidate_bank",
        "candidate_bank_size": int(len(bank)),
        "matching_candidate_indices": matching,
        "boundary_action_frame": True,
        "selected_pair": {"inward_magnitude": float(inward_magnitude), "tangent_magnitude": float(tangent_magnitude)},
        "deployed_tangent_magnitudes": [float(value) for value in deployed_tangent_magnitudes],
        "deployed_contact_normal_magnitudes": [float(value) for value in deployed_contact_normal_magnitudes],
    }
    if len(matching) != 2:
        audit["pair_valid"] = False
        return {}, audit
    first, second = (bank[index] for index in matching)
    first_sign = float(action_moment_sign(state, first, env=env))
    second_sign = float(action_moment_sign(state, second, env=env))
    if first_sign == second_sign or first_sign == 0.0 or second_sign == 0.0:
        audit["pair_valid"] = False
        audit["candidate_moment_signs"] = [first_sign, second_sign]
        return {}, audit
    positive, negative = (first, second) if first_sign > 0.0 else (second, first)
    normal_difference = float(abs(np.dot(positive - negative, normal)))
    tangent_sum = float(abs(np.dot(positive + negative - 2.0 * float(inward_magnitude) * normal, tangent)))
    norm_difference = float(abs(np.linalg.norm(positive) - np.linalg.norm(negative)))
    audit.update({
        "pair_valid": bool(normal_difference <= 1.0e-6 and tangent_sum <= 1.0e-6 and norm_difference <= 1.0e-6),
        "positive_moment_sign": float(action_moment_sign(state, positive, env=env)),
        "negative_moment_sign": float(action_moment_sign(state, negative, env=env)),
        "pair_normal_component_difference": normal_difference,
        "pair_tangent_antisymmetry_residual": tangent_sum,
        "pair_norm_difference": norm_difference,
    })
    return {
        "zero": np.zeros(2, dtype=np.float32),
        "plus": np.asarray(positive, dtype=np.float32),
        "minus": np.asarray(negative, dtype=np.float32),
    }, audit


def _collect(env: gym.Env, args: argparse.Namespace, split: str, band: int, per_class: int):
    return collect_band(
        env, split=split, band_seed=int(band), per_class=int(per_class),
        pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps),
        history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap),
        minimum_margin=float(args.minimum_preference_margin), preference_fn=signed_offline_preference,
        probe_action_fn=lambda active_env, state: candidate_bank_action_pair(
            active_env, state, inward_magnitude=float(args.contact_inward_magnitude),
            tangent_magnitude=float(args.contact_tangent_magnitude),
            deployed_tangent_magnitudes=tuple(float(value) for value in args.deployed_tangent_magnitudes),
            deployed_contact_normal_magnitudes=tuple(float(value) for value in args.deployed_contact_normal_magnitudes),
        ),
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable candidate-bank signed-torque output: {args.output}")
    if args.history_steps < 2 or args.pool_transitions < 100:
        raise ValueError("confirmation requires multi-frame history and a substantial fixed root pool")
    if not (
        0.0 < args.history_magnitude <= 1.0 and 0.0 < args.minimum_preference_margin
        and 0.0 < args.contact_inward_magnitude <= 1.0 and 0.0 < args.contact_tangent_magnitude <= 1.0
    ):
        raise ValueError("history, pulse magnitudes, and label margin must be positive and bounded")
    if any(not 0.0 < float(value) <= 1.0 for value in args.deployed_tangent_magnitudes + args.deployed_contact_normal_magnitudes):
        raise ValueError("all deployed candidate-bank magnitudes must be positive and bounded")
    if not (0.5 < args.minimum_balanced_accuracy <= 1.0 and 0.5 <= args.minimum_ci_lower <= 1.0):
        raise ValueError("confirmation thresholds must lie above chance and no more than one")
    if not (0.5 < args.minimum_gate_calibration_ba <= 1.0 and 0.0 < args.minimum_gate_coverage <= 1.0 and args.minimum_gate_per_class >= 2):
        raise ValueError("signed-pair calibration gate thresholds are invalid")
    bands = {"train": args.train_seed_band, "calibration": args.calibration_seed_band, "test": args.test_seed_band}
    validate_seed_bands(bands)
    runtime = require_cuda_allocation(args.device)
    started = time.perf_counter()
    collection: dict[str, Any] = {}
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=128)
    try:
        train, train_info = _collect(env, args, "train", bands["train"], args.train_per_class)
        collection["train"] = train_info
        calibration, calibration_info = _collect(env, args, "calibration", bands["calibration"], args.calibration_per_class)
        collection["calibration"] = calibration_info
        test, test_info = _collect(env, args, "test", bands["test"], args.test_per_class)
        collection["test"] = test_info
    except BalancedCohortUnavailable as error:
        collection[str(error.collection.get("split", "unknown"))] = error.collection
        write_pre_outcome_rejection(
            args.output, runtime=runtime, bands=bands, collection=collection, error=error, started=started,
            schema=SCHEMA,
        )
        return
    finally:
        env.close()
    assert_root_separation(train, calibration, test)
    two_frame, raw_probability = evaluate(
        train, calibration, test, representation="two_frame_raw", image_size=int(args.image_size), device=args.device,
        epochs=int(args.epochs), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay),
        bootstrap=int(args.bootstrap), seed=int(bands["train"] + 1),
    )
    temporal, temporal_probability = evaluate(
        train, calibration, test, representation="temporal_history", image_size=int(args.image_size), device=args.device,
        epochs=int(args.epochs), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay),
        bootstrap=int(args.bootstrap), seed=int(bands["train"] + 2),
    )
    delta = paired_bootstrap_delta(
        raw_probability, temporal_probability, labels(test), draws=int(args.bootstrap),
        seed=int(bands["train"] + 999), device=args.device,
    )
    decision = confirmation_rule(
        temporal, two_frame, delta, minimum_ba=float(args.minimum_balanced_accuracy),
        minimum_ci_lower=float(args.minimum_ci_lower),
    )
    temporal_calibration_probability, temporal_gate_test_probability = signed_pair_calibration_and_test_probabilities(
        train, calibration, test, image_size=int(args.image_size), device=args.device, epochs=int(args.epochs),
        learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay), seed=int(bands["train"] + 2),
    )
    if not np.array_equal(temporal_probability, temporal_gate_test_probability):
        raise RuntimeError("signed-pair gate test probabilities differ from the frozen temporal readout")
    gate = select_signed_pair_release_gate(
        temporal_calibration_probability, labels(calibration),
        minimum_balanced_accuracy=float(args.minimum_gate_calibration_ba),
        minimum_coverage=float(args.minimum_gate_coverage), minimum_per_class=int(args.minimum_gate_per_class),
    )
    task_ranking = {
        "two_frame_raw": signed_pair_task_ranking(raw_probability, test),
        "temporal_history": signed_pair_task_ranking(temporal_probability, test),
        "temporal_history_calibrated_release": signed_pair_task_ranking(
            temporal_probability, test, release_mask=apply_signed_pair_release_gate(temporal_probability, gate),
        ),
    }
    report = {
        "schema": SCHEMA,
        "runtime": runtime,
        "protocol": {
            "environment": "swm/PushT-v1",
            "mechanism": "signed zero-drift-corrected response contrast of opposite-moment pulses extracted from the deployed boundary-frame candidate bank",
            "candidate_bank": {
                "boundary_action_frame": True,
                "tangent_magnitudes": [float(value) for value in args.deployed_tangent_magnitudes],
                "contact_normal_magnitudes": [float(value) for value in args.deployed_contact_normal_magnitudes],
                "selected_pair_inward_magnitude": float(args.contact_inward_magnitude),
                "selected_pair_tangent_magnitude": float(args.contact_tangent_magnitude),
            },
            "runtime_inputs": "rendered RGB history and declared history actions only",
            "offline_only": "source reset and zero/positive-moment/negative-moment successors create labels; no successor is a model input",
            "split_protocol": "three disjoint fresh high root-seed bands; labels are balanced before any decoder fit",
            "feature_comparison": ["two_frame_raw", "temporal_history"],
            "decoder": "same train-root-only standardized linear binary readout, fixed threshold 0.5",
            "selective_release": "confidence gate is frozen on calibration roots; test roots can be released or abstained but never refit",
        },
        "seed_bands": bands,
        "collection": collection,
        "sanity_checks": {
            "root_disjoint": True,
            "source_state_digest_disjoint": True,
            "all_labels_collected_before_fitting": True,
            "held_test_is_balanced_before_training": True,
            "signed_label_preserves_geometric_moment_direction": True,
            "pair_is_extracted_from_deployed_candidate_bank": True,
            "counterfactual_reset_seed_shared_across_zero_plus_minus": True,
            "no_runtime_action_selection": True,
        },
        "results": {
            "two_frame_raw": two_frame,
            "temporal_history": temporal,
            "paired_test_delta": delta,
            "signed_pair_task_ranking": task_ranking,
            "signed_pair_release_gate": gate,
        },
        "confirmation": decision,
        "claim_boundary": "An offline signed torque-response identifiability test on the deployed candidate-bank action semantics. It does not select or execute actions online and does not establish planning, pose closure, causal dynamics, or a universal world model.",
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True)
    (args.output / "candidate_bank_signed_temporal_torque_confirmation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "confirmation": decision, "wall_seconds": report["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
