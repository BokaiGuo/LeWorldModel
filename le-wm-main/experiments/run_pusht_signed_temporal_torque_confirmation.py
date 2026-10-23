"""Fresh RGB confirmation for signed PushT torque-response identifiability.

The prior balanced panel asked which mirrored pulse caused the larger
*absolute* angular response.  That erases torque direction and cannot by
itself support angle closure.  This protocol instead labels which pulse has
the more positive zero-drift-corrected angular response.  It tests a necessary
offline capability for directional orientation control, not a planner.
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
    TORQUE_DURATION,
    TORQUE_MAGNITUDE,
    assert_root_separation,
    collect_band,
    confirmation_rule,
    evaluate,
    labels,
    paired_bootstrap_delta,
    signed_offline_preference,
    validate_seed_bands,
)
from experiments.audit_pusht_orientation_response import edge_chart, matched_actions


SCHEMA = "pusht.signed_temporal_torque_confirmation.v2_contact_frame"
SIGNED_SEED_BANDS = {
    "train": 1_701_210_000,
    "calibration": 1_701_220_000,
    "test": 1_701_230_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train-seed-band", type=int, default=SIGNED_SEED_BANDS["train"])
    parser.add_argument("--calibration-seed-band", type=int, default=SIGNED_SEED_BANDS["calibration"])
    parser.add_argument("--test-seed-band", type=int, default=SIGNED_SEED_BANDS["test"])
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--calibration-per-class", type=int, default=32)
    parser.add_argument("--test-per-class", type=int, default=64)
    parser.add_argument("--pool-transitions", type=int, default=8192)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--history-magnitude", type=float, default=0.30)
    parser.add_argument("--max-contact-gap", type=float, default=65.0)
    parser.add_argument("--minimum-preference-margin", type=float, default=2.0e-4)
    parser.add_argument("--contact-inward-magnitude", type=float, default=0.18)
    parser.add_argument("--contact-tangent-magnitude", type=float, default=TORQUE_MAGNITUDE)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-ci-lower", type=float, default=0.60)
    return parser.parse_args()


def contact_frame_action_pair(
    chart: dict[str, Any], *, inward_magnitude: float, tangent_magnitude: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct equal-norm, opposite-moment actions in the actual contact frame.

    ``matched_actions`` defines its algebra from the block-centred contact
    radius.  We rename its pair by its *geometric moment* so ``plus`` always
    means the positive-moment pulse, independent of world orientation or
    which asymmetric-T edge is touched.
    """

    matched = matched_actions(chart, float(inward_magnitude), float(tangent_magnitude))
    if not bool(matched["pair_valid"]):
        return {}, {"family": "contact_frame_radial_tangent", "pair_valid": False}
    first_name, second_name = ("plus", "minus") if float(matched["moment_plus"]) > 0.0 else ("minus", "plus")
    positive = np.asarray(matched[first_name], dtype=np.float32)
    negative = np.asarray(matched[second_name], dtype=np.float32)
    contact_radius = np.asarray(chart["contact_radius"], dtype=np.float64)
    positive_moment = float(contact_radius[0] * positive[1] - contact_radius[1] * positive[0])
    negative_moment = float(contact_radius[0] * negative[1] - contact_radius[1] * negative[0])
    norm_difference = float(abs(np.linalg.norm(positive) - np.linalg.norm(negative)))
    audit = {
        "family": "contact_frame_radial_tangent",
        "pair_valid": bool(
            norm_difference <= 1.0e-6
            and positive_moment > 1.0e-8
            and negative_moment < -1.0e-8
        ),
        "positive_moment": positive_moment,
        "negative_moment": negative_moment,
        "pair_norm_difference": norm_difference,
    }
    return {"zero": np.zeros(2, dtype=np.float32), "plus": positive, "minus": negative}, audit


def contact_frame_probe_actions(
    env: gym.Env, state: np.ndarray, *, inward_magnitude: float, tangent_magnitude: float, max_contact_gap: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a contact-relative pulse pair; this geometry remains label-only."""

    chart = edge_chart(env, np.asarray(state, dtype=np.float64))
    actions, audit = contact_frame_action_pair(
        chart,
        inward_magnitude=float(inward_magnitude), tangent_magnitude=float(tangent_magnitude),
    )
    contact_gap = float(chart["contact_gap"])
    audit["contact_gap"] = contact_gap
    audit["pair_valid"] = bool(audit["pair_valid"] and contact_gap <= float(max_contact_gap))
    return actions, audit


def _collect(env: gym.Env, args: argparse.Namespace, split: str, band: int, per_class: int):
    return collect_band(
        env, split=split, band_seed=int(band), per_class=int(per_class),
        pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps),
        history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap),
        minimum_margin=float(args.minimum_preference_margin), preference_fn=signed_offline_preference,
        probe_action_fn=lambda active_env, state: contact_frame_probe_actions(
            active_env, state, inward_magnitude=float(args.contact_inward_magnitude),
            tangent_magnitude=float(args.contact_tangent_magnitude),
            max_contact_gap=float(args.max_contact_gap),
        ),
    )


def write_pre_outcome_rejection(
    output: Path, *, runtime: dict[str, Any], bands: dict[str, int], collection: dict[str, Any], error: Exception,
    started: float, schema: str = SCHEMA,
) -> None:
    """Persist an immutable failed-cohort receipt without fitting or scoring."""

    output.mkdir(parents=True)
    report = {
        "schema": str(schema),
        "runtime": runtime,
        "seed_bands": bands,
        "collection": collection,
        "status": "rejected_pre_outcome_balanced_cohort",
        "rejection_reason": str(error),
        "sanity_checks": {
            "decoder_fit_attempted": False,
            "calibration_or_test_labels_collected_after_rejection": False,
            "outcome_score_materialized": False,
            "counterfactual_reset_contract": "zero/plus/minus share one reset seed per source root",
        },
        "claim_boundary": (
            "A pre-outcome cohort-availability rejection. It is neither a decoder result nor evidence about "
            "signed torque identifiability, planning, pose closure, causal dynamics, or a universal world model."
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "signed_temporal_torque_confirmation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "status": report["status"], "rejection_reason": str(error)}, indent=2))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable signed-torque output: {args.output}")
    if args.history_steps < 2 or args.pool_transitions < 100:
        raise ValueError("confirmation requires multi-frame history and a substantial fixed root pool")
    if not (
        0.0 < args.history_magnitude <= 1.0
        and 0.0 < args.minimum_preference_margin
        and 0.0 < args.contact_inward_magnitude <= 1.0
        and 0.0 < args.contact_tangent_magnitude <= 1.0
    ):
        raise ValueError("history, contact-frame pulse magnitudes, and label margin must be positive and bounded")
    if not (0.5 < args.minimum_balanced_accuracy <= 1.0 and 0.5 <= args.minimum_ci_lower <= 1.0):
        raise ValueError("confirmation thresholds must lie above chance and no more than one")
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
    report: dict[str, Any] = {
        "schema": SCHEMA, "runtime": runtime,
        "protocol": {
            "environment": "swm/PushT-v1",
            "mechanism": "signed zero-drift-corrected response contrast of positive- versus negative-moment contact-frame pulses",
            "mirrored_pulse": {
                "frame": "block-centred contact radius",
                "common_component": "radial inward",
                "opposed_component": "tangent about the contact radius",
                "inward_magnitude": float(args.contact_inward_magnitude),
                "tangent_magnitude": float(args.contact_tangent_magnitude),
                "duration_steps": TORQUE_DURATION,
            },
            "runtime_inputs": "rendered RGB history and declared history actions only",
            "offline_only": "source reset and zero/positive-moment/negative-moment successors create labels; no successor is a model input",
            "split_protocol": "three disjoint fresh high root-seed bands; labels are balanced before any decoder fit",
            "feature_comparison": ["two_frame_raw", "temporal_history"],
            "decoder": "same train-root-only standardized linear binary readout, fixed threshold 0.5",
        },
        "seed_bands": bands,
        "collection": collection,
        "sanity_checks": {
            "root_disjoint": True, "source_state_digest_disjoint": True,
            "all_labels_collected_before_fitting": True, "held_test_is_balanced_before_training": True,
            "signed_label_preserves_torque_direction": True, "no_runtime_action_selection": True,
            "contact_frame_pair_has_equal_norm_and_opposite_geometric_moment": True,
            "counterfactual_reset_seed_shared_across_zero_plus_minus": True,
        },
        "results": {"two_frame_raw": two_frame, "temporal_history": temporal, "paired_test_delta": delta},
        "confirmation": decision,
        "claim_boundary": "An offline signed torque-response identifiability test. It does not select or execute actions online and does not establish planning, pose closure, causal dynamics, or a universal world model.",
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True)
    (args.output / "signed_temporal_torque_confirmation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "confirmation": decision, "wall_seconds": report["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
