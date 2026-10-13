"""Replay a saved RGB local-effect adapter under planning-only interventions.

This runner deliberately does not fit any readout.  It restores the frozen
base model and every root-disjoint readout from a completed adapter artifact,
then replays the terminal planning protocol while changing only explicitly
declared runtime action constraints.  It is therefore suitable for isolating
terminal-action interface failures without silently changing the training
split, ranker, or model-selection gate.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from run_rgb_local_effect_adapter import (
    ActionCostRanker,
    GoalPoseDecoder,
    LinearFeatureXYDecoder,
    LinearRGBPoseCalibrator,
    LocalEffectAdapter,
    OrientationDecoder,
    balanced_source_cases,
    rgb_shape_raster_features,
    terminal_pose_cases,
    train_rgb_angle_knn,
    run_planning,
    load_base,
    validate_terminal_ranker_contract,
)


def _torch_module(cls: Any, state: dict[str, Any], input_key: str) -> Any:
    input_dim = int(state["net.0.weight"].shape[1])
    hidden_dim = int(state["net.0.weight"].shape[0])
    module = cls(input_dim, hidden_dim)
    module.load_state_dict(state)
    return module.eval()


def _rgb_calibrator(state: dict[str, Any] | None) -> LinearRGBPoseCalibrator | None:
    """Restore an optional affine RGB pose readout.

    Artifacts deliberately store ``None`` when the root-disjoint validation
    gate rejects the RGB calibrator.  Replay must preserve that fail-closed
    decision and let ``run_planning`` use the declared non-RGB interface,
    rather than treating an absent optional readout as a malformed artifact.
    """
    if state is None:
        return None
    return LinearRGBPoseCalibrator(
        np.asarray(state["coefficients"], dtype=np.float32),
        np.asarray(state["feature_mean"], dtype=np.float32),
        np.asarray(state["feature_std"], dtype=np.float32),
    ).eval()


def _load_replay_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"adapter", "adapter_training", "goal_decoder", "goal_decoder_training"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"replay artifact is missing keys: {sorted(missing)}")
    return payload


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": rows,
        "strict_success_count": int(sum(bool(row["strict_success"]) for row in rows)),
        "strict_success_rate": float(np.mean([row["strict_success"] for row in rows])) if rows else None,
        "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in rows])) if rows else None,
        "mean_angle_error": float(np.mean([row["angle_error"] for row in rows])) if rows else None,
        "terminal_tube_entry_rate": float(np.mean([
            any(float(item.get("block_xy", np.inf)) <= 24.0 for item in row["trace"])
            for row in rows
        ])) if rows else None,
        "torque_phase_entry_rate": float(np.mean([
            any(item.get("phase") in {"torque", "hold"} for item in row["trace"])
            for row in rows
        ])) if rows else None,
        "contact_retention_rate": float(np.mean([
            float(item.get("agent_block_distance", np.inf)) <= 100.0
            for row in rows for item in row["trace"]
        ])) if rows and any(row["trace"] for row in rows) else None,
        "reacquisition_attempt_count": int(sum(
            float(item.get("reacquisition_attempt", 0.0)) > 0.0
            for row in rows for item in row["trace"]
        )),
        "reacquisition_success_count": int(sum(
            float(item.get("reacquisition_success", 0.0)) > 0.0
            for row in rows for item in row["trace"]
        )),
        "reacquisition_route_abort_count": int(sum(
            float(item.get("reacquisition_route_abort", 0.0)) > 0.0
            for row in rows for item in row["trace"]
        )),
        "reacquisition_route_lockout_count": int(sum(
            float(item.get("reacquisition_route_lockout", 0.0)) > 0.0
            for row in rows for item in row["trace"]
        )),
        "reacquisition_bounded_translation_repair_count": int(sum(
            float(item.get("reacquisition_bounded_translation_repair", 0.0)) > 0.0
            for row in rows for item in row["trace"]
        )),
        "reacquisition_hold_action_count": int(sum(
            item.get("phase") == "hold"
            for row in rows for item in row["trace"]
        )),
        "agent_coordinate_sentinel_count": int(sum(
            np.isfinite(np.asarray(item.get("audit_observed_agent_xy", [np.nan, np.nan]), dtype=np.float32)).all()
            and float(np.linalg.norm(
                np.asarray(item.get("audit_observed_agent_xy", [np.nan, np.nan]), dtype=np.float32) - 256.0
            )) <= 1.0e-3
            for row in rows for item in row["trace"]
        )),
        "terminal_action_norm_mean": float(np.mean([
            np.linalg.norm(np.asarray(item["action"], dtype=np.float32))
            for row in rows for item in row["trace"]
            if item.get("phase") in {"torque", "hold"}
        ])) if any(item.get("phase") in {"torque", "hold"} for row in rows for item in row["trace"]) else None,
        "terminal_block_motion_mean": float(np.mean([
            float(item.get("audit_physical_block_motion", np.nan))
            for row in rows for item in row["trace"]
            if item.get("phase") in {"torque", "hold"}
            and np.isfinite(float(item.get("audit_physical_block_motion", np.nan)))
        ])) if any(
            item.get("phase") in {"torque", "hold"}
            and np.isfinite(float(item.get("audit_physical_block_motion", np.nan)))
            for row in rows for item in row["trace"]
        ) else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-action-max-magnitude", type=float, required=True)
    parser.add_argument("--terminal-contact-inward-min", type=float, default=0.0)
    parser.add_argument("--terminal-contact-tangent-ratio", type=float, default=0.0)
    parser.add_argument("--terminal-contact-lease-gap-max", type=float, default=None)
    parser.add_argument(
        "--support-contact-side-agnostic-terminal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable the declared support-gap-only lease for terminal-root closure replay",
    )
    parser.add_argument(
        "--terminal-proximity-lease-fallback", action=argparse.BooleanOptionalAction, default=None,
        help="override the RGB near-contact fallback that bypasses a transient boundary-gap approach",
    )
    parser.add_argument(
        "--terminal-contact-gap-max", type=float, default=None,
        help="override the RGB support-gap threshold required before terminal torque entry",
    )
    parser.add_argument("--terminal-agent-distance-max", type=float, default=None)
    parser.add_argument("--terminal-angle-progress-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rgb-angle-disagreement-policy", choices=("none", "latent_fallback"), default=None,
        help="override the declared RGB-vs-latent current-angle disagreement gate for an audited replay",
    )
    parser.add_argument(
        "--rgb-angle-disagreement-terminal-xy", type=float, default=None,
        help="terminal XY radius within which the disagreement gate may fall back to the latent angle",
    )
    parser.add_argument(
        "--use-rgb-current-angle-runtime", action=argparse.BooleanOptionalAction, default=None,
        help="override whether the artifact's gated RGB current-angle readout is used during replay",
    )
    parser.add_argument(
        "--disable-rgb-goal-angle", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "disable the serialized RGB goal-angle calibrator during replay; "
            "useful for backward-compatible artifacts whose calibration feature "
            "interface predates the current runtime"
        ),
    )
    parser.add_argument(
        "--disable-rgb-goal-xy", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "disable the serialized RGB goal-XY calibrator during replay; "
            "use the frozen goal decoder when replaying older artifacts"
        ),
    )
    parser.add_argument(
        "--rgb-temporal-angle-runtime", action=argparse.BooleanOptionalAction, default=False,
        help="enable the RGB-only temporal silhouette angle observer as an audited replay intervention",
    )
    parser.add_argument(
        "--rgb-temporal-angle-local-effect", action=argparse.BooleanOptionalAction, default=False,
        help="also feed the temporal increment into online local-effect identification; disabled keeps audit-only behavior",
    )
    parser.add_argument(
        "--rgb-temporal-angle-pulse", action=argparse.BooleanOptionalAction, default=False,
        help="also use the temporal increment in pulse confirmation; disabled keeps pulse decisions on the legacy absolute readout",
    )
    parser.add_argument("--rgb-temporal-angle-max-delta", type=float, default=None)
    parser.add_argument("--rgb-temporal-angle-min-confidence", type=float, default=None)
    parser.add_argument("--torque-sign-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--terminal-ranker-blend", type=float, default=None,
        help="mix normalized terminal-ranker scores with normalized frozen-adapter pose scores; 1 preserves ranker-only replay",
    )
    parser.add_argument(
        "--forbid-zero-before-terminal", action=argparse.BooleanOptionalAction, default=None,
        help="override the terminal zero-action guard; zero remains allowed after visual pose closure",
    )
    parser.add_argument(
        "--force-terminal-ranker", action=argparse.BooleanOptionalAction, default=False,
        help="audit-only override to replay a stored terminal ranker rejected by its held-out gate",
    )
    parser.add_argument(
        "--force-rgb-current-calibrator", action=argparse.BooleanOptionalAction, default=False,
        help="audit-only override to replay the stored dynamic RGB current-XY calibrator",
    )
    parser.add_argument(
        "--contact-action-projection", action=argparse.BooleanOptionalAction, default=None,
        help="override the final RGB contact-manifold projection",
    )
    parser.add_argument("--contact-translation-inward-min", type=float, default=None)
    parser.add_argument("--contact-translation-tangent-ratio", type=float, default=None)
    parser.add_argument("--contact-translation-progress-min", type=float, default=None)
    parser.add_argument("--terminal-motion-damping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminal-motion-damping-motion", type=float, default=None)
    parser.add_argument("--terminal-motion-damping-regression", type=float, default=None)
    parser.add_argument(
        "--approach-action-max-magnitude", type=float, default=None,
        help=(
            "optional RGB-only L2 cap for approach actions during replay; "
            "unset preserves the stored protocol"
        ),
    )
    parser.add_argument(
        "--hybrid-translation-action-max-magnitude", type=float, default=None,
        help=(
            "optional L2 cap for geometry translation actions selected by the "
            "hybrid replay; unset preserves the stored protocol"
        ),
    )
    parser.add_argument(
        "--reacquisition-post-success-translation-cap", type=float, default=None,
        help=(
            "optional temporary geometry-translation cap after RGB "
            "reacquisition success; unset preserves the stored protocol"
        ),
    )
    parser.add_argument("--terminal-pulse-confirmation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--terminal-pulse-min-angle-gain", type=float, default=0.0)
    parser.add_argument("--terminal-pulse-max-angle-regression", type=float, default=0.08)
    parser.add_argument("--terminal-pulse-max-xy-regression", type=float, default=2.0)
    parser.add_argument("--terminal-pulse-fallback-magnitude", type=float, default=0.06)
    parser.add_argument("--terminal-pulse-freeze-steps", type=int, default=1)
    parser.add_argument("--terminal-pulse-max-failures", type=int, default=2)
    parser.add_argument("--geometric-torque-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--geometric-torque-support-lever", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--geometric-torque-raw-rgb-angle", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--geometric-torque-magnitude", type=float, default=None)
    parser.add_argument("--geometric-torque-inward-magnitude", type=float, default=None)
    parser.add_argument("--geometric-torque-adaptive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--geometric-torque-motion-feedback", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--geometric-torque-drift-abort", action=argparse.BooleanOptionalAction, default=None,
        help="switch back to geometry translation when observed torque motion exceeds the declared drift gate",
    )
    parser.add_argument("--planning-episodes", type=int, default=None)
    parser.add_argument("--planning-steps", type=int, default=None)
    parser.add_argument("--planning-start", choices=("random", "terminal"), default=None)
    parser.add_argument(
        "--terminal-start-phase", choices=("push", "approach"), default=None,
        help=(
            "terminal-root state-machine entry; push preserves the historical "
            "replay, while approach requires an RGB contact handoff before "
            "terminal ranking"
        ),
    )
    parser.add_argument(
        "--planning-controller", choices=("adapter", "hybrid"), default=None,
        help=(
            "replay-only controller branch override: adapter uses the learned "
            "translation selector, while hybrid uses RGB geometry for translation "
            "and preserves the stored learned terminal interface"
        ),
    )
    parser.add_argument("--online-local-angle-weight", type=float, default=None)
    parser.add_argument("--terminal-local-angle-weight", type=float, default=None)
    parser.add_argument("--online-local-min-samples", type=int, default=None)
    parser.add_argument("--online-local-window", type=int, default=None)
    parser.add_argument("--side-recovery-mode", choices=("contact_target", "retreat"), default=None)
    parser.add_argument("--terminal-side-reacquisition", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--online-signed-torque-probe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--signed-torque-probe-xy-repair", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--signed-torque-probe-max-pulses", type=int, default=None)
    parser.add_argument(
        "--signed-torque-probe-max-magnitude", type=float, default=None,
        help="independent norm cap for the online signed-torque probe; does not change ordinary terminal actions",
    )
    parser.add_argument(
        "--terminal-action-commitment", type=int, default=None,
        help="reuse a selected terminal ranker action for this many decisions; 1 preserves per-frame replanning",
    )
    parser.add_argument(
        "--terminal-torque-prior-mode",
        choices=("none", "prior", "combined", "ranker_prior"),
        default="none",
        help=(
            "audit-only RGB support-lever prior: prior replaces the learned "
            "selector, combined mixes with the legacy baseline, and "
            "ranker_prior mixes with the restored learned terminal ranker"
        ),
    )
    parser.add_argument("--terminal-torque-prior-weight", type=float, default=None)
    parser.add_argument("--terminal-torque-prior-blend", type=float, default=None)
    parser.add_argument(
        "--recovery-action-max-magnitude", type=float, default=None,
        help="override the ordinary RGB recovery true L2 cap; unset preserves the artifact protocol",
    )
    parser.add_argument(
        "--reacquisition-action-max-magnitude", type=float, default=None,
        help="override the RGB side-route action cap; unset preserves the artifact protocol",
    )
    parser.add_argument(
        "--reacquisition-max-retries", type=int, default=None,
        help="override the bounded fresh-anchor retry budget after a route abort",
    )
    parser.add_argument(
        "--reacquisition-lockout-steps", type=int, default=None,
        help="override the bounded post-abort lockout; zero permits immediate fresh-anchor recovery",
    )
    parser.add_argument(
        "--reacquisition-failure-translation-cap", type=float, default=None,
        help="override the one-step bounded local repair cap after a route abort",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--runtime-current-xy-policy", choices=(
            "validation", "rgb_mask", "rgb_mask_fused", "rgb_calibrated",
            "paired_rgb_calibrated",
        ),
        default="paired_rgb_calibrated",
        help="observer to use for replay; paired RGB reconstructs the missing current-angle KNN support",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    artifact = cli.adapter_artifact.resolve()
    report_path = artifact.parent / "rgb_local_effect_adapter.json"
    report = json.loads(report_path.read_text())
    payload = _load_replay_artifact(artifact)
    protocol = dict(report["protocol"])
    args = Namespace(**protocol)
    args.device = str(cli.device)
    args.output = str(cli.output)
    args.terminal_action_max_magnitude = float(cli.terminal_action_max_magnitude)
    args.terminal_contact_inward_min = float(cli.terminal_contact_inward_min)
    args.terminal_contact_tangent_ratio = float(cli.terminal_contact_tangent_ratio)
    if cli.terminal_contact_lease_gap_max is not None:
        args.terminal_contact_lease_gap_max = float(cli.terminal_contact_lease_gap_max)
    if cli.terminal_proximity_lease_fallback is not None:
        args.terminal_proximity_lease_fallback = bool(cli.terminal_proximity_lease_fallback)
    if cli.terminal_contact_gap_max is not None:
        args.terminal_contact_gap_max = float(cli.terminal_contact_gap_max)
    if cli.support_contact_side_agnostic_terminal is not None:
        args.support_contact_side_agnostic_terminal = bool(
            cli.support_contact_side_agnostic_terminal
        )
    if cli.terminal_agent_distance_max is not None:
        args.terminal_agent_distance_max = float(cli.terminal_agent_distance_max)
    args.terminal_angle_progress_guard = bool(cli.terminal_angle_progress_guard)
    if cli.rgb_angle_disagreement_policy is not None:
        args.rgb_angle_disagreement_policy = str(cli.rgb_angle_disagreement_policy)
    if cli.rgb_angle_disagreement_terminal_xy is not None:
        args.rgb_angle_disagreement_terminal_xy = float(cli.rgb_angle_disagreement_terminal_xy)
    args.torque_sign_guard = bool(cli.torque_sign_guard)
    if cli.terminal_ranker_blend is not None:
        args.terminal_ranker_blend = float(cli.terminal_ranker_blend)
    if cli.forbid_zero_before_terminal is not None:
        args.forbid_zero_before_terminal = bool(cli.forbid_zero_before_terminal)
    if cli.contact_action_projection is not None:
        args.contact_action_projection = bool(cli.contact_action_projection)
    if cli.contact_translation_inward_min is not None:
        args.contact_translation_inward_min = float(cli.contact_translation_inward_min)
    if cli.contact_translation_tangent_ratio is not None:
        args.contact_translation_tangent_ratio = float(cli.contact_translation_tangent_ratio)
    if cli.contact_translation_progress_min is not None:
        args.contact_translation_progress_min = float(cli.contact_translation_progress_min)
    if cli.terminal_contact_inward_min is not None:
        args.terminal_contact_inward_min = float(cli.terminal_contact_inward_min)
    if cli.terminal_contact_tangent_ratio is not None:
        args.terminal_contact_tangent_ratio = float(cli.terminal_contact_tangent_ratio)
    args.terminal_motion_damping = bool(cli.terminal_motion_damping)
    if cli.terminal_motion_damping_motion is not None:
        args.terminal_motion_damping_motion = float(cli.terminal_motion_damping_motion)
    if cli.terminal_motion_damping_regression is not None:
        args.terminal_motion_damping_regression = float(cli.terminal_motion_damping_regression)
    if cli.approach_action_max_magnitude is not None:
        args.approach_action_max_magnitude = float(cli.approach_action_max_magnitude)
    if cli.hybrid_translation_action_max_magnitude is not None:
        args.hybrid_translation_action_max_magnitude = float(
            cli.hybrid_translation_action_max_magnitude
        )
    if cli.reacquisition_post_success_translation_cap is not None:
        args.reacquisition_post_success_translation_cap = float(
            cli.reacquisition_post_success_translation_cap
        )
    args.terminal_pulse_confirmation = bool(cli.terminal_pulse_confirmation)
    args.terminal_pulse_min_angle_gain = float(cli.terminal_pulse_min_angle_gain)
    args.terminal_pulse_max_angle_regression = float(cli.terminal_pulse_max_angle_regression)
    args.terminal_pulse_max_xy_regression = float(cli.terminal_pulse_max_xy_regression)
    args.terminal_pulse_fallback_magnitude = float(cli.terminal_pulse_fallback_magnitude)
    args.terminal_pulse_freeze_steps = int(cli.terminal_pulse_freeze_steps)
    args.terminal_pulse_max_failures = int(cli.terminal_pulse_max_failures)
    args.geometric_torque_fallback = bool(cli.geometric_torque_fallback)
    args.geometric_torque_support_lever = bool(cli.geometric_torque_support_lever)
    if cli.geometric_torque_raw_rgb_angle is not None:
        args.geometric_torque_raw_rgb_angle = bool(cli.geometric_torque_raw_rgb_angle)
    if cli.geometric_torque_magnitude is not None:
        args.geometric_torque_magnitude = float(cli.geometric_torque_magnitude)
    if cli.geometric_torque_inward_magnitude is not None:
        args.geometric_torque_inward_magnitude = float(cli.geometric_torque_inward_magnitude)
    if cli.geometric_torque_adaptive is not None:
        args.geometric_torque_adaptive = bool(cli.geometric_torque_adaptive)
    if cli.geometric_torque_motion_feedback is not None:
        args.geometric_torque_motion_feedback = bool(cli.geometric_torque_motion_feedback)
    if cli.geometric_torque_drift_abort is not None:
        args.geometric_torque_drift_abort = bool(cli.geometric_torque_drift_abort)
    if cli.planning_episodes is not None:
        args.planning_episodes = int(cli.planning_episodes)
    if cli.planning_steps is not None:
        args.planning_steps = int(cli.planning_steps)
    if cli.planning_start is not None:
        args.planning_start = str(cli.planning_start)
    if cli.terminal_start_phase is not None:
        args.terminal_start_phase = str(cli.terminal_start_phase)
    if cli.planning_controller is not None:
        args.planning_controller = str(cli.planning_controller)
    if cli.online_local_angle_weight is not None:
        args.online_local_angle_weight = float(cli.online_local_angle_weight)
    if cli.terminal_local_angle_weight is not None:
        args.terminal_local_angle_weight = float(cli.terminal_local_angle_weight)
    if cli.online_local_min_samples is not None:
        args.online_local_min_samples = int(cli.online_local_min_samples)
    if cli.online_local_window is not None:
        args.online_local_window = int(cli.online_local_window)
    if cli.side_recovery_mode is not None:
        args.side_recovery_mode = str(cli.side_recovery_mode)
    if cli.terminal_side_reacquisition is not None:
        args.terminal_side_reacquisition = bool(cli.terminal_side_reacquisition)
    if cli.online_signed_torque_probe is not None:
        args.online_signed_torque_probe = bool(cli.online_signed_torque_probe)
    if cli.signed_torque_probe_xy_repair is not None:
        args.signed_torque_probe_xy_repair = bool(cli.signed_torque_probe_xy_repair)
    if cli.signed_torque_probe_max_pulses is not None:
        args.signed_torque_probe_max_pulses = int(cli.signed_torque_probe_max_pulses)
    if cli.signed_torque_probe_max_magnitude is not None:
        args.signed_torque_probe_max_magnitude = float(cli.signed_torque_probe_max_magnitude)
    if cli.terminal_action_commitment is not None:
        args.terminal_action_commitment = int(cli.terminal_action_commitment)
    else:
        args.terminal_action_commitment = int(
            getattr(args, "terminal_action_commitment", 1)
        )
    if args.terminal_action_commitment < 1:
        raise ValueError("terminal_action_commitment must be positive")
    args._terminal_torque_prior_runtime_mode = str(cli.terminal_torque_prior_mode)
    if cli.terminal_torque_prior_weight is not None:
        args.terminal_torque_prior_weight = float(cli.terminal_torque_prior_weight)
    if cli.terminal_torque_prior_blend is not None:
        args.terminal_torque_prior_blend = float(cli.terminal_torque_prior_blend)
    if cli.recovery_action_max_magnitude is not None:
        args.recovery_action_max_magnitude = float(cli.recovery_action_max_magnitude)
    if cli.reacquisition_action_max_magnitude is not None:
        args.reacquisition_action_max_magnitude = float(cli.reacquisition_action_max_magnitude)
    if cli.reacquisition_max_retries is not None:
        args.reacquisition_max_retries = int(cli.reacquisition_max_retries)
    if cli.reacquisition_lockout_steps is not None:
        args.reacquisition_lockout_steps = int(cli.reacquisition_lockout_steps)
    if cli.reacquisition_failure_translation_cap is not None:
        args.reacquisition_failure_translation_cap = float(
            cli.reacquisition_failure_translation_cap
        )
    # These are runtime decisions made after the held-out gates in the
    # original training run.  Restore the declared selected interface.
    args._runtime_current_xy_policy = str(cli.runtime_current_xy_policy)
    args._use_rgb_current_angle_runtime = bool(protocol.get("_use_rgb_current_angle_runtime", False))
    args._use_rgb_temporal_angle_runtime = bool(cli.rgb_temporal_angle_runtime)
    args._use_rgb_temporal_angle_local_effect = bool(cli.rgb_temporal_angle_local_effect)
    args._use_rgb_temporal_angle_pulse = bool(cli.rgb_temporal_angle_pulse)
    if cli.rgb_temporal_angle_max_delta is not None:
        args.rgb_temporal_angle_max_delta = float(cli.rgb_temporal_angle_max_delta)
    if cli.rgb_temporal_angle_min_confidence is not None:
        args.rgb_temporal_angle_min_confidence = float(cli.rgb_temporal_angle_min_confidence)
    if cli.use_rgb_current_angle_runtime is not None:
        args._use_rgb_current_angle_runtime = bool(cli.use_rgb_current_angle_runtime)
    args._runtime_last_block_xy = None
    args._force_geometry_translation = False
    args._signed_torque_probe_state = {}
    args._signed_torque_probe_magnitude_scale = 1.0
    args.output = str(cli.output)

    base_checkpoint = Path(payload["base_checkpoint"])
    if not base_checkpoint.is_absolute():
        base_checkpoint = (artifact.parent.parent.parent / base_checkpoint).resolve()
    model, normalizer, _ = load_base(base_checkpoint, args.device)

    adapter = LocalEffectAdapter(int(payload["input_dim"]), int(payload["hidden_dim"]))
    adapter.load_state_dict(payload["adapter"])
    adapter = adapter.to(args.device).eval()
    terminal_angular_ranker = None
    terminal_angular_ranker_training = None
    terminal_metrics = report.get("held_out_terminal_angular_ranker", {})
    if payload.get("terminal_angular_ranker_training") is not None:
        # Validate the serialized objective before restoring the module.  An
        # old artifact with a pose label but an angle score mode is not a
        # usable baseline: replaying it would silently evaluate a different
        # control problem from the held-out gate that promoted it.
        validate_terminal_ranker_contract(
            payload["terminal_angular_ranker_training"],
            args,
            report_metrics=terminal_metrics,
        )
    if (
        bool(terminal_metrics.get("selected_for_runtime", False))
        and (
            payload.get("terminal_angular_ranker") is None
            or payload.get("terminal_angular_ranker_training") is None
        )
        and not bool(cli.force_terminal_ranker)
    ):
        raise RuntimeError(
            "artifact is inconsistent: held-out terminal ranker is selected "
            "but its serialized weights/training state are missing"
        )
    if (
        (bool(terminal_metrics.get("selected_for_runtime", False)) or bool(cli.force_terminal_ranker))
        and payload.get("terminal_angular_ranker") is not None
        and payload.get("terminal_angular_ranker_training") is not None
    ):
        terminal_angular_ranker = _torch_module(
            ActionCostRanker,
            payload["terminal_angular_ranker"],
            "net.0.weight",
        ).to(args.device)
        terminal_angular_ranker_training = payload["terminal_angular_ranker_training"]
    goal_decoder = _torch_module(GoalPoseDecoder, payload["goal_decoder"], "net.0.weight").to(args.device)
    orientation_decoder = _torch_module(OrientationDecoder, payload["orientation_decoder"], "net.0.weight").to(args.device)
    current_xy_decoder = _torch_module(OrientationDecoder, payload["current_xy_decoder"], "net.0.weight").to(args.device)
    rgb_current_xy_decoder = _rgb_calibrator(payload.get("rgb_current_xy_decoder"))
    runtime_xy_decoder = current_xy_decoder
    runtime_xy_training = payload["current_xy_decoder_training"]
    if bool(cli.force_rgb_current_calibrator):
        if rgb_current_xy_decoder is None or payload.get("rgb_current_xy_training") is None:
            raise RuntimeError("audit requested force-rgb-current-calibrator but artifact has no RGB calibrator")
        runtime_xy_decoder = rgb_current_xy_decoder
        runtime_xy_training = payload["rgb_current_xy_training"]
        args._runtime_current_xy_policy = "rgb_calibrated"
    goal_angle_decoder = None
    goal_xy_decoder = None
    rgb_goal_xy_decoder = _rgb_calibrator(payload["rgb_goal_xy_decoder"])
    if bool(cli.disable_rgb_goal_xy):
        rgb_goal_xy_decoder = None
    rgb_goal_angle_decoder = _rgb_calibrator(payload["rgb_goal_angle_decoder"])
    if bool(cli.disable_rgb_goal_angle):
        # Some older artifacts serialized an RGB goal-angle calibrator using a
        # feature width that changed when the shape-raster interface was
        # hardened.  Failing closed to the frozen goal decoder preserves the
        # trained planning interface and makes the compatibility intervention
        # explicit in the replay manifest; it never silently mixes old and
        # new calibration coordinates.
        rgb_goal_angle_decoder = None
    if rgb_goal_xy_decoder is not None and not isinstance(rgb_goal_xy_decoder, LinearRGBPoseCalibrator):
        raise RuntimeError(f"replay RGB goal XY decoder type mismatch: {type(rgb_goal_xy_decoder)!r}")
    agent_xy_decoder = LinearFeatureXYDecoder(
        np.asarray(payload["agent_xy_decoder"]["coefficients"], dtype=np.float32),
        np.asarray(payload["agent_xy_decoder"]["feature_mean"], dtype=np.float32),
        np.asarray(payload["agent_xy_decoder"]["feature_std"], dtype=np.float32),
        np.asarray(payload["agent_xy_decoder"]["target_mean"], dtype=np.float32),
        np.asarray(payload["agent_xy_decoder"]["target_std"], dtype=np.float32),
    ).eval()
    goal_selection = report["goal_component_selection"]
    env = gym.make(
        "swm/PushT-v1", render_mode="rgb_array",
        max_episode_steps=int(args.planning_steps) * int(args.action_repeat),
    )
    try:
        # The original artifact predates serialization of the raster KNN
        # support used by the paired observer. Reconstruct this readout from
        # the same root protocol and labels, without fitting any dynamics or
        # planning component. This restores the runtime interface rather than
        # silently replacing it with a different XY observer.
        rgb_current_angle_decoder = None
        rgb_current_angle_training = None
        if (
            str(cli.runtime_current_xy_policy) == "paired_rgb_calibrated"
            or bool(protocol.get("_use_rgb_current_angle_runtime", False))
        ):
            train_terminal = int(round(int(args.train_roots) * float(args.terminal_root_fraction)))
            roots = balanced_source_cases(
                env,
                int(args.train_roots) - train_terminal,
                int(args.root_rollout_steps),
                int(args.seed),
                float(args.contact_root_fraction),
                int(args.contact_approach_steps),
            )
            roots.extend(terminal_pose_cases(
                env, train_terminal, int(args.seed) + 400_000,
                position_jitter=float(args.terminal_position_jitter),
                support_gap_max=float(getattr(args, "terminal_root_support_gap", 0.0)),
                support_angle_samples=int(getattr(args, "terminal_root_support_angle_samples", 32)),
            ))
            images = np.asarray([root[2] for root in roots], dtype=np.uint8)
            features = rgb_shape_raster_features(images, "block")
            targets = np.asarray([
                [np.sin(float(root[0][4])), np.cos(float(root[0][4]))]
                for root in roots
            ], dtype=np.float32)
            rgb_current_angle_decoder, rgb_current_angle_training = train_rgb_angle_knn(
                features, targets,
            )
            # Raster KNN is a torch module, whereas affine calibration is a
            # NumPy-only readout.  Move only the former to the declared device.
            if hasattr(rgb_current_angle_decoder, "to"):
                rgb_current_angle_decoder = rgb_current_angle_decoder.to(args.device)
            rgb_current_angle_training.update({
                "feature_interface": "raster",
                "feature_role": "block",
                "readout": "support_constrained_knn",
                "reconstructed_for_replay": True,
                "reconstruction_roots": len(roots),
            })
        replay_controller = str(
            cli.planning_controller
            if cli.planning_controller is not None
            else getattr(args, "planning_controller", "hybrid")
        )
        rows = run_planning(
            env, model, normalizer, adapter, payload["adapter_training"], args,
            replay_controller,
            goal_decoder=goal_decoder,
            goal_decoder_training=payload["goal_decoder_training"],
            orientation_decoder=orientation_decoder,
            orientation_decoder_training=payload["orientation_decoder_training"],
            goal_component_selection=goal_selection,
            goal_xy_decoder=goal_xy_decoder,
            goal_xy_decoder_training=payload.get("goal_xy_decoder_training"),
            goal_angle_decoder=goal_angle_decoder,
            goal_angle_decoder_training=payload.get("goal_angle_decoder_training"),
            xy_decoder=runtime_xy_decoder,
            xy_decoder_training=runtime_xy_training,
            action_cost_ranker=None,
            action_cost_ranker_training=None,
            terminal_angular_ranker=terminal_angular_ranker,
            terminal_angular_ranker_training=terminal_angular_ranker_training,
            contact_mode_classifier=None,
            contact_mode_training=None,
            contact_mode_penalty=0.0,
            contact_knn_training=None,
            rgb_current_angle_decoder=rgb_current_angle_decoder,
            rgb_current_angle_training=rgb_current_angle_training,
            rgb_goal_angle_decoder=rgb_goal_angle_decoder,
            rgb_goal_angle_training=payload["rgb_goal_angle_training"],
            rgb_goal_xy_decoder=rgb_goal_xy_decoder,
            rgb_goal_xy_training=payload["rgb_goal_xy_training"],
            agent_xy_decoder=agent_xy_decoder,
            agent_xy_decoder_training=payload["agent_xy_decoder_training"],
        )
    finally:
        env.close()
    summary = _summary(rows)
    cli.output.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol": {
            "source_adapter_artifact": str(artifact),
            "planning_controller": replay_controller,
            "disable_rgb_goal_angle": bool(cli.disable_rgb_goal_angle),
            "disable_rgb_goal_xy": bool(cli.disable_rgb_goal_xy),
            "base_checkpoint": str(base_checkpoint),
            "planning_only_replay": True,
            "terminal_action_max_magnitude": float(args.terminal_action_max_magnitude),
            "terminal_contact_inward_min": float(args.terminal_contact_inward_min),
            "terminal_contact_tangent_ratio": float(args.terminal_contact_tangent_ratio),
            "terminal_contact_lease_gap_max": float(args.terminal_contact_lease_gap_max),
            "terminal_contact_gap_max": float(args.terminal_contact_gap_max),
            "terminal_agent_distance_max": float(args.terminal_agent_distance_max),
            "terminal_angle_progress_guard": bool(args.terminal_angle_progress_guard),
            "torque_sign_guard": bool(args.torque_sign_guard),
            "terminal_motion_damping": bool(args.terminal_motion_damping),
            "terminal_motion_damping_motion": float(
                getattr(args, "terminal_motion_damping_motion", 2.0)
            ),
            "terminal_motion_damping_regression": float(
                getattr(args, "terminal_motion_damping_regression", 0.5)
            ),
            "terminal_pulse_confirmation": bool(args.terminal_pulse_confirmation),
            "terminal_pulse_min_angle_gain": float(args.terminal_pulse_min_angle_gain),
            "terminal_pulse_max_angle_regression": float(args.terminal_pulse_max_angle_regression),
            "terminal_pulse_max_xy_regression": float(args.terminal_pulse_max_xy_regression),
            "terminal_pulse_fallback_magnitude": float(args.terminal_pulse_fallback_magnitude),
            "terminal_pulse_freeze_steps": int(args.terminal_pulse_freeze_steps),
            "terminal_pulse_max_failures": int(args.terminal_pulse_max_failures),
            "terminal_action_commitment": int(args.terminal_action_commitment),
            "terminal_local_angle_weight": float(
                getattr(args, "terminal_local_angle_weight", 0.0)
            ),
            "online_local_angle_weight": float(
                getattr(args, "online_local_angle_weight", 0.0)
            ),
            "online_local_min_samples": int(
                getattr(args, "online_local_min_samples", 4)
            ),
            "online_local_window": int(
                getattr(args, "online_local_window", 4)
            ),
            "approach_action_max_magnitude": float(
                getattr(args, "approach_action_max_magnitude", -1.0)
            ),
            "signed_torque_probe_max_magnitude": (
                float(args.signed_torque_probe_max_magnitude)
                if getattr(args, "signed_torque_probe_max_magnitude", None) is not None
                else None
            ),
            "geometric_torque_fallback": bool(args.geometric_torque_fallback),
            "geometric_torque_support_lever": bool(args.geometric_torque_support_lever),
            "geometric_torque_magnitude": float(getattr(args, "geometric_torque_magnitude", 0.24)),
            "geometric_torque_inward_magnitude": float(getattr(args, "geometric_torque_inward_magnitude", 0.08)),
            "seed": int(args.seed),
            "planning_episodes": int(args.planning_episodes),
            "planning_steps": int(args.planning_steps),
            "device": str(args.device),
            "planning_start": str(args.planning_start),
            "terminal_start_phase": str(
                getattr(args, "terminal_start_phase", "push")
            ),
            "terminal_ranker_restored": bool(terminal_angular_ranker is not None),
            "terminal_torque_prior_mode": str(
                getattr(args, "_terminal_torque_prior_runtime_mode", "none")
            ),
            "terminal_torque_prior_weight": float(
                getattr(args, "terminal_torque_prior_weight", 1.0)
            ),
            "terminal_torque_prior_blend": float(
                getattr(args, "terminal_torque_prior_blend", 0.25)
            ),
        },
        "planning": summary,
        "claim_boundary": "Planning-only replay of a frozen RGB adapter; no retraining, no simulator-state control input, and no universal planning claim.",
    }
    (cli.output / "planning_replay.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    (cli.output / "planning_replay.md").write_text(
        "# Planning-only replay\n\n"
        f"- Strict success: {summary['strict_success_count']}/{int(args.planning_episodes)}.\n"
        f"- Mean block XY: {summary['mean_block_xy_distance']:.6g}.\n"
        f"- Mean angle error: {summary['mean_angle_error']:.6g}.\n"
        f"- Terminal action cap: {float(args.terminal_action_max_magnitude):.6g}.\n"
        f"- Signed-torque probe cap: {getattr(args, 'signed_torque_probe_max_magnitude', None)}.\n"
        f"- Device: {args.device}.\n"
        f"- Reacquisition attempts/successes: {summary['reacquisition_attempt_count']}/{summary['reacquisition_success_count']}.\n"
        "This artifact replays a fixed trained adapter and changes only the declared terminal action interface.\n"
    )
    print(json.dumps({"output": str(cli.output), "planning": summary}, indent=2))


if __name__ == "__main__":
    main()
