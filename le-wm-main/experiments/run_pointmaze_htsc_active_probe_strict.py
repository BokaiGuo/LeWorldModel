"""Strict RGB PointMaze test of active-probe repair for HTSC non-release.

This wrapper deliberately does not modify the live planner.  It invokes the
existing RGB-only controller once per arm, freezes the probe schedule using
only calibration roots, and reconciles the physical action budget externally:
the active arm receives fewer planning pulses because its observed-effect
probes consume the same finite execution budget as ordinary actions.

The active implementation is the existing
``effect_adapter_rgb_online_verified_local`` controller.  It performs an
episode-start RGB-only HTSC dry-run, collects a goal-progress-only RGB probe
chart only when that dry-run abstains, re-observes each successor, and can
locally re-rank only when the ordinary HTSC gate has abstained and the local
chart has passed its pre-existing verification checks.  It never receives
simulator state as a policy input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VISUAL_METHOD = "visual_dynamics"
ADAPTER_METHOD = "effect_adapter_rgb"
HTSC_METHOD = "effect_adapter_rgb_htsc"
# The legacy residual arm remains the default for replaying historical
# artifacts.  The V4 launcher explicitly chooses the affine chart through
# this environment-bound protocol value; the value is inherited by the raw
# trace auditor and is recorded in every arm report, so it cannot be a hidden
# run-time switch.
ACTIVE_METHOD = os.environ.get(
    "POINTMAZE_HTSC_ACTIVE_METHOD",
    "effect_adapter_rgb_online_verified_local",
)
if ACTIVE_METHOD not in {
    "effect_adapter_rgb_online_verified_local",
    "effect_adapter_rgb_online_affine_verified_local",
    "effect_adapter_rgb_online_affine_candidate_anchored_verified_local",
    "effect_adapter_rgb_online_prior_anchored_candidate_residual_verified_local",
    "effect_adapter_rgb_online_prior_anchored_candidate_residual_contrast_verified_local",
}:
    raise ValueError(f"unsupported POINTMAZE_HTSC_ACTIVE_METHOD={ACTIVE_METHOD!r}")
FULL_METHOD = "effect_adapter_rgb_htsc_consistency"
DEPLOYABLE_METHODS = (
    VISUAL_METHOD,
    ADAPTER_METHOD,
    HTSC_METHOD,
    ACTIVE_METHOD,
    FULL_METHOD,
)
# The legacy strict panel used one context warm-up plus at most 24 action
# blocks.  We retain that exact 25-pulse physical ceiling for every arm.
TOTAL_BLOCK_BUDGET = 25
BOOTSTRAP_DRAWS = 10_000
PROBE_TRIGGERS = (
    "conditional_state_support_failure",
    "conditional_decision_ambiguity",
)
AFFINE_ACTIVE_METHODS = frozenset({
    "effect_adapter_rgb_online_affine_verified_local",
    "effect_adapter_rgb_online_affine_candidate_anchored_verified_local",
})
PRIOR_ANCHORED_RESIDUAL_METHOD = (
    "effect_adapter_rgb_online_prior_anchored_candidate_residual_verified_local"
)
PRIOR_ANCHORED_CONTRAST_METHOD = (
    "effect_adapter_rgb_online_prior_anchored_candidate_residual_contrast_verified_local"
)
LOCAL_CHART_ACTIVE_METHODS = AFFINE_ACTIVE_METHODS | frozenset({
    PRIOR_ANCHORED_RESIDUAL_METHOD,
    PRIOR_ANCHORED_CONTRAST_METHOD,
})
TASK_SAFE_PROBE_DESIGNS = frozenset({
    "task_aligned_positive_half_plane_with_predicted_progress_gate",
    "candidate_anchored_task_safe_affine_d_optimal",
    "candidate_anchored_task_safe_prior_residual_rank2",
})
CANDIDATE_ANCHORED_AFFINE_METHOD = "effect_adapter_rgb_online_affine_candidate_anchored_verified_local"
# A schedule that forms a valid chart on a single fortunate calibration root
# is not a deployable information action.  This lower bound is evaluated only
# on the gate-calibration roots before any held-out root is materialized.
MIN_CANDIDATE_ANCHORED_PROBE_COVERAGE = 0.75


def expected_probe_design_for_method(method: str) -> str:
    """Return the one probe geometry authorized for an active-arm method."""

    if method in {PRIOR_ANCHORED_RESIDUAL_METHOD, PRIOR_ANCHORED_CONTRAST_METHOD}:
        return "candidate_anchored_task_safe_prior_residual_rank2"
    if method == CANDIDATE_ANCHORED_AFFINE_METHOD:
        return "candidate_anchored_task_safe_affine_d_optimal"
    return "task_aligned_positive_half_plane_with_predicted_progress_gate"


def parse_roots(value: str, label: str) -> list[int]:
    roots = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not roots or len(roots) != len(set(roots)):
        raise ValueError(f"{label} must be a non-empty, duplicate-free root list")
    return roots


def parse_probe_triggers(value: str) -> list[str]:
    triggers = [token.strip() for token in value.split(",") if token.strip()]
    if not triggers or len(triggers) != len(set(triggers)) or any(
        trigger not in PROBE_TRIGGERS for trigger in triggers
    ):
        raise ValueError(f"probe trigger grid must be duplicate-free members of {PROBE_TRIGGERS}")
    return triggers


def active_planning_budget(total_block_budget: int, probe_directions: int) -> int:
    """Reserve warm-up and observed probe actions from the shared budget."""

    if int(total_block_budget) < 1:
        raise ValueError("total execution budget must be positive")
    minimum = 2 if ACTIVE_METHOD in {
        PRIOR_ANCHORED_RESIDUAL_METHOD, PRIOR_ANCHORED_CONTRAST_METHOD,
    } else 4
    if int(probe_directions) < minimum or (
        ACTIVE_METHOD not in {
            PRIOR_ANCHORED_RESIDUAL_METHOD, PRIOR_ANCHORED_CONTRAST_METHOD,
        } and int(probe_directions) % 2
    ):
        raise ValueError(
            "prior-anchored residual probes require at least two actions; "
            "other active probes require an even integer >= 4"
        )
    planning = int(total_block_budget) - 1 - int(probe_directions)
    if planning <= 0:
        raise ValueError("active probe size exceeds the shared total action budget")
    return planning


def _paired_delta(
    left_rows: dict[int, dict[str, Any]],
    right_rows: dict[int, dict[str, Any]],
    key: str,
) -> np.ndarray:
    if set(left_rows) != set(right_rows):
        raise ValueError("paired comparison requires exactly equal root sets")
    return np.asarray([
        float(left_rows[root][key]) - float(right_rows[root][key])
        for root in sorted(left_rows)
    ], dtype=np.float64)


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _rows_by_root(report: dict[str, Any], method: str) -> dict[int, dict[str, Any]]:
    rows = [row for row in report["rows"] if row["method"] == method]
    output = {int(row["episode_id"]): row for row in rows}
    if len(output) != len(rows) or not output:
        raise ValueError(f"report is missing unique rows for {method}")
    return output


def choose_calibration_only_probe_config(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze a probe schedule without reading final held-out outcomes.

    Non-inferior calibration success is required before terminal distance is
    used as a tie-break.  A deterministic smaller-probe tie-break means this
    is a finite information-action budget selection, not a test-set search.
    """

    if not candidates or any(not bool(row.get("calibration_only")) for row in candidates):
        raise ValueError("probe configuration selection requires calibration-only candidates")
    viable = [
        row for row in candidates
        if (
            float(row["paired_success_delta_vs_htsc"]) >= 0.0
            and float(row["paired_final_distance_delta_vs_htsc"]) <= 0.0
        )
    ]
    pool = viable if viable else candidates
    selected = min(
        pool,
        key=lambda row: (
            -float(row["paired_success_delta_vs_htsc"]),
            float(row["paired_final_distance_delta_vs_htsc"]),
            int(row["probe_directions"]),
            float(row.get("probe_magnitude", 0.5)),
            str(row.get("active_probe_trigger", "conditional_state_support_failure")),
            float(row.get("active_probe_ambiguity_margin", 0.0)),
            float(row.get("local_chart_loo_tolerance", float("inf"))),
            float(row.get("local_contrast_min_advantage", float("inf"))),
        ),
    )
    trigger = str(selected.get("active_probe_trigger", "conditional_state_support_failure"))
    trigger_semantics = (
        "The probe runs only when the frozen RGB effect policy's best-vs-second-best "
        "candidate cost gap is no larger than its calibration-selected ambiguity margin. "
        "Local re-rank additionally requires successor verification."
        if trigger == "conditional_decision_ambiguity"
        else "The probe runs only after an RGB-only episode-start state-support failure; "
        "action-bank mismatch or bound-only failure remains HTSC fallback. Local re-rank "
        "additionally requires successor verification."
    )
    return {
        **selected,
        "trigger": f"{trigger}_then_local_rerank",
        "trigger_semantics": trigger_semantics,
        "selection_rule": (
            "authorize only if paired calibration success is non-inferior AND paired "
            "final distance is non-inferior; among authorized schedules maximize success, "
            "then minimize final distance, probe size, probe magnitude, trigger name, ambiguity margin, and local LOO tolerance"
        ),
        "selection_authorized": bool(viable),
        "outcome_test_used_for_selection": False,
    }


def candidate_anchored_repairable_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fail closed unless V5 proved its local-chart mechanism on calibration.

    This is an eligibility gate, not an efficacy selector: it consumes only
    the calibration traces needed to show that the proposed intervention was
    actually candidate-anchored and locally repairable before planning.
    """

    eligible = [
        candidate for candidate in candidates
        if (
            float(candidate.get("candidate_anchored_probe_coverage", 0.0))
            >= MIN_CANDIDATE_ANCHORED_PROBE_COVERAGE
            and int(candidate.get("candidate_anchored_repairable_root_count", 0)) > 0
        )
    ]
    if not eligible:
        raise RuntimeError(
            "candidate-anchored V5 calibration produced no repairable, sufficiently covered local chart; "
            "held-out roots remain sealed"
        )
    return eligible


def _charged_blocks(row: dict[str, Any]) -> int:
    return int(
        row["warmup_blocks"]
        + row["active_probe_blocks"]
        + row.get("phase_reprobe_blocks", 0)
        + row["blocks_executed"]
    )


def validate_active_probe_report(
    report: dict[str, Any],
    *,
    expected_roots: list[int],
    total_block_budget: int,
    selected_probe_directions: int,
    selected_probe_magnitude: float = 0.5,
    selected_probe_trigger: str = "conditional_state_support_failure",
    selected_probe_ambiguity_margin: float = 0.0,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Audit root pairing, RGB-only deployment, fixed banks, and budget parity."""

    grouped: dict[int, dict[str, dict[str, Any]]] = {int(root): {} for root in expected_roots}
    for row in report["rows"]:
        root = int(row["episode_id"])
        method = str(row["method"])
        if root not in grouped or method not in DEPLOYABLE_METHODS:
            raise ValueError("active-probe report has an undeclared root or method")
        if method in grouped[root]:
            raise ValueError("active-probe report has duplicate root/method rows")
        contract = row["policy_contract"]
        if bool(contract["simulator_state_in_policy_input"]):
            raise ValueError(f"deployable arm {method} received simulator state")
        if not bool(contract["strict_shared_root_candidate_bank"]):
            raise ValueError(f"deployable arm {method} did not pin its candidate bank")
        charged = _charged_blocks(row)
        if charged > int(total_block_budget):
            raise ValueError(f"{method} root {root} exceeded the equal total action budget")
        if method == ACTIVE_METHOD:
            observed_probe_blocks = int(row["active_probe_blocks"])
            if not 0 <= observed_probe_blocks <= int(selected_probe_directions):
                raise ValueError(
                    "conditional active arm exceeded the frozen observed-effect probe size"
                )
            if observed_probe_blocks == 0 and bool(row.get("probe_triggered")):
                raise ValueError("active arm marked a skipped probe as triggered")
            if observed_probe_blocks == int(selected_probe_directions) and not bool(row.get("probe_triggered")):
                raise ValueError("active arm executed a probe without recording its trigger")
            observed_magnitude = float(
                contract.get("active_probe_magnitude", selected_probe_magnitude)
            )
            if not np.isclose(observed_magnitude, float(selected_probe_magnitude)):
                raise ValueError("active arm did not execute the frozen observed-effect probe magnitude")
            if selected_probe_trigger == "conditional_decision_ambiguity":
                if contract.get("active_probe_trigger") != selected_probe_trigger:
                    raise ValueError("active arm did not execute the frozen decision-ambiguity trigger")
                if not np.isclose(
                    float(contract.get("active_probe_ambiguity_margin", float("nan"))),
                    float(selected_probe_ambiguity_margin),
                ):
                    raise ValueError("active arm did not execute the frozen decision-ambiguity margin")
                # A trace may contain a second ``probe_decision`` record when
                # the ambiguity trigger fires but the calibration dynamics
                # cannot supply a task-safe information action.  That record
                # explains the safe skip; the unique auditable *trigger*
                # decision is the one carrying the frozen dry-run margin.
                decisions = [
                    event for event in row.get("active_probe_trace", [])
                    if (
                        isinstance(event, dict)
                        and bool(event.get("probe_decision"))
                        and "dry_run_decision_margin" in event
                        and "dry_run_decision_ambiguity_margin" in event
                    )
                ]
                if len(decisions) != 1:
                    raise ValueError("decision-ambiguity active arm lacks one auditable probe decision")
                decision = decisions[0]
                margin = float(decision.get("dry_run_decision_margin", float("nan")))
                threshold = float(decision.get("dry_run_decision_ambiguity_margin", float("nan")))
                if not np.isfinite(margin) or not np.isclose(threshold, float(selected_probe_ambiguity_margin)):
                    raise ValueError("decision-ambiguity trace lacks the frozen score-margin evidence")
                if observed_probe_blocks and margin > float(selected_probe_ambiguity_margin):
                    raise ValueError("decision-ambiguity active arm probed outside its frozen margin")
            probe_events = [
                item for item in row.get("active_probe_trace", [])
                if "probe_index" in item and "error" not in item
            ]
            if observed_probe_blocks and len(probe_events) != observed_probe_blocks:
                raise ValueError("active arm probe accounting does not match its RGB trace")
            if observed_probe_blocks and not probe_events:
                raise ValueError("active arm executed probes without an auditable task-safe trace")
            harmful_seen = False
            for probe_event in probe_events:
                design = probe_event.get("probe_design")
                if design not in TASK_SAFE_PROBE_DESIGNS:
                    raise ValueError("active arm did not use the declared task-safe probe geometry")
                if design != expected_probe_design_for_method(method):
                    raise ValueError("active arm did not execute its method-specific frozen probe geometry")
                if design in {
                    "candidate_anchored_task_safe_affine_d_optimal",
                    "candidate_anchored_task_safe_prior_residual_rank2",
                }:
                    candidate_index = probe_event.get("probe_candidate_index")
                    candidates = np.asarray(contract.get("candidate_actions", []), dtype=np.float64)
                    if (
                        not isinstance(candidate_index, int)
                        or candidates.ndim != 2 or candidates.shape[1:] != (2,)
                        or not 0 <= candidate_index < len(candidates)
                        or not np.allclose(
                            np.asarray(probe_event.get("action"), dtype=np.float64),
                            candidates[candidate_index], atol=1.0e-10, rtol=0.0,
                        )
                    ):
                        raise ValueError(
                            "candidate-anchored probe is not an exact member of the fixed candidate bank"
                        )
                if float(probe_event.get("action_goal_projection", float("nan"))) <= 0.0:
                    raise ValueError("active arm trace contains a non-goal-aligned probe action")
                if float(probe_event.get("predicted_goal_progress", float("nan"))) <= 0.0:
                    raise ValueError("active arm executed a probe rejected by its pre-action safety gate")
                if harmful_seen:
                    raise ValueError("active arm continued probing after RGB-observed task regression")
                if float(probe_event.get("observed_goal_progress", 0.0)) <= 0.0:
                    harmful_seen = True
                    if not bool(probe_event.get("probe_sequence_stopped", False)):
                        raise ValueError("active arm did not stop after nonpositive observed probe progress")
        elif int(row["active_probe_blocks"]) != 0 or int(row.get("phase_reprobe_blocks", 0)) != 0:
            raise ValueError(f"non-active arm {method} spent undeclared probe actions")
        row = {**row, "charged_blocks": charged}
        grouped[root][method] = row
    for root, rows in grouped.items():
        if set(rows) != set(DEPLOYABLE_METHODS):
            raise ValueError(f"root {root} is missing a deployable comparison arm")
        digests = {rows[method]["policy_contract"]["candidate_bank_digest"] for method in DEPLOYABLE_METHODS}
        banks = {
            json.dumps(rows[method]["policy_contract"]["candidate_actions"], separators=(",", ":"))
            for method in DEPLOYABLE_METHODS
        }
        if len(digests) != 1 or None in digests or len(banks) != 1:
            raise ValueError(f"root {root} did not receive an exact shared RGB action bank")
    return grouped


def _method_summary(grouped: dict[int, dict[str, dict[str, Any]]], method: str) -> dict[str, Any]:
    rows = [grouped[root][method] for root in sorted(grouped)]
    return {
        "roots": len(rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "mean_final_distance": float(np.mean([float(row["final_distance"]) for row in rows])),
        "mean_charged_blocks": float(np.mean([
            int(row.get("charged_blocks", row.get("warmup_blocks", 0) + row.get("active_probe_blocks", 0) + row.get("blocks_executed", 0)))
            for row in rows
        ])),
        "mean_active_probe_blocks": float(np.mean([int(row["active_probe_blocks"]) for row in rows])),
        "mean_htsc_fallback_rate": float(np.mean([
            np.mean([
                bool(trace.get("action_transfer_gate", {}).get("fallback_used", False))
                for trace in row.get("trace", [])
                if "action_transfer_gate" in trace
            ]) if any("action_transfer_gate" in trace for trace in row.get("trace", [])) else 0.0
            for row in rows
        ])),
    }


def _paired_summary(
    grouped: dict[int, dict[str, dict[str, Any]]],
    *,
    left: str,
    right: str,
    seed: int,
) -> dict[str, Any]:
    left_rows = {root: grouped[root][left] for root in grouped}
    right_rows = {root: grouped[root][right] for root in grouped}
    distance = _paired_delta(left_rows, right_rows, "final_distance")
    success = _paired_delta(left_rows, right_rows, "success")
    return {
        "left": left,
        "right": right,
        "n_root_paired": int(len(distance)),
        "paired_final_distance_delta_left_minus_right": float(distance.mean()),
        "paired_final_distance_delta_95pct_bootstrap": _bootstrap_interval(distance, seed=seed),
        "paired_success_delta_left_minus_right": float(success.mean()),
        "paired_success_delta_95pct_bootstrap": _bootstrap_interval(success, seed=seed + 1),
    }


def active_probe_mechanism_diagnostics(
    grouped: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Separate an executed probe from an actually exercised local repair."""

    rows = [grouped[root][ACTIVE_METHOD] for root in sorted(grouped)]
    gates = [
        trace.get("action_transfer_gate", {})
        for row in rows
        for trace in row.get("trace", [])
        if "action_transfer_gate" in trace
    ]
    nonrelease = [gate for gate in gates if not bool(gate.get("effect_policy_released", True))]
    local_override = [
        gate for gate in gates
        if bool(gate.get("local_verified_release", False))
        and not bool(gate.get("fallback_used", True))
    ]
    return {
        "active_probe_schedule_executed": bool(any(int(row["active_probe_blocks"]) > 0 for row in rows)),
        "probe_trigger_rate": float(np.mean([
            bool(row.get("probe_triggered")) for row in rows
            if row.get("probe_triggered") is not None
        ])) if any(row.get("probe_triggered") is not None for row in rows) else None,
        "probe_skipped_roots": int(sum(int(row["active_probe_blocks"]) == 0 for row in rows)),
        "probe_triggered_roots": int(sum(int(row["active_probe_blocks"]) > 0 for row in rows)),
        "mean_charged_blocks": float(np.mean([
            int(row.get("charged_blocks", row.get("warmup_blocks", 0) + row.get("active_probe_blocks", 0) + row.get("blocks_executed", 0)))
            for row in rows
        ])),
        "mean_probe_progress": float(np.mean([float(row["probe_progress"]) for row in rows])),
        "htsc_nonrelease_decisions": int(len(nonrelease)),
        "support_failure_nonrelease_decisions": int(sum(
            not bool(gate.get("state_support_passed", True))
            or not bool(gate.get("action_support_passed", True))
            for gate in nonrelease
        )),
        "local_verified_release_decisions": int(sum(bool(gate.get("local_verified_release", False)) for gate in gates)),
        "local_rerank_override_decisions": int(len(local_override)),
        "repair_mechanism_exercised": bool(local_override),
        "interpretation": (
            "A scheduled RGB probe is not a repair by itself. The repair is exercised only when a HTSC non-release is "
            "subsequently replaced by the verified local re-rank. Zero overrides localize failure to support/verification "
            "gating or probe-induced state displacement, rather than demonstrating a failed local ranking rule."
        ),
    }


def _run_live(
    *,
    dataset_root: Path,
    checkpoint: Path,
    dinov2_source: Path,
    effect_policy_checkpoint: Path,
    output: Path,
    visual_calibration_roots: list[int],
    effect_consistency_roots: list[int] | None,
    test_roots: list[int],
    method: str,
    max_blocks: int,
    active_probe_directions: int,
    active_probe_magnitude: float,
    active_probe_trigger: str = "always",
    active_probe_ambiguity_margin: float = 0.0,
    local_chart_loo_tolerance: float | None = None,
    local_contrast_min_advantage: float | None = None,
    test_state_root: Path | None = None,
    test_visual_source: str = "stored_episode",
) -> dict[str, Any]:
    if test_state_root is not None and test_visual_source != "dataset_matched_renderer":
        raise ValueError("an external test-state root requires dataset_matched_renderer")
    command = [
        sys.executable, "-m", "experiments.run_live_pointmaze_planning",
        "--dataset-root", str(dataset_root.resolve()),
        "--checkpoint", str(checkpoint.resolve()),
        "--dinov2-source", str(dinov2_source.resolve()),
        "--effect-policy-checkpoint", str(effect_policy_checkpoint.resolve()),
        "--output", str(output.resolve()),
        "--calibration-episodes", ",".join(map(str, visual_calibration_roots)),
        "--test-episodes", ",".join(map(str, test_roots)),
        "--test-visual-source", str(test_visual_source),
        "--max-blocks", str(max_blocks),
        "--active-probe-directions", str(active_probe_directions),
        "--active-probe-magnitude", str(active_probe_magnitude),
        "--active-probe-trigger", str(active_probe_trigger),
        "--active-probe-ambiguity-margin", str(active_probe_ambiguity_margin),
        "--strict-shared-root-candidate-bank",
        "--methods", method,
    ]
    if local_chart_loo_tolerance is not None:
        if not np.isfinite(local_chart_loo_tolerance) or local_chart_loo_tolerance <= 0.0:
            raise ValueError("local chart LOO tolerance must be finite and positive when declared")
        command.extend(["--local-chart-loo-tolerance", str(local_chart_loo_tolerance)])
    if local_contrast_min_advantage is not None:
        if (
            not np.isfinite(local_contrast_min_advantage)
            or local_contrast_min_advantage < 0.0
        ):
            raise ValueError("local contrast minimum advantage must be finite and non-negative")
        command.extend([
            "--local-contrast-min-advantage", str(local_contrast_min_advantage),
        ])
    if test_state_root is not None:
        command.extend(["--test-state-root", str(test_state_root.resolve())])
    if effect_consistency_roots is not None:
        command.extend([
            "--effect-consistency-calibration-episodes",
            ",".join(map(str, effect_consistency_roots)),
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8",
    )
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def _calibration_candidate(
    *,
    baseline: dict[str, Any],
    active: dict[str, Any],
    probe_directions: int,
    probe_magnitude: float,
    active_probe_trigger: str,
    active_probe_ambiguity_margin: float,
    local_chart_loo_tolerance: float | None,
    artifact_name: str,
    local_contrast_min_advantage: float | None = None,
) -> dict[str, Any]:
    htsc = _rows_by_root(baseline, HTSC_METHOD)
    repair = _rows_by_root(active, ACTIVE_METHOD)
    distance = _paired_delta(repair, htsc, "final_distance")
    success = _paired_delta(repair, htsc, "success")
    probe_events = [
        event
        for row in repair.values()
        for event in row.get("active_probe_trace", [])
        if isinstance(event, dict) and "probe_index" in event and "error" not in event
    ]
    candidate_anchored_events = [
        event for event in probe_events
        if event.get("probe_design") == expected_probe_design_for_method(ACTIVE_METHOD)
    ]
    repairable_roots = {
        int(row["episode_id"])
        for row in repair.values()
        if any(
            isinstance(event, dict)
            and bool(event.get("preplanning_affine_repairability", False))
            and bool(event.get("repairable_before_planning", False))
            for event in row.get("active_probe_trace", [])
        )
    }
    candidate_anchored_roots = {
        int(row["episode_id"])
        for row in repair.values()
        if _candidate_anchored_trace_membership_passes(row)
    }
    calibration_root_count = int(len(repair))
    if calibration_root_count <= 0:
        raise ValueError("candidate calibration report has no active-arm roots")
    result = {
        "probe_directions": int(probe_directions),
        "probe_magnitude": float(probe_magnitude),
        "probe_action_duration_blocks": 1,
        "probe_geometry": expected_probe_design_for_method(ACTIVE_METHOD),
        "local_override_visual_regret_cap": (
            None if ACTIVE_METHOD == PRIOR_ANCHORED_CONTRAST_METHOD else 0.0
        ),
        "active_planning_blocks_if_triggered": active_planning_budget(TOTAL_BLOCK_BUDGET, probe_directions),
        "active_planning_blocks_if_skipped": int(TOTAL_BLOCK_BUDGET) - 1,
        "active_probe_trigger": str(active_probe_trigger),
        "active_probe_ambiguity_margin": float(active_probe_ambiguity_margin),
        "artifact_name": str(artifact_name),
        "calibration_only": True,
        "paired_success_delta_vs_htsc": float(success.mean()),
        "paired_final_distance_delta_vs_htsc": float(distance.mean()),
        "paired_final_distance_delta_95pct_bootstrap": _bootstrap_interval(distance, seed=9041 + probe_directions),
        # These mechanism fields are calibration-only eligibility evidence.
        # They are intentionally recorded before outcome-root materialization,
        # and do not call any held-out planner result.
        "calibration_probe_event_count": int(len(probe_events)),
        "candidate_anchored_probe_event_count": int(len(candidate_anchored_events)),
        "candidate_anchored_probe_root_count": int(len(candidate_anchored_roots)),
        "candidate_anchored_probe_coverage": float(
            len(candidate_anchored_roots) / calibration_root_count
        ),
        "candidate_anchored_probe_coverage_minimum": float(
            MIN_CANDIDATE_ANCHORED_PROBE_COVERAGE
        ),
        "preplanning_affine_diagnostic_root_count": int(len({
            int(row["episode_id"])
            for row in repair.values()
            if any(
                isinstance(event, dict)
                and bool(event.get("preplanning_affine_repairability", False))
                for event in row.get("active_probe_trace", [])
            )
        })),
        "preplanning_repairable_root_count": int(len(repairable_roots)),
        "candidate_anchored_repairable_root_count": int(len(
            candidate_anchored_roots & repairable_roots
        )),
    }
    if local_chart_loo_tolerance is not None:
        result["local_chart_loo_tolerance"] = float(local_chart_loo_tolerance)
    if local_contrast_min_advantage is not None:
        result["local_contrast_min_advantage"] = float(local_contrast_min_advantage)
    return result


def _candidate_anchored_trace_membership_passes(row: dict[str, Any]) -> bool:
    """Check that declared V5 probe actions are exact fixed-bank entries."""

    candidates = np.asarray(row.get("policy_contract", {}).get("candidate_actions", []), dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1:] != (2,) or not len(candidates):
        return False
    events = [
        event for event in row.get("active_probe_trace", [])
        if isinstance(event, dict)
        and event.get("probe_design") == expected_probe_design_for_method(ACTIVE_METHOD)
    ]
    if not events:
        return False
    return bool(all(
        isinstance(event.get("probe_candidate_index"), int)
        and 0 <= int(event["probe_candidate_index"]) < len(candidates)
        and np.allclose(
            np.asarray(event.get("action"), dtype=np.float64),
            candidates[int(event["probe_candidate_index"])], atol=1.0e-10, rtol=0.0,
        )
        for event in events
    ))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_root_provenance(
    *,
    path: Path,
    source_manifest: Path,
    visual_roots: list[int],
    gate_roots: list[int],
    test_roots: list[int],
    additional_excluded_roots: list[int],
    calibration_dataset_root: Path,
    test_state_root: Path | None,
    test_visual_source: str,
) -> None:
    """Materialize the root exclusion and split contract before any run.

    The active-probe panel must not rely on an unauditable hand-written root
    list.  We retain the exact source manifest digest and copy the relevant
    prior-design exclusions into the output artifact, while keeping the
    held-out roots explicit and pairwise disjoint.
    """

    if not source_manifest.is_file():
        raise FileNotFoundError(f"root provenance source manifest not found: {source_manifest}")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    extension_manifest = (
        None
        if test_state_root is None
        else test_state_root.resolve() / "extension_manifest.json"
    )
    if extension_manifest is not None and not extension_manifest.is_file():
        raise FileNotFoundError(
            f"external test-state root lacks extension provenance: {extension_manifest}"
        )
    _write_json(path, {
        "schema": "pointmaze.htsc_active_probe.root_provenance.v1",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_panel_schema": source.get("panel_schema"),
        "excluded_prior_roots": {
            "interaction_design_roots": source.get("interaction_design_roots", []),
            "renderer_chart_roots": source.get("renderer_chart_provenance", {}).get("chart_roots", []),
            "v6_split_roots": source.get("renderer_chart_provenance", {}).get("v6_split_roots_excluded", []),
            "additional_runtime_roots": additional_excluded_roots,
        },
        "current_split": {
            "visual_calibration_roots": visual_roots,
            "gate_calibration_roots": gate_roots,
            "held_out_test_roots": test_roots,
            "pairwise_disjoint": True,
            "held_out_root_count": len(test_roots),
        },
        "runtime_observation_contract": {
            "calibration_dataset_root": str(calibration_dataset_root.resolve()),
            "test_state_root": (
                None if test_state_root is None else str(test_state_root.resolve())
            ),
            "test_visual_source": str(test_visual_source),
            "test_state_is_external_and_renderer_only": bool(test_state_root is not None),
            "test_state_extension_manifest": (
                None if extension_manifest is None else str(extension_manifest)
            ),
            "test_state_extension_manifest_sha256": (
                None if extension_manifest is None else _sha256(extension_manifest)
            ),
        },
        "selection_contract": {
            "held_out_outcomes_read_before_freeze": False,
            "candidate_probe_schedule_selected_on_gate_calibration_only": True,
            "no_cpu_fallback": True,
        },
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/point_maze_tail1800_1899")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar")
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    parser.add_argument("--effect-policy-checkpoint", type=Path, default=ROOT / "results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-calibration-roots", default="1868,1869,1870,1871,1872,1873,1874,1875,1876,1877")
    parser.add_argument("--gate-calibration-roots", default="1878,1879,1880,1881,1882,1883,1884,1885,1886,1887")
    parser.add_argument("--test-roots", default="1888,1889,1890,1891,1892,1893,1894,1895,1896,1897,1898,1899")
    parser.add_argument("--probe-direction-grid", default="4,6")
    parser.add_argument(
        "--active-probe-trigger-grid",
        default="conditional_state_support_failure",
        help=(
            "comma-separated calibration-only trigger families: state-support failure "
            "and/or frozen action-ranking ambiguity"
        ),
    )
    parser.add_argument("--probe-magnitude", type=float, default=0.5)
    parser.add_argument(
        "--probe-magnitude-grid",
        default="",
        help="optional comma-separated calibration-only probe magnitudes",
    )
    parser.add_argument(
        "--probe-ambiguity-margin-grid",
        default="0.0",
        help=(
            "comma-separated non-negative score gaps tested only for the "
            "conditional_decision_ambiguity trigger"
        ),
    )
    parser.add_argument(
        "--local-chart-loo-tolerance-grid",
        default="0.05,0.10,0.20",
        help=(
            "finite calibration-only leave-one-out effect-error thresholds for the "
            "affine local-repair arm; ignored by the legacy residual replay arm"
        ),
    )
    parser.add_argument(
        "--local-contrast-min-advantage-grid",
        default="0.00,0.02,0.05",
        help=(
            "calibration-only terminal-distance advantages for the prior-anchored "
            "contrast-release arm; ignored by all other active arms"
        ),
    )
    parser.add_argument("--total-block-budget", type=int, default=TOTAL_BLOCK_BUDGET)
    parser.add_argument(
        "--test-state-root",
        type=Path,
        default=None,
        help=(
            "optional fresh held-out state/action dataset; it is legal only with the "
            "calibration-only dataset-matched RGB renderer"
        ),
    )
    parser.add_argument(
        "--test-visual-source",
        choices=("stored_episode", "dataset_matched_renderer"),
        default="stored_episode",
        help="held-out RGB source; fresh state roots require dataset_matched_renderer",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "results/pointmaze_temporal_ood_v6_9_protocol_repair_20261021_cuda/temporal_ood_v6_manifest.json",
    )
    parser.add_argument(
        "--additional-excluded-roots",
        default="",
        help="Previously used runtime roots excluded from this fresh panel.",
    )
    args = parser.parse_args()

    if args.test_state_root is not None and args.test_visual_source != "dataset_matched_renderer":
        raise ValueError("--test-state-root requires --test-visual-source dataset_matched_renderer")

    visual_roots = parse_roots(args.visual_calibration_roots, "visual calibration roots")
    gate_roots = parse_roots(args.gate_calibration_roots, "gate calibration roots")
    test_roots = parse_roots(args.test_roots, "test roots")
    additional_excluded_roots = (
        parse_roots(args.additional_excluded_roots, "additional excluded roots")
        if args.additional_excluded_roots.strip() else []
    )
    named = {"visual": set(visual_roots), "gate": set(gate_roots), "test": set(test_roots)}
    if any(named[left] & named[right] for left in named for right in named if left < right):
        raise ValueError("visual, gate, and held-out root splits must be pairwise disjoint")
    if any(named[name] & set(additional_excluded_roots) for name in named):
        raise ValueError("runtime split overlaps an explicitly excluded prior root")
    direction_grid = parse_roots(args.probe_direction_grid, "probe direction grid")
    trigger_grid = parse_probe_triggers(args.active_probe_trigger_grid)
    residual_two_pulse_methods = {
        PRIOR_ANCHORED_RESIDUAL_METHOD, PRIOR_ANCHORED_CONTRAST_METHOD,
    }
    minimum_probe_actions = 2 if ACTIVE_METHOD in residual_two_pulse_methods else 4
    if any(
        value < minimum_probe_actions
        or (ACTIVE_METHOD not in residual_two_pulse_methods and value % 2)
        for value in direction_grid
    ):
        raise ValueError(
            "prior-anchored residual grid requires values >= 2; other grids require even values >= 4"
        )
    for value in direction_grid:
        active_planning_budget(args.total_block_budget, value)
    if args.probe_magnitude_grid.strip():
        try:
            magnitude_grid = [
                float(token.strip()) for token in args.probe_magnitude_grid.split(",")
                if token.strip()
            ]
        except ValueError as exc:
            raise ValueError("probe magnitude grid must be comma-separated floats") from exc
    else:
        magnitude_grid = [float(args.probe_magnitude)]
    if (
        not magnitude_grid
        or len(set(magnitude_grid)) != len(magnitude_grid)
        or any(not 0.0 < value <= 1.0 for value in magnitude_grid)
    ):
        raise ValueError("probe magnitude grid must be duplicate-free values in (0, 1]")
    try:
        ambiguity_margin_grid = [
            float(token.strip()) for token in args.probe_ambiguity_margin_grid.split(",")
            if token.strip()
        ]
    except ValueError as exc:
        raise ValueError("probe ambiguity margin grid must be comma-separated floats") from exc
    if (
        not ambiguity_margin_grid
        or len(set(ambiguity_margin_grid)) != len(ambiguity_margin_grid)
        or any(value < 0.0 or not np.isfinite(value) for value in ambiguity_margin_grid)
    ):
        raise ValueError("probe ambiguity margin grid must be duplicate-free finite values >= 0")
    if ACTIVE_METHOD in AFFINE_ACTIVE_METHODS:
        try:
            local_chart_loo_grid: list[float | None] = [
                float(token.strip()) for token in args.local_chart_loo_tolerance_grid.split(",")
                if token.strip()
            ]
        except ValueError as exc:
            raise ValueError("local chart LOO tolerance grid must be comma-separated floats") from exc
        if (
            not local_chart_loo_grid
            or len(set(local_chart_loo_grid)) != len(local_chart_loo_grid)
            or any(value is None or not np.isfinite(value) or value <= 0.0 for value in local_chart_loo_grid)
        ):
            raise ValueError("local chart LOO tolerance grid must be duplicate-free finite values > 0")
    else:
        local_chart_loo_grid = [None]
    if ACTIVE_METHOD == PRIOR_ANCHORED_CONTRAST_METHOD:
        try:
            local_contrast_grid: list[float | None] = [
                float(token.strip()) for token in args.local_contrast_min_advantage_grid.split(",")
                if token.strip()
            ]
        except ValueError as exc:
            raise ValueError("local contrast advantage grid must be comma-separated floats") from exc
        if (
            not local_contrast_grid
            or len(set(local_contrast_grid)) != len(local_contrast_grid)
            or any(value is None or not np.isfinite(value) or value < 0.0 for value in local_contrast_grid)
        ):
            raise ValueError("local contrast advantage grid must be duplicate-free finite values >= 0")
    else:
        local_contrast_grid = [None]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_root_provenance(
        path=args.output_dir / "root_provenance.json",
        source_manifest=args.source_manifest,
        visual_roots=visual_roots,
        gate_roots=gate_roots,
        test_roots=test_roots,
        additional_excluded_roots=additional_excluded_roots,
        calibration_dataset_root=args.dataset_root,
        test_state_root=args.test_state_root,
        test_visual_source=args.test_visual_source,
    )
    preflight_path = args.output_dir / "gpu_preflight.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.gpu_preflight",
            "--device",
            "cuda",
            "--recurrent-smoke",
            "--output",
            str(preflight_path),
        ],
        cwd=ROOT,
        check=True,
    )

    # Calibration outcomes choose only the finite probe schedule.  The final
    # test roots are not mentioned until this object has been materialized.
    calibration_dir = args.output_dir / "calibration_only"
    htsc_calibration = _run_live(
        dataset_root=args.dataset_root, checkpoint=args.checkpoint, dinov2_source=args.dinov2_source,
        effect_policy_checkpoint=args.effect_policy_checkpoint,
        output=calibration_dir / "htsc_fallback.json",
        visual_calibration_roots=visual_roots, effect_consistency_roots=None,
        test_roots=gate_roots, method=HTSC_METHOD,
        max_blocks=int(args.total_block_budget) - 1, active_probe_directions=4,
        active_probe_magnitude=float(args.probe_magnitude),
    )
    candidates: list[dict[str, Any]] = []
    for trigger in trigger_grid:
        trigger_margins = ambiguity_margin_grid if trigger == "conditional_decision_ambiguity" else [0.0]
        for ambiguity_margin in trigger_margins:
            for directions in direction_grid:
                for magnitude in magnitude_grid:
                    for local_chart_loo_tolerance in local_chart_loo_grid:
                        for local_contrast_min_advantage in local_contrast_grid:
                            magnitude_tag = str(magnitude).replace(".", "p")
                            ambiguity_tag = str(ambiguity_margin).replace(".", "p")
                            loo_tag = "" if local_chart_loo_tolerance is None else (
                                f"_loo{str(local_chart_loo_tolerance).replace('.', 'p')}"
                            )
                            contrast_tag = "" if local_contrast_min_advantage is None else (
                                f"_adv{str(local_contrast_min_advantage).replace('.', 'p')}"
                            )
                            artifact_name = (
                                f"active_probe_{trigger}_{directions}_{magnitude_tag}_margin"
                                f"{ambiguity_tag}{loo_tag}{contrast_tag}"
                            )
                            candidate_report = _run_live(
                                dataset_root=args.dataset_root,
                                checkpoint=args.checkpoint,
                                dinov2_source=args.dinov2_source,
                                effect_policy_checkpoint=args.effect_policy_checkpoint,
                                output=calibration_dir / f"{artifact_name}.json",
                                visual_calibration_roots=visual_roots,
                                effect_consistency_roots=None,
                                test_roots=gate_roots,
                                method=ACTIVE_METHOD,
                                max_blocks=int(args.total_block_budget) - 1,
                                active_probe_directions=directions,
                                active_probe_magnitude=float(magnitude),
                                active_probe_trigger=trigger,
                                active_probe_ambiguity_margin=float(ambiguity_margin),
                                local_chart_loo_tolerance=local_chart_loo_tolerance,
                                local_contrast_min_advantage=local_contrast_min_advantage,
                            )
                            candidates.append(_calibration_candidate(
                                baseline=htsc_calibration,
                                active=candidate_report,
                                probe_directions=directions,
                                probe_magnitude=float(magnitude),
                                active_probe_trigger=trigger,
                                active_probe_ambiguity_margin=float(ambiguity_margin),
                                local_chart_loo_tolerance=local_chart_loo_tolerance,
                                local_contrast_min_advantage=local_contrast_min_advantage,
                                artifact_name=artifact_name,
                            ))
    # V5 is a decision-support-aligned local-repair intervention, not merely
    # an action-budget ablation.  Do not expose held-out roots if calibration
    # never produced one candidate-anchored chart that is already rank/LOO/
    # hull repairable before planning.  Otherwise a no-probe or
    # non-identifiable schedule could create an uninterpretable null result.
    selection_candidates = candidates
    if ACTIVE_METHOD in {
        CANDIDATE_ANCHORED_AFFINE_METHOD,
        PRIOR_ANCHORED_RESIDUAL_METHOD,
        PRIOR_ANCHORED_CONTRAST_METHOD,
    }:
        selection_candidates = candidate_anchored_repairable_candidates(candidates)
    frozen_probe = choose_calibration_only_probe_config(selection_candidates)
    if ACTIVE_METHOD in AFFINE_ACTIVE_METHODS:
        selected_loo = frozen_probe.get("local_chart_loo_tolerance")
        if selected_loo is None or not np.isfinite(float(selected_loo)) or float(selected_loo) <= 0.0:
            raise RuntimeError("affine active arm did not freeze a finite calibration-only LOO threshold")
    if ACTIVE_METHOD == PRIOR_ANCHORED_CONTRAST_METHOD:
        selected_advantage = frozen_probe.get("local_contrast_min_advantage")
        if (
            selected_advantage is None
            or not np.isfinite(float(selected_advantage))
            or float(selected_advantage) < 0.0
        ):
            raise RuntimeError("contrast active arm did not freeze a finite calibration-only advantage")
    _write_json(args.output_dir / "frozen_probe_protocol.json", {
        "schema": "pointmaze.htsc_active_probe.frozen_protocol.v1",
        "visual_calibration_roots": visual_roots,
        "gate_calibration_roots": gate_roots,
        "held_out_test_roots_read_after_freeze": test_roots,
        "candidate_calibration": candidates,
        "selection_candidate_count": int(len(selection_candidates)),
        "candidate_anchored_repairability_required": bool(
            ACTIVE_METHOD in {
                CANDIDATE_ANCHORED_AFFINE_METHOD,
                PRIOR_ANCHORED_RESIDUAL_METHOD,
                PRIOR_ANCHORED_CONTRAST_METHOD,
            }
        ),
        "frozen_probe": frozen_probe,
        "test_outcomes_read_for_selection": False,
    })

    # Each arm is invoked separately only because the immutable planner takes
    # one max-block value per process.  The auditor below proves exact root
    # banks and equal total (warm-up + probe + planning) caps after merging.
    final_dir = args.output_dir / "held_out"
    planning_budget = int(args.total_block_budget) - 1
    reports: dict[str, dict[str, Any]] = {}
    for method in (VISUAL_METHOD, ADAPTER_METHOD, HTSC_METHOD, FULL_METHOD):
        reports[method] = _run_live(
            dataset_root=args.dataset_root, checkpoint=args.checkpoint, dinov2_source=args.dinov2_source,
            effect_policy_checkpoint=args.effect_policy_checkpoint,
            output=final_dir / f"{method}.json",
            visual_calibration_roots=visual_roots,
            effect_consistency_roots=gate_roots if method == FULL_METHOD else None,
            test_roots=test_roots, method=method, max_blocks=planning_budget,
            active_probe_directions=int(frozen_probe["probe_directions"]),
            active_probe_magnitude=float(frozen_probe["probe_magnitude"]),
            test_state_root=args.test_state_root,
            test_visual_source=args.test_visual_source,
        )
    reports[ACTIVE_METHOD] = _run_live(
        dataset_root=args.dataset_root, checkpoint=args.checkpoint, dinov2_source=args.dinov2_source,
        effect_policy_checkpoint=args.effect_policy_checkpoint,
        output=final_dir / f"{ACTIVE_METHOD}.json",
        visual_calibration_roots=visual_roots, effect_consistency_roots=None,
        test_roots=test_roots, method=ACTIVE_METHOD,
            max_blocks=int(args.total_block_budget) - 1,
        active_probe_directions=int(frozen_probe["probe_directions"]),
        active_probe_magnitude=float(frozen_probe["probe_magnitude"]),
        active_probe_trigger=str(frozen_probe["active_probe_trigger"]),
        active_probe_ambiguity_margin=float(frozen_probe["active_probe_ambiguity_margin"]),
        local_chart_loo_tolerance=(
            None if frozen_probe.get("local_chart_loo_tolerance") is None
            else float(frozen_probe["local_chart_loo_tolerance"])
        ),
        local_contrast_min_advantage=(
            None if frozen_probe.get("local_contrast_min_advantage") is None
            else float(frozen_probe["local_contrast_min_advantage"])
        ),
        test_state_root=args.test_state_root,
        test_visual_source=args.test_visual_source,
    )
    merged = {"rows": [
        row for method in DEPLOYABLE_METHODS for row in reports[method]["rows"]
    ]}
    grouped = validate_active_probe_report(
        merged,
        expected_roots=test_roots,
        total_block_budget=int(args.total_block_budget),
        selected_probe_directions=int(frozen_probe["probe_directions"]),
        selected_probe_magnitude=float(frozen_probe["probe_magnitude"]),
        selected_probe_trigger=str(frozen_probe["active_probe_trigger"]),
        selected_probe_ambiguity_margin=float(frozen_probe["active_probe_ambiguity_margin"]),
    )
    paired = {
        method: _paired_summary(grouped, left=method, right=VISUAL_METHOD, seed=4400 + index)
        for index, method in enumerate(DEPLOYABLE_METHODS)
    }
    paired["active_probe_vs_htsc_fallback"] = _paired_summary(
        grouped, left=ACTIVE_METHOD, right=HTSC_METHOD, seed=5599,
    )
    mechanism = active_probe_mechanism_diagnostics(grouped)
    active_arm_status = (
        "calibration_rejected_diagnostic_only"
        if not bool(frozen_probe["selection_authorized"])
        else (
            "authorized_but_repair_not_exercised"
            if not bool(mechanism["repair_mechanism_exercised"])
            else "authorized_and_mechanism_exercised"
        )
    )
    report = {
        "schema": "pointmaze.htsc_active_probe.strict_rgb_closed_loop.v1",
        "status": "completed",
        "active_arm_status": active_arm_status,
        "methods": {
            VISUAL_METHOD: "frozen_visual_dynamics",
            ADAPTER_METHOD: "intervention_supervised_effect_adapter",
            HTSC_METHOD: "HTSC_fallback",
            ACTIVE_METHOD: "calibration_frozen_short_observed_effect_probe_then_local_rerank",
            FULL_METHOD: "effect_adapter_HTSC_visual_consistency",
        },
        "split_protocol": {
            "adapter_train_and_effect_calibration": "inherited and verified inside the frozen effect-policy checkpoint",
            "visual_calibration_roots": visual_roots,
            "gate_calibration_roots": gate_roots,
            "held_out_test_roots": test_roots,
            "pairwise_disjoint_named_runtime_splits": True,
        },
        "budget_contract": {
            "total_physical_block_budget": int(args.total_block_budget),
            "baseline_planning_block_cap": planning_budget,
            "active_planning_block_cap_if_triggered": active_planning_budget(args.total_block_budget, int(frozen_probe["probe_directions"])),
            "active_planning_block_cap_if_skipped": planning_budget,
            "active_probe_trigger": str(frozen_probe["active_probe_trigger"]),
            "active_probe_ambiguity_margin": float(frozen_probe["active_probe_ambiguity_margin"]),
            "local_chart_loo_tolerance": frozen_probe.get("local_chart_loo_tolerance"),
            "local_contrast_min_advantage": frozen_probe.get("local_contrast_min_advantage"),
            "active_probe_actions_count_against_total_budget": True,
            "active_probe_geometry": expected_probe_design_for_method(ACTIVE_METHOD),
            "candidate_anchored_repairability_required": bool(
            ACTIVE_METHOD in {
                CANDIDATE_ANCHORED_AFFINE_METHOD,
                PRIOR_ANCHORED_RESIDUAL_METHOD,
                PRIOR_ANCHORED_CONTRAST_METHOD,
            }
            ),
            "local_override_visual_regret_cap": (
                None if ACTIVE_METHOD == PRIOR_ANCHORED_CONTRAST_METHOD else 0.0
            ),
            "all_methods_rgb_only": True,
            "privileged_methods_included": False,
        },
        "frozen_probe": frozen_probe,
        "aggregate": {method: _method_summary(grouped, method) for method in DEPLOYABLE_METHODS},
        "paired": paired,
        "active_probe_mechanism_diagnostics": mechanism,
        "artifacts": {
            "gpu_preflight": str(preflight_path.resolve()),
            "calibration_only_reports": {
                "htsc": str((calibration_dir / "htsc_fallback.json").resolve()),
                "active_probe_candidates": [
                    str((calibration_dir / f"{candidate['artifact_name']}.json").resolve())
                    for candidate in candidates
                ],
            },
            "held_out_arm_reports": {method: str((final_dir / f"{method}.json").resolve()) for method in DEPLOYABLE_METHODS},
        },
        "claim_boundary": (
            f"This is a single legacy PointMaze environment with a fixed RGB action bank and {len(test_roots)} held-out roots. "
            "It tests whether a calibration-frozen observed-effect probe repairs HTSC non-release under an equal physical "
            "pulse budget. It does not establish finite-sample coverage transport, universal transfer, physical causality, "
            "or a universally reliable world model. A calibration-rejected active configuration is reported only as a "
            "mechanism diagnostic, never as a deployment authorization."
        ),
    }
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "report": str((args.output_dir / "report.json").resolve()),
        "frozen_probe": frozen_probe,
        "paired_active_vs_htsc": paired["active_probe_vs_htsc_fallback"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
