"""Prospectively certify an endpoint-coupled repair on a fresh PointMaze panel.

V5 established a specific structural limitation: its planning-shaped arm uses
future-effect supervision only as an auxiliary loss while an independent cost
head still ranks actions.  This runner tests the predeclared repair: predict a
candidate-conditioned terminal state from visual history and obtain the task
score directly from that endpoint and the declared goal.

The primary comparison is fixed cost-only versus the endpoint-coupled arm.
The legacy planning-shaped auxiliary arm is retained as a same-panel
descriptive bridge, but is not selected after observing outcomes.  Calibration
and outcome roots are disjoint; the outcome terminal states are materialized
only after the selector/certification decision is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.gdd_wm.run_pointmaze_planning_shaped_control import (
    Objective,
    _calibration_terminal,
    _cost,
    _canonical_action_summary,
    _feature_canonical_action_summary,
    _feature_ordered_action_sequence,
    _fresh_terminal,
    _group_rows,
    _sha256,
    _terminal_from_ref,
    _fit_counterfactual_endpoint,
    action_representation_input_fields,
    endpoint_task_cost,
)
from experiments.gdd_wm.run_pointmaze_planning_shaped_repairability import (
    COST_ONLY,
    PLANNING_SHAPED,
    _fit_ensemble,
    _post_execution_labels,
    _upper_bootstrap_delta,
)
from experiments.decision_benchmark.task_contrast import (
    calibrate_contrast_bound,
    certify_candidate_bank,
    evaluate_contrast_bound,
)
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_counterfactual_endpoint_control.v1"
CONTROL_LABEL = "predeclared_endpoint_coupled_counterfactual_repair"
MINIMUM_DISAGREEMENT_ROOTS = 12
MINIMUM_SUMMARY_COLLISION_PAIRS_PER_ROOT = 8
MINIMUM_GOAL_BALANCED_DESIGN_WINNER_ENTROPY = 0.5


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _root_fold(root: str) -> int:
    return int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:8], 16) % 4


def _calibration_role(root: str) -> int:
    """Assign one of three disjoint calibration roles by root identity."""

    return int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:8], 16) % 3


def _choice(rows: list[dict[str, Any]], score: np.ndarray) -> int:
    return min(range(len(rows)), key=lambda index: (float(score[index]), str(rows[index]["candidate_id"])))


def _regret(cost: np.ndarray, choice: int) -> float:
    return float(cost[choice] - np.min(cost))


def _enforce_arm_contract(contract: dict[str, Any], *, expected_representation: str) -> None:
    """Reject a missing or mismatched action schema before calibration/outcomes.

    These fields are not decorative report metadata: V6's primary comparison
    fixes the ordered action representation.  Silently filling a missing field
    would make a malformed producer appear valid only after model fitting.
    """

    expected_fields = action_representation_input_fields(expected_representation)
    declared = contract.get("action_representation")
    if declared != expected_representation:
        raise ValueError(
            "V6 arm declared incompatible action representation: "
            f"expected={expected_representation}, actual={declared}"
        )
    fields = contract.get("input_fields")
    if fields != expected_fields:
        raise ValueError(
            "V6 arm declared incompatible input fields: "
            f"expected={expected_fields}, actual={fields}"
        )


def _fit_endpoint_ensemble(
    rows: list[dict[str, Any]], terminal: np.ndarray, *, seeds: tuple[int, ...], width: int, steps: int, device: str,
    feature_fn: Callable[[dict[str, Any]], np.ndarray] = _feature_ordered_action_sequence,
    action_representation: str = "full_ordered_20x2",
) -> tuple[dict[str, Any], Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    """Average endpoints first, then derive one coupled candidate score."""

    if len(rows) != len(terminal):
        raise ValueError("rows and terminal targets differ")
    features = np.stack([feature_fn(row) for row in rows])
    goal = np.asarray([row["goal_raw"] for row in rows], dtype=np.float64)
    groups = _group_rows(rows)
    members = [
        _fit_counterfactual_endpoint(
            features, terminal, goal, groups, width=width, steps=steps, device=device, seed=seed,
            action_representation=action_representation,
        )
        for seed in seeds
    ]
    contract = {
        **members[0][0], "initialization_seeds": list(seeds), "ensemble_size": len(seeds),
        "ensemble_readout": "mean predicted terminal, then endpoint task readout",
    }

    def predict(values: np.ndarray, candidate_goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        endpoint = np.mean(member_endpoints(values, candidate_goal), axis=0)
        return endpoint_task_cost(endpoint, candidate_goal), endpoint

    def member_endpoints(values: np.ndarray, candidate_goal: np.ndarray) -> np.ndarray:
        """Return member terminal predictions without changing deployed scoring.

        V5 averaged four initialization replicas and discarded their spread.
        That makes a negative repair result ambiguous: it can be a readout
        failure or an intrinsically unstable counterfactual prediction.  V6
        records this spread only as a pre-outcome diagnostic; authorization
        still uses exactly the frozen mean-endpoint policy below.
        """

        endpoints = np.stack([member(values, candidate_goal)[1] for _contract, member in members])
        if endpoints.ndim != 3 or endpoints.shape[1:] != (len(values), 4) or not np.isfinite(endpoints).all():
            raise ValueError("endpoint ensemble members must return finite [member,candidate,4] predictions")
        return endpoints

    # Attribute access keeps the public predictor interface unchanged while
    # allowing the runner to serialize a descriptive pre-outcome uncertainty
    # audit.  No policy code reads this attribute.
    setattr(predict, "member_endpoint_predictions", member_endpoints)

    return contract, predict


def _endpoint_ensemble_diagnostics(
    grouped: dict[str, list[dict[str, Any]]],
    predictor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Describe ensemble counterfactual disagreement before outcome execution.

    This is not a confidence certificate and it cannot authorize, reject, or
    re-rank a candidate.  It exposes whether a future V7 uncertainty gate is
    even scientifically warranted, without using realized terminal labels.
    """

    member_predictor = getattr(predictor, "member_endpoint_predictions", None)
    if not callable(member_predictor):
        return {
            "available": False,
            "reason": "predictor does not expose member endpoint predictions",
            "claim_boundary": "unavailable diagnostic; no uncertainty conclusion",
        }
    per_root: dict[str, dict[str, Any]] = {}
    for root, rows in sorted(grouped.items()):
        features = np.stack([_feature_ordered_action_sequence(row) for row in rows])
        goals = np.asarray([row["goal_raw"] for row in rows], dtype=np.float64)
        deployed_score, deployed_endpoint = predictor(features, goals)
        members = np.asarray(member_predictor(features, goals), dtype=np.float64)
        if members.ndim != 3 or members.shape[1:] != (len(rows), 4):
            raise ValueError(f"{root}: endpoint-member predictions require [member,candidate,4]")
        member_score = np.stack([endpoint_task_cost(member, goals) for member in members])
        deployed_choice = _choice(rows, np.asarray(deployed_score, dtype=np.float64))
        member_choice = np.asarray([_choice(rows, score) for score in member_score], dtype=np.int64)
        selected_spread = np.linalg.norm(
            members[:, deployed_choice, :2] - np.asarray(deployed_endpoint, dtype=np.float64)[deployed_choice, :2], axis=1,
        )
        per_root[root] = {
            "candidate_count": len(rows), "member_count": int(len(members)),
            "deployed_candidate_id": str(rows[deployed_choice]["candidate_id"]),
            "member_agreement_with_deployed_choice": float(np.mean(member_choice == deployed_choice)),
            "distinct_member_winner_count": int(len(set(member_choice.tolist()))),
            "mean_candidate_score_standard_deviation": float(member_score.std(axis=0).mean()),
            "selected_endpoint_xy_spread_mean": float(selected_spread.mean()),
            "selected_endpoint_xy_spread_max": float(selected_spread.max()),
        }
    agreement = np.asarray([row["member_agreement_with_deployed_choice"] for row in per_root.values()], dtype=np.float64)
    spread = np.asarray([row["selected_endpoint_xy_spread_mean"] for row in per_root.values()], dtype=np.float64)
    return {
        "available": True, "root_count": len(per_root), "per_root": per_root,
        "mean_member_agreement_with_deployed_choice": float(agreement.mean()),
        "mean_selected_endpoint_xy_spread": float(spread.mean()),
        "claim_boundary": "pre-outcome initialization-ensemble disagreement diagnostic only; it does not calibrate epistemic uncertainty or change policy authorization.",
    }


def _group_rows_by_root(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["instance_id"])].append(row)
    if not grouped or any(len(value) < 2 for value in grouped.values()):
        raise ValueError("each root must contain at least two candidates")
    return dict(grouped)


def _cost_map(rows: list[dict[str, Any]], terminal: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, terminal):
        grouped[str(row["instance_id"])].append(float(np.linalg.norm(value[:2] - np.asarray(row["goal_raw"], dtype=np.float64))))
    return {root: np.asarray(value, dtype=np.float64) for root, value in grouped.items()}


def _assert_split_contract(
    train_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]], outcome_rows: list[dict[str, Any]],
    context_metadata: Any,
) -> dict[str, Any]:
    """Fail closed on root reuse or outcome-label leakage before any fitting."""

    train = {str(row.get("instance_id", "")) for row in train_rows}
    calibration = {str(row.get("instance_id", "")) for row in calibration_rows}
    outcome = {str(row.get("instance_id", "")) for row in outcome_rows}
    if not train or not calibration or not outcome or "" in train | calibration | outcome:
        raise ValueError("all train, calibration, and outcome rows require non-empty root identities")
    overlaps = {
        "train_calibration": sorted(train & calibration), "train_outcome": sorted(train & outcome),
        "calibration_outcome": sorted(calibration & outcome),
    }
    if any(overlaps.values()):
        raise ValueError(f"root split overlap is forbidden: {overlaps}")
    if not isinstance(context_metadata, dict) or context_metadata.get("candidate_generalization_regime") != "shared_candidate_in_support":
        raise ValueError("V6 requires declared shared_candidate_in_support context metadata")
    if context_metadata.get("candidate_design") != "goal_balanced_interaction_pairs":
        raise ValueError("V6 requires the declared goal_balanced_interaction_pairs candidate design")
    preflight = context_metadata.get("frozen_preflight_ledger")
    if not isinstance(preflight, dict) or preflight.get("schema") != "pointmaze_counterfactual_endpoint_v6_static_preflight.v1":
        raise ValueError("V6 requires a frozen static-preflight ledger bound into panel metadata")
    if preflight.get("status") != "passed_static_preflight" or not isinstance(preflight.get("sha256"), str) or len(preflight["sha256"]) != 64:
        raise ValueError("V6 panel metadata has an invalid frozen static-preflight ledger provenance")
    expected_counts = {
        "train": len(train), "calibration": len(calibration), "outcome": len(outcome),
    }
    if preflight.get("root_counts") != expected_counts:
        raise ValueError("V6 panel root counts differ from its frozen static-preflight ledger")
    interaction_design = context_metadata.get("interaction_pair_design")
    if not isinstance(interaction_design, dict):
        raise ValueError("V6 requires serialized independent goal-balanced candidate-design evidence")
    entropy = interaction_design.get("design_winner_entropy_normalized")
    if not isinstance(entropy, (float, int)) or not np.isfinite(float(entropy)) or float(entropy) < MINIMUM_GOAL_BALANCED_DESIGN_WINNER_ENTROPY:
        raise ValueError(
            "V6 candidate bank lacks the required independent winner-diversity evidence: "
            f"entropy={entropy}, minimum={MINIMUM_GOAL_BALANCED_DESIGN_WINNER_ENTROPY}"
        )
    forbidden_outcome_fields = ("realized_terminal_state", "real_task_cost", "realized_terminal_state_ref")
    leaked = {
        str(row["instance_id"]): [field for field in forbidden_outcome_fields if row.get(field) is not None]
        for row in outcome_rows
        if any(row.get(field) is not None for field in forbidden_outcome_fields)
    }
    if leaked:
        raise ValueError(f"outcome terminal labels must remain sealed until policy and certificate freeze: {leaked}")
    return {
        "train_root_count": len(train), "calibration_root_count": len(calibration), "outcome_root_count": len(outcome),
        "root_disjoint": True, "outcome_labels_sealed_before_execution": True,
        "candidate_generalization_regime": context_metadata["candidate_generalization_regime"],
        "candidate_design": context_metadata["candidate_design"],
        "frozen_preflight_ledger_sha256": preflight["sha256"],
        "frozen_preflight_ledger_context_count": int(preflight.get("context_count", 0)),
        "goal_balanced_design_winner_entropy_normalized": float(entropy),
        "minimum_goal_balanced_design_winner_entropy_normalized": MINIMUM_GOAL_BALANCED_DESIGN_WINNER_ENTROPY,
    }


def _assert_action_summary_collision_contract(
    train_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], *,
    minimum_pairs_per_root: int,
) -> dict[str, Any]:
    """Require the predeclared temporal-action information stressor.

    This check reads only candidate actions and identities.  It deliberately
    runs before model fitting and before outcome materialization, so a result
    cannot be attributed to ordered-action information unless the competing
    summary representation truly aliases distinct candidate sequences.
    """

    if minimum_pairs_per_root < 1:
        raise ValueError("minimum summary collision pairs per root must be positive")
    report: dict[str, dict[str, int]] = {}
    for split, rows in (("train", train_rows), ("calibration", calibration_rows), ("outcome", outcome_rows)):
        grouped = _group_rows_by_root(rows)
        for root, group in grouped.items():
            collisions = 0
            for left_index, left in enumerate(group):
                left_summary = _canonical_action_summary(left)
                left_action = np.asarray(left["action_sequence"], dtype=np.float64)
                for right in group[left_index + 1:]:
                    right_action = np.asarray(right["action_sequence"], dtype=np.float64)
                    if (
                        not np.array_equal(left_action, right_action)
                        and np.array_equal(left_summary, _canonical_action_summary(right))
                    ):
                        collisions += 1
            report[f"{split}/{root}"] = {
                "candidate_count": len(group), "summary_collision_pair_count": collisions,
            }
    deficient = {
        root: row["summary_collision_pair_count"]
        for root, row in report.items()
        if row["summary_collision_pair_count"] < minimum_pairs_per_root
    }
    if deficient:
        raise ValueError(
            "V6 action-representation stress contract requires summary-colliding action pairs per root: "
            f"minimum={minimum_pairs_per_root}, deficient={deficient}"
        )
    return {
        "representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "minimum_summary_collision_pairs_per_root": int(minimum_pairs_per_root),
        "checked_root_count": len(report),
        "per_root": report,
    }


def _assert_summary_prediction_alias_contract(
    grouped: dict[str, list[dict[str, Any]]],
    summary_cost_predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    summary_endpoint_predictor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Prove the summary controls remain action-aliased before outcomes open.

    Equal canonical features should yield equal predictions.  Checking this at
    the predictor boundary rules out an accidental downstream candidate ID or
    action-sequence side channel in the descriptive controls.  No terminal
    labels or task costs are accepted by this function.
    """

    per_root: dict[str, dict[str, Any]] = {}
    violations: dict[str, list[dict[str, Any]]] = {}
    for root, rows in sorted(grouped.items()):
        feature = np.stack([_feature_canonical_action_summary(row) for row in rows])
        goal = np.asarray([row["goal_raw"] for row in rows], dtype=np.float64)
        cost_score, _effect = summary_cost_predictor(feature)
        endpoint_score, _endpoint = summary_endpoint_predictor(feature, goal)
        cost_score, endpoint_score = np.asarray(cost_score, dtype=np.float64), np.asarray(endpoint_score, dtype=np.float64)
        if cost_score.shape != (len(rows),) or endpoint_score.shape != (len(rows),):
            raise ValueError(f"{root}: summary predictors must return one score per candidate")
        if not np.isfinite(cost_score).all() or not np.isfinite(endpoint_score).all():
            raise ValueError(f"{root}: summary predictors returned non-finite scores")
        collision_count = 0
        max_cost_delta = max_endpoint_delta = 0.0
        root_violations: list[dict[str, Any]] = []
        for left_index, left in enumerate(rows):
            left_action = np.asarray(left["action_sequence"], dtype=np.float64)
            left_summary = _canonical_action_summary(left)
            for right_index, right in enumerate(rows[left_index + 1:], start=left_index + 1):
                if np.array_equal(left_action, np.asarray(right["action_sequence"], dtype=np.float64)):
                    continue
                if not np.array_equal(left_summary, _canonical_action_summary(right)):
                    continue
                collision_count += 1
                cost_delta = float(abs(cost_score[left_index] - cost_score[right_index]))
                endpoint_delta = float(abs(endpoint_score[left_index] - endpoint_score[right_index]))
                max_cost_delta, max_endpoint_delta = max(max_cost_delta, cost_delta), max(max_endpoint_delta, endpoint_delta)
                if cost_delta > 1.0e-12 or endpoint_delta > 1.0e-12:
                    root_violations.append({
                        "candidate_ids": [str(left["candidate_id"]), str(right["candidate_id"])],
                        "summary_cost_delta": cost_delta, "summary_endpoint_delta": endpoint_delta,
                    })
        per_root[root] = {
            "summary_collision_pair_count": collision_count,
            "maximum_summary_cost_score_delta": max_cost_delta,
            "maximum_summary_endpoint_score_delta": max_endpoint_delta,
        }
        if root_violations:
            violations[root] = root_violations
    if violations:
        raise ValueError(f"summary controls violate canonical action aliasing: {violations}")
    return {
        "checked_before_outcome_execution": True,
        "score_tolerance": 1.0e-12,
        "root_count": len(per_root),
        "per_root": per_root,
    }


def _evaluate(
    grouped: dict[str, list[dict[str, Any]],], costs: dict[str, np.ndarray],
    cost_predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    planning_predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    endpoint_predictor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    summary_cost_predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    summary_endpoint_predictor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    feature_fn: Callable[[dict[str, Any]], np.ndarray] = _feature_ordered_action_sequence,
    summary_feature_fn: Callable[[dict[str, Any]], np.ndarray] = _feature_canonical_action_summary,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for root, rows in grouped.items():
        feature = np.stack([feature_fn(row) for row in rows])
        summary_feature = np.stack([summary_feature_fn(row) for row in rows])
        goal = np.asarray([row["goal_raw"] for row in rows], dtype=np.float64)
        cost_score, _cost_effect = cost_predictor(feature)
        planning_score, _planning_effect = planning_predictor(feature)
        endpoint_score, _endpoint = endpoint_predictor(feature, goal)
        summary_cost_score, _summary_cost_effect = summary_cost_predictor(summary_feature)
        summary_endpoint_score, _summary_endpoint = summary_endpoint_predictor(summary_feature, goal)
        cost_choice = _choice(rows, cost_score)
        planning_choice = _choice(rows, planning_score)
        endpoint_choice = _choice(rows, endpoint_score)
        summary_cost_choice = _choice(rows, summary_cost_score)
        summary_endpoint_choice = _choice(rows, summary_endpoint_score)
        costs_root = costs[root]
        report[root] = {
            "candidate_ids": [str(row["candidate_id"]) for row in rows],
            "realized_candidate_cost": costs_root.tolist(),
            "cost_predicted_cost": np.asarray(cost_score, dtype=np.float64).tolist(),
            "planning_predicted_cost": np.asarray(planning_score, dtype=np.float64).tolist(),
            "endpoint_predicted_cost": np.asarray(endpoint_score, dtype=np.float64).tolist(),
            "summary_cost_predicted_cost": np.asarray(summary_cost_score, dtype=np.float64).tolist(),
            "summary_endpoint_predicted_cost": np.asarray(summary_endpoint_score, dtype=np.float64).tolist(),
            "cost_candidate_id": str(rows[cost_choice]["candidate_id"]),
            "planning_candidate_id": str(rows[planning_choice]["candidate_id"]),
            "endpoint_candidate_id": str(rows[endpoint_choice]["candidate_id"]),
            "summary_cost_candidate_id": str(rows[summary_cost_choice]["candidate_id"]),
            "summary_endpoint_candidate_id": str(rows[summary_endpoint_choice]["candidate_id"]),
            "cost_regret": _regret(costs_root, cost_choice),
            "planning_regret": _regret(costs_root, planning_choice),
            "endpoint_regret": _regret(costs_root, endpoint_choice),
            "summary_cost_regret": _regret(costs_root, summary_cost_choice),
            "summary_endpoint_regret": _regret(costs_root, summary_endpoint_choice),
            "endpoint_disagrees_with_cost_pre_outcome": bool(cost_choice != endpoint_choice),
            "endpoint_beats_cost": bool(_regret(costs_root, endpoint_choice) + 1.0e-10 < _regret(costs_root, cost_choice)),
        }
    return report


def _summary(rows: dict[str, dict[str, Any]], *, deployed: str | None = None) -> dict[str, Any]:
    if not rows:
        raise ValueError("summary requires at least one root")
    disagreement = np.asarray([row["endpoint_disagrees_with_cost_pre_outcome"] for row in rows.values()], dtype=bool)
    result = {
        "root_count": len(rows),
        "endpoint_disagreement_count": int(disagreement.sum()),
        "endpoint_disagreement_coverage": float(disagreement.mean()),
        "mean_regret": {
            arm: float(np.mean([row[f"{arm}_regret"] for row in rows.values()]))
            for arm in ("summary_cost", "cost", "planning", "summary_endpoint", "endpoint")
        },
    }
    if disagreement.any():
        result["disagreement_stratum"] = {
            "root_count": int(disagreement.sum()),
            "mean_endpoint_minus_cost_regret": float(np.mean([
                row["endpoint_regret"] - row["cost_regret"]
                for row in rows.values() if row["endpoint_disagrees_with_cost_pre_outcome"]
            ])),
            "endpoint_better_count": int(sum(
                row["endpoint_beats_cost"] for row in rows.values() if row["endpoint_disagrees_with_cost_pre_outcome"]
            )),
        }
    else:
        result["disagreement_stratum"] = None
    if deployed is not None:
        if deployed not in {"cost", "endpoint"}:
            raise ValueError("deployment must be cost or endpoint")
        result["authorized_deployment_policy"] = deployed
        result["authorized_mean_regret"] = result["mean_regret"][deployed]
    return result


def _factorial_summary(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Describe the frozen 2x2 representation/readout comparison.

    These contrasts are diagnostic only.  The authorization policy below still
    depends exclusively on the ordered endpoint-versus-cost comparison, so no
    outcome observation can select an arm by this summary.
    """

    if not rows:
        raise ValueError("factorial summary requires at least one root")

    def mean(name: str) -> float:
        return float(np.mean([row[f"{name}_regret"] for row in rows.values()]))

    summary_cost, ordered_cost = mean("summary_cost"), mean("cost")
    summary_endpoint, ordered_endpoint = mean("summary_endpoint"), mean("endpoint")
    return {
        "summary_cost_mean_regret": summary_cost,
        "ordered_cost_mean_regret": ordered_cost,
        "summary_endpoint_mean_regret": summary_endpoint,
        "ordered_endpoint_mean_regret": ordered_endpoint,
        "ordered_minus_summary_cost_regret": ordered_cost - summary_cost,
        "ordered_minus_summary_endpoint_regret": ordered_endpoint - summary_endpoint,
        "endpoint_minus_cost_summary_regret": summary_endpoint - summary_cost,
        "endpoint_minus_cost_ordered_regret": ordered_endpoint - ordered_cost,
        "readout_by_representation_interaction": (
            (ordered_endpoint - ordered_cost) - (summary_endpoint - summary_cost)
        ),
    }


def _partition(rows: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Separate arm selection, arm certification, and contrast calibration."""

    selector = {root: row for root, row in rows.items() if _calibration_role(root) == 0}
    certification = {root: row for root, row in rows.items() if _calibration_role(root) == 1}
    risk_calibration = {root: row for root, row in rows.items() if _calibration_role(root) == 2}
    if not selector or not certification or not risk_calibration:
        raise ValueError("calibration must populate selector, certification, and risk-calibration roots")
    return selector, certification, risk_calibration


def _policy(rows: dict[str, dict[str, Any]], *, seed: int, minimum_disagreement_roots: int = MINIMUM_DISAGREEMENT_ROOTS) -> dict[str, Any]:
    """Select endpoint only with stratum support and a disjoint paired certificate."""

    selector, certification, risk_calibration = _partition(rows)
    selector_summary, certification_summary = _summary(selector), _summary(certification)
    selector_delta = float(np.mean([row["endpoint_regret"] - row["cost_regret"] for row in selector.values()]))
    certification_delta = float(np.mean([row["endpoint_regret"] - row["cost_regret"] for row in certification.values()]))
    selector_support = int(selector_summary["endpoint_disagreement_count"])
    certification_support = int(certification_summary["endpoint_disagreement_count"])
    endpoint_selected = bool(selector_support >= minimum_disagreement_roots and selector_delta < 0.0)
    endpoint_certified = False
    upper: float | None = None
    if endpoint_selected and certification_support >= minimum_disagreement_roots:
        candidate = np.asarray([row["endpoint_regret"] for row in certification.values()], dtype=np.float64)
        baseline = np.asarray([row["cost_regret"] for row in certification.values()], dtype=np.float64)
        upper = _upper_bootstrap_delta(candidate, baseline, seed=int(seed) + 701)
        endpoint_certified = bool(certification_delta < 0.0 and upper <= 0.0)
    return {
        "selector_root_count": len(selector), "certification_root_count": len(certification),
        "risk_calibration_root_count": len(risk_calibration),
        "minimum_disagreement_roots": int(minimum_disagreement_roots),
        "selector_summary": selector_summary, "certification_summary": certification_summary,
        "selector_endpoint_minus_cost_delta": selector_delta,
        "certification_endpoint_minus_cost_delta": certification_delta,
        "certification_upper_95_endpoint_minus_cost_delta": upper,
        "endpoint_selected": endpoint_selected, "endpoint_certified": endpoint_certified,
        "deployment_policy": "endpoint" if endpoint_certified else "cost",
    }


def _contrast_arrays(rows: dict[str, dict[str, Any]], arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten one declared fixed candidate bank for the HTSC helper."""

    real, predicted, roots = [], [], []
    for root, row in sorted(rows.items()):
        actual = np.asarray(row["realized_candidate_cost"], dtype=np.float64)
        score = np.asarray(row[f"{arm}_predicted_cost"], dtype=np.float64)
        if actual.shape != score.shape or actual.ndim != 1 or len(actual) < 2:
            raise ValueError(f"{root}: candidate costs and predictions must align with at least two candidates")
        real.extend(actual.tolist())
        predicted.extend(score.tolist())
        roots.extend([root] * len(actual))
    return np.asarray(real, dtype=np.float64), np.asarray(predicted, dtype=np.float64), np.asarray(roots, dtype=str)


def _freeze_contrast_certificate(
    rows: dict[str, dict[str, Any]], *, arm: str, alpha: float,
) -> dict[str, Any]:
    real, predicted, roots = _contrast_arrays(rows, arm)
    return calibrate_contrast_bound(real, predicted, roots, sorted(rows), alpha=alpha)


def _evaluate_contrast_certificate(
    rows: dict[str, dict[str, Any]], *, arm: str, calibration: dict[str, Any], support_passed: bool,
) -> dict[str, Any]:
    """Evaluate one frozen fixed-bank certificate only after outcome execution."""

    bound = calibration.get("bound")
    per_root = {}
    released_regret: list[float] = []
    for root, row in sorted(rows.items()):
        certificate = certify_candidate_bank(
            np.asarray(row[f"{arm}_predicted_cost"], dtype=np.float64), bound,
            support_passed=support_passed, candidate_ids=row["candidate_ids"], candidate_bank_mode="fixed",
        )
        per_root[root] = certificate
        if certificate["status"] == "permit":
            released_regret.append(float(row[f"{arm}_regret"]))
    real, predicted, roots = _contrast_arrays(rows, arm)
    coverage = evaluate_contrast_bound(real, predicted, roots, bound)
    return {
        "arm": arm, "calibration": calibration, "support_passed": bool(support_passed),
        "per_root": per_root, "outcome_coverage": coverage,
        "released_root_count": len(released_regret), "released_root_coverage": float(len(released_regret) / len(rows)),
        "released_mean_regret": None if not released_regret else float(np.mean(released_regret)),
        "claim_boundary": "fixed declared candidate-bank marginal contrast certificate; release is not conditional state-wise coverage or an online-planning guarantee.",
    }


def run(*, source_dataset: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path,
        device: str = "cuda", seed: int = 202610340, width: int = 32, steps: int = 400, ensemble_size: int = 4,
        risk_alpha: float = 0.1, minimum_summary_collision_pairs_per_root: int = MINIMUM_SUMMARY_COLLISION_PAIRS_PER_ROOT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive")
    source, context_payload, calibration = _load(source_dataset), _load(context), _load(calibration_labels)
    train_rows = [dict(row) for row in source.get("records", []) if str(row.get("split")) == "train"]
    calibration_rows = [dict(row) for row in context_payload.get("records", []) if str(row.get("split")) == "calibration"]
    outcome_rows = [dict(row) for row in context_payload.get("records", []) if str(row.get("split")) == "outcome"]
    if not train_rows or not calibration_rows or not outcome_rows:
        raise ValueError("source/context require non-empty train, calibration, and outcome records")
    split_contract = _assert_split_contract(train_rows, calibration_rows, outcome_rows, context_payload.get("metadata"))
    action_representation_stress = _assert_action_summary_collision_contract(
        train_rows, calibration_rows, outcome_rows,
        minimum_pairs_per_root=int(minimum_summary_collision_pairs_per_root),
    )
    cuda = require_cuda_allocation(device)
    with np.load(source_arrays, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    train_terminal = np.stack([_terminal_from_ref(row, arrays) for row in train_rows])
    seeds = tuple(int(seed) + offset for offset in range(int(ensemble_size)))
    representation = "full_ordered_20x2"
    summary_representation = "canonical_summary_mean_std_first_last"
    summary_cost_contract, summary_cost_predictor = _fit_ensemble(
        train_rows, train_terminal, COST_ONLY, seeds=seeds, width=width, steps=steps, device=device,
        feature_fn=_feature_canonical_action_summary, action_representation=summary_representation,
    )
    summary_endpoint_contract, summary_endpoint_predictor = _fit_endpoint_ensemble(
        train_rows, train_terminal, seeds=seeds, width=width, steps=steps, device=device,
        feature_fn=_feature_canonical_action_summary, action_representation=summary_representation,
    )
    cost_contract, cost_predictor = _fit_ensemble(
        train_rows, train_terminal, COST_ONLY, seeds=seeds, width=width, steps=steps, device=device,
        feature_fn=_feature_ordered_action_sequence, action_representation=representation,
    )
    planning_contract, planning_predictor = _fit_ensemble(
        train_rows, train_terminal, PLANNING_SHAPED, seeds=seeds, width=width, steps=steps, device=device,
        feature_fn=_feature_ordered_action_sequence, action_representation=representation,
    )
    endpoint_contract, endpoint_predictor = _fit_endpoint_ensemble(
        train_rows, train_terminal, seeds=seeds, width=width, steps=steps, device=device,
        feature_fn=_feature_ordered_action_sequence, action_representation=representation,
    )
    for contract in (summary_cost_contract, summary_endpoint_contract):
        _enforce_arm_contract(contract, expected_representation=summary_representation)
    for contract in (cost_contract, planning_contract, endpoint_contract):
        _enforce_arm_contract(contract, expected_representation=representation)

    # This is deliberately computed before outcome terminal states are
    # materialized. It proves that the canonical-summary controls cannot rank
    # different interior-pulse permutations by an accidental side channel.
    summary_prediction_aliasing_pre_outcome = {
        "calibration": _assert_summary_prediction_alias_contract(
            _group_rows_by_root(calibration_rows), summary_cost_predictor, summary_endpoint_predictor,
        ),
        "outcome": _assert_summary_prediction_alias_contract(
            _group_rows_by_root(outcome_rows), summary_cost_predictor, summary_endpoint_predictor,
        ),
    }
    endpoint_ensemble_diagnostics_pre_outcome = {
        "calibration": _endpoint_ensemble_diagnostics(_group_rows_by_root(calibration_rows), endpoint_predictor),
        "outcome": _endpoint_ensemble_diagnostics(_group_rows_by_root(outcome_rows), endpoint_predictor),
    }

    calibration_terminal = _calibration_terminal(calibration_rows, calibration)
    calibration_report = _evaluate(
        _group_rows_by_root(calibration_rows), _cost_map(calibration_rows, calibration_terminal),
        cost_predictor, planning_predictor, endpoint_predictor,
        summary_cost_predictor, summary_endpoint_predictor,
        _feature_ordered_action_sequence, _feature_canonical_action_summary,
    )
    policy = _policy(calibration_report, seed=seed)
    _selector, _certificate, risk_rows = _partition(calibration_report)
    deployment = str(policy["deployment_policy"])
    contrast_calibration = _freeze_contrast_certificate(risk_rows, arm=deployment, alpha=float(risk_alpha))
    support_passed = True

    # Do not materialize outcome labels until all models, eligibility, and the
    # independent arm and contrast certificates are frozen.
    outcome_terminal = _fresh_terminal(outcome_rows)
    outcome_report = _evaluate(
        _group_rows_by_root(outcome_rows), _cost_map(outcome_rows, outcome_terminal),
        cost_predictor, planning_predictor, endpoint_predictor,
        summary_cost_predictor, summary_endpoint_predictor,
        _feature_ordered_action_sequence, _feature_canonical_action_summary,
    )
    outcome_certificate = _evaluate_contrast_certificate(
        outcome_report, arm=deployment, calibration=contrast_calibration, support_passed=support_passed,
    )
    result = {
        "schema": SCHEMA, "status": "completed_counterfactual_endpoint_control",
        "claim_boundary": "PointMaze visual-history/action endpoint-readout control; not a PlanWAM replication, online RGB deployment, or universal counterfactual certificate.",
        "control_contract": {
            "label": CONTROL_LABEL, "candidate_score_is_terminal_endpoint_distance": True,
            "raw_source_state_live_input_forbidden": True, "outcome_execution_deferred_until_policy_and_contrast_bound_frozen": True,
            "legacy_auxiliary_planning_arm_descriptive_only": True,
            "endpoint_ensemble_disagreement_diagnostic_only": True,
            "action_representation": representation,
            "primary_authorization_holds_ordered_action_representation_fixed": True,
            "factorial_diagnostic": {
                "summary_cost": "summary representation plus direct cost head",
                "ordered_cost": "ordered representation plus direct cost head; primary incumbent",
                "summary_endpoint": "summary representation plus endpoint-coupled readout",
                "ordered_endpoint": "ordered representation plus endpoint-coupled readout; primary repair",
                "selection_uses_ordered_endpoint_vs_ordered_cost_only": True,
            },
        },
        "split_isolation": {
            **split_contract,
            "calibration_roles": {
                "selector": int(policy["selector_root_count"]), "arm_certificate": int(policy["certification_root_count"]),
                "risk_calibration": int(policy["risk_calibration_root_count"]),
            },
        },
        "action_representation_stress": action_representation_stress,
        "summary_prediction_aliasing_pre_outcome": summary_prediction_aliasing_pre_outcome,
        "endpoint_ensemble_diagnostics_pre_outcome": endpoint_ensemble_diagnostics_pre_outcome,
        "arms": {
            "summary_cost_only": summary_cost_contract,
            "cost_only": cost_contract,
            "planning_shaped_auxiliary": planning_contract,
            "summary_counterfactual_endpoint": summary_endpoint_contract,
            "counterfactual_endpoint": endpoint_contract,
        },
        "calibration": {
            "policy": policy, "per_root": calibration_report,
            "factorial_diagnostic": _factorial_summary(calibration_report),
        },
        "outcome": {
            "authorized_deployment_policy": deployment,
            "authorized_deployment_summary": _summary(outcome_report, deployed=deployment),
            "descriptive_summary": _summary(outcome_report),
            "factorial_diagnostic": _factorial_summary(outcome_report),
            "per_root": outcome_report,
        },
        "post_repair_contrast_certificate": outcome_certificate,
        "post_execution_outcome_labels": _post_execution_labels(outcome_rows, outcome_terminal),
        "cuda_preflight": cuda,
        "provenance": {
            "source_dataset": str(source_dataset.resolve()), "source_dataset_sha256": _sha256(source_dataset),
            "source_arrays": str(source_arrays.resolve()), "source_arrays_sha256": _sha256(source_arrays),
            "context": str(context.resolve()), "context_sha256": _sha256(context),
            "calibration_labels": str(calibration_labels.resolve()), "calibration_labels_sha256": _sha256(calibration_labels),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--source-arrays", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--calibration-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=202610340)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--risk-alpha", type=float, default=0.1)
    parser.add_argument("--minimum-summary-collision-pairs-per-root", type=int, default=MINIMUM_SUMMARY_COLLISION_PAIRS_PER_ROOT)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"status": result["status"], "outcome": result["outcome"]["authorized_deployment_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
