"""Execute V10's prospective routing between a fallback and a real repair arm.

The endpoint arm predicts a candidate-conditioned terminal state and derives a
goal cost from that prediction.  It is therefore an actual counterfactual
repair, unlike the earlier repair-family audit that only emitted a label.  A
small router is trained from *cross-fitted train-root* regret advantages and
may release endpoint only on roots where endpoint and cost choose different
candidates.  Calibration freezes the threshold and independently certifies
all-root non-inferiority versus the direct cost fallback before outcome labels
are materialized.

This is an offline fixed-bank repairability intervention.  It does not make a
conditional coverage, online MPC, universal-world-model, or causal guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control import _fit_endpoint_ensemble
from experiments.gdd_wm.run_pointmaze_planning_shaped_control import (
    _calibration_terminal,
    _cost,
    _feature_ordered_action_sequence,
    _fresh_terminal,
    _group_rows,
    _sha256,
    _terminal_from_ref,
)
from experiments.gdd_wm.run_pointmaze_planning_shaped_repairability import (
    COST_ONLY,
    V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE,
    V9_MIN_TRAIN_OPPORTUNITY_ROOTS,
    V9_ROUTER_FEATURE_CONTRACT,
    V9_ROUTER_FEATURE_DIMENSION,
    _fit_ensemble,
    _fit_selected_ridge_advantage,
    _opportunity_compact_features,
    _upper_bootstrap_delta,
)
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_endpoint_repairability_router_v10.v1"
PREFLIGHT_SCHEMA = panel.ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA
CONTROL_LABEL = "opportunity_gated_endpoint_coupled_counterfactual_repair"
ACTION_REPRESENTATION = "full_ordered_20x2"
MINIMUM_RELEASE_COVERAGE = 0.10


class _CalibrationOpportunityError(ValueError):
    """A valid pre-outcome stop, distinct from an implementation failure."""

    def __init__(self, *, selector_root_count: int, certification_root_count: int,
                 selector_opportunity_count: int, certification_opportunity_count: int,
                 minimum_opportunity_count: int) -> None:
        super().__init__(
            "V10 calibration lacks the predeclared opportunity count in a selector/certificate role"
        )
        self.details = {
            "selector_root_count": int(selector_root_count),
            "certification_root_count": int(certification_root_count),
            "selector_opportunity_root_count": int(selector_opportunity_count),
            "certification_opportunity_root_count": int(certification_opportunity_count),
            "minimum_opportunity_roots_per_role": int(minimum_opportunity_count),
        }


class _CrossfitOpportunityError(ValueError):
    """A valid train-only stop when no router can be fit at declared capacity."""

    def __init__(self, *, train_root_count: int, opportunity_count: int,
                 minimum_opportunity_count: int, mean_cost_minus_endpoint_regret: float) -> None:
        super().__init__("V10 lacks the predeclared minimum cross-fitted cost/endpoint disagreement roots")
        self.details = {
            "train_root_count": int(train_root_count),
            "crossfit_opportunity_root_count": int(opportunity_count),
            "minimum_crossfit_opportunity_root_count": int(minimum_opportunity_count),
            "crossfit_scorer_disagreement_rate": float(opportunity_count / train_root_count),
            "mean_cost_minus_endpoint_regret": float(mean_cost_minus_endpoint_regret),
        }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _root_fold(root: str) -> int:
    return int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:8], 16) % 4


def _calibration_role(root: str) -> int:
    return int(hashlib.sha256(f"v10/calibration/{root}".encode("utf-8")).hexdigest()[:8], 16) % 2


def _choice(rows: list[dict[str, Any]], score: np.ndarray) -> int:
    values = np.asarray(score, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("candidate score must be finite and align with its group")
    return min(range(len(rows)), key=lambda index: (float(values[index]), str(rows[index]["candidate_id"])))


def _regret(cost: np.ndarray, choice: int) -> float:
    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("realized candidate cost must be a finite non-empty vector")
    return float(values[int(choice)] - values.min())


def _root_costs(rows: list[dict[str, Any]], terminal: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, state in zip(rows, np.asarray(terminal, dtype=np.float64)):
        grouped[str(row["instance_id"])].append(float(_cost([row], np.asarray(state).reshape(1, -1))[0]))
    return {root: np.asarray(value, dtype=np.float64) for root, value in grouped.items()}


def _partition_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    # The shared helper deliberately returns row *indices* for model-loss
    # construction.  Convert them at this boundary: downstream routing needs
    # full candidate records, not integer offsets.  Keeping this explicit
    # prevents a silent candidate-ID/order side channel in V10 evaluation.
    grouped_indices = _group_rows(rows)
    if not grouped_indices or any(len(group) < 2 for group in grouped_indices.values()):
        raise ValueError("each root requires at least two candidates")
    return {
        str(root): [dict(rows[int(index)]) for index in indices]
        for root, indices in grouped_indices.items()
    }


def _validate_candidate_action_records(rows: list[dict[str, Any]], *, name: str) -> set[str]:
    """Bind declared candidate identities to the actual ordered action bytes."""

    values: set[str] = set()
    for index, row in enumerate(rows):
        declared = row.get("candidate_action_hash")
        action = np.asarray(row.get("action_sequence"), dtype=np.float64)
        if not isinstance(declared, str) or not declared or action.shape != (20, 2) or not np.isfinite(action).all():
            raise ValueError(f"V10 {name} row {index} lacks a finite ordered action and identity hash")
        observed = panel.sequence_hash(action)
        if declared != observed:
            raise ValueError(f"V10 {name} row {index} candidate_action_hash does not match action_sequence")
        values.add(declared)
    if not values:
        raise ValueError(f"V10 {name} has no candidate actions")
    return values


def _split_root_identities(rows: list[dict[str, Any]], *, allowed_splits: set[str], name: str) -> dict[str, set[int]]:
    """Read root identities from records rather than trusting split counts.

    A V10 ledger freezes the *particular* state roots on which each role is
    allowed to operate.  Matching 192/128/64 counts alone would permit a
    substituted panel with equally sized but outcome-informed roots.
    """

    result = {split: set() for split in allowed_splits}
    for index, row in enumerate(rows):
        split = row.get("split")
        if split not in result:
            raise ValueError(f"V10 {name} row {index} has an unexpected split {split!r}")
        try:
            root = int(row.get("start_episode_id"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"V10 {name} row {index} lacks an integer start_episode_id") from error
        result[str(split)].add(root)
    if any(not roots for roots in result.values()):
        raise ValueError(f"V10 {name} lacks records for a required split")
    return result


def _validate_v10_contract(source: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Bind execution to V10's static ledger and exact supported action bank."""

    metadata = context.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("V10 requires context metadata")
    required_metadata = {
        "candidate_generalization_regime": "shared_candidate_in_support",
        "candidate_design": "goal_balanced_interaction_pairs",
        "scorer_candidate_bank_protocol": "held_exact_shared_bank",
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise ValueError("V10 context is not the declared shared-support goal-balanced protocol")
    source_rows, held_rows = source.get("records"), context.get("records")
    if not isinstance(source_rows, list) or not isinstance(held_rows, list):
        raise ValueError("V10 source/context records are missing")
    train_hashes = _validate_candidate_action_records(source_rows, name="source")
    held_hashes = _validate_candidate_action_records(held_rows, name="context")
    if train_hashes != held_hashes:
        raise ValueError("V10 scorer training and held candidate banks must be exactly identical")
    ledger_receipt = metadata.get("frozen_preflight_ledger")
    if not isinstance(ledger_receipt, dict) or ledger_receipt.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("V10 requires its matching static-preflight provenance receipt")
    ledger_path = ledger_receipt.get("path")
    if not isinstance(ledger_path, str) or not Path(ledger_path).is_file():
        raise ValueError("V10 preflight provenance receipt lacks a readable immutable ledger")
    receipt_sha256 = ledger_receipt.get("sha256")
    if not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64 or _sha256(Path(ledger_path)) != receipt_sha256:
        raise ValueError("V10 preflight provenance receipt SHA-256 does not bind the readable ledger")
    ledger = _load(Path(ledger_path))
    bank = ledger.get("candidate_bank")
    if (
        ledger.get("schema") != PREFLIGHT_SCHEMA
        or ledger.get("status") != "passed_static_preflight"
        or not isinstance(bank, dict)
        or bank.get("shared_candidate_in_support") is not True
        or bank.get("held_action_hash_count") != len(held_hashes)
        or bank.get("held_train_action_hash_overlap_count") != len(held_hashes)
        or bank.get("scorer_candidate_bank_protocol") != "held_exact_shared_bank"
        or bank.get("scorer_training_action_hash_count") != len(held_hashes)
        or bank.get("router_feature_contract") != V9_ROUTER_FEATURE_CONTRACT
        or bank.get("router_feature_dimension") != V9_ROUTER_FEATURE_DIMENSION
        or bank.get("minimum_router_train_root_surplus", -1) < 16
        or bank.get("router_kind") != "advantage_ridge"
        or bank.get("router_action_representation") != ACTION_REPRESENTATION
        or bank.get("router_training_scope") != "scorer_disagreement_only"
        or bank.get("opportunity_gate") != "cost_endpoint_candidate_disagreement"
        or bank.get("minimum_router_opportunity_root_count") != V9_MIN_TRAIN_OPPORTUNITY_ROOTS
        or bank.get("minimum_calibration_opportunity_roots_per_role") != V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE
        or bank.get("repair_arm") != "endpoint_coupled_counterfactual"
        or ledger.get("post_execution_aliasing_contract") != panel.V7_POST_EXECUTION_ALIASING_CONTRACT
    ):
        raise ValueError("V10 static preflight lacks the declared support, capacity, or repair-arm contract")
    train_root_count = len({str(row.get("instance_id", "")) for row in source_rows if row.get("split") == "train"})
    if (
        bank.get("router_train_root_count") != train_root_count
        or train_root_count < V9_ROUTER_FEATURE_DIMENSION + int(bank["minimum_router_train_root_surplus"])
    ):
        raise ValueError("V10 static preflight train-root count differs from execution input")
    declared_split_roots = ledger.get("split_roots")
    if not isinstance(declared_split_roots, dict) or set(declared_split_roots) != {"train", "calibration", "outcome"}:
        raise ValueError("V10 static preflight lacks exact split-root identities")
    try:
        expected_split_roots = {
            split: {int(root) for root in declared_split_roots[split]}
            for split in ("train", "calibration", "outcome")
        }
    except (TypeError, ValueError) as error:
        raise ValueError("V10 static preflight split-root identities must be integers") from error
    if (
        any(len(expected_split_roots[split]) != int(bank["router_train_root_count"] if split == "train" else ledger["root_counts"][split])
            for split in expected_split_roots)
        or len(set().union(*expected_split_roots.values())) != sum(len(roots) for roots in expected_split_roots.values())
    ):
        raise ValueError("V10 static preflight split-root identities are incomplete or overlapping")
    observed_source = _split_root_identities(source_rows, allowed_splits={"train"}, name="source")
    observed_context = _split_root_identities(held_rows, allowed_splits={"calibration", "outcome"}, name="context")
    observed_split_roots = {
        "train": observed_source["train"],
        "calibration": observed_context["calibration"],
        "outcome": observed_context["outcome"],
    }
    if observed_split_roots != expected_split_roots:
        raise ValueError("V10 execution split-root identities differ from the immutable static-preflight ledger")
    expected_role_counts = ledger.get("calibration_role_counts")
    role_names = ("selector", "certification")
    if not isinstance(expected_role_counts, dict) or set(expected_role_counts) != set(role_names):
        raise ValueError("V10 static preflight lacks calibration role-count binding")
    actual_role_counts = {name: 0 for name in role_names}
    calibration_instance_ids = {str(row["instance_id"]) for row in held_rows if row.get("split") == "calibration"}
    for root in calibration_instance_ids:
        actual_role_counts[role_names[_calibration_role(root)]] += 1
    if (
        any(not isinstance(expected_role_counts[name], int) for name in role_names)
        or actual_role_counts != expected_role_counts
    ):
        raise ValueError(
            "V10 calibration role partition differs from the immutable static-preflight ledger: "
            f"actual={actual_role_counts}, expected={expected_role_counts}"
        )
    return {
        "verified": True,
        "ledger_path": str(Path(ledger_path).resolve()),
        "held_action_hash_count": len(held_hashes),
        "router_feature_dimension": V9_ROUTER_FEATURE_DIMENSION,
        "router_train_root_count": train_root_count,
        "repair_arm": str(bank["repair_arm"]),
        "declared_root_counts": dict(ledger.get("root_counts", {})),
        "declared_split_roots": {name: sorted(values) for name, values in expected_split_roots.items()},
        "calibration_role_counts": actual_role_counts,
    }


def _fit_models(
    rows: list[dict[str, Any]], terminal: np.ndarray, *, seeds: tuple[int, ...], width: int, steps: int, device: str,
) -> tuple[dict[str, Any], Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], dict[str, Any], Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    cost_contract, cost_predictor = _fit_ensemble(
        rows, terminal, COST_ONLY, seeds=seeds, width=width, steps=steps, device=device,
        feature_fn=_feature_ordered_action_sequence, action_representation=ACTION_REPRESENTATION,
    )
    endpoint_contract, endpoint_predictor = _fit_endpoint_ensemble(
        rows, terminal, seeds=seeds, width=width, steps=steps, device=device,
        feature_fn=_feature_ordered_action_sequence, action_representation=ACTION_REPRESENTATION,
    )
    return cost_contract, cost_predictor, endpoint_contract, endpoint_predictor


def _crossfit_router(
    groups: dict[str, list[dict[str, Any]]], terminals: dict[str, np.ndarray], *, seeds: tuple[int, ...], width: int,
    steps: int, device: str,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any], dict[str, np.ndarray], dict[str, float], dict[str, bool]]:
    features: dict[str, np.ndarray] = {}
    advantage: dict[str, float] = {}
    opportunities: dict[str, bool] = {}
    ordered = sorted(groups)
    for fold in range(4):
        held = [root for root in ordered if _root_fold(root) == fold]
        fit = [root for root in ordered if _root_fold(root) != fold]
        if not held or not fit:
            raise ValueError("V10 scorer cross-fitting produced an empty fold")
        fit_rows = [row for root in fit for row in groups[root]]
        fit_terminal = np.concatenate([terminals[root] for root in fit])
        _cost_contract, cost_predictor, _endpoint_contract, endpoint_predictor = _fit_models(
            fit_rows, fit_terminal, seeds=seeds, width=width, steps=steps, device=device,
        )
        for root in held:
            rows, terminal = groups[root], terminals[root]
            action = np.stack([_feature_ordered_action_sequence(row) for row in rows])
            goal = np.asarray([row["goal_raw"] for row in rows], dtype=np.float64)
            cost_prediction, _cost_effect = cost_predictor(action)
            endpoint_prediction, _endpoint = endpoint_predictor(action, goal)
            cost = _cost(rows, terminal)
            cost_pick, endpoint_pick = _choice(rows, cost_prediction), _choice(rows, endpoint_prediction)
            features[root] = _opportunity_compact_features(rows, cost_prediction, endpoint_prediction)
            advantage[root] = _regret(cost, cost_pick) - _regret(cost, endpoint_pick)
            opportunities[root] = bool(cost_pick != endpoint_pick)
    train_roots = [root for root in ordered if opportunities[root]]
    if len(train_roots) < V9_MIN_TRAIN_OPPORTUNITY_ROOTS:
        raise _CrossfitOpportunityError(
            train_root_count=len(ordered), opportunity_count=len(train_roots),
            minimum_opportunity_count=V9_MIN_TRAIN_OPPORTUNITY_ROOTS,
            mean_cost_minus_endpoint_regret=float(np.mean(list(advantage.values()))),
        )
    matrix = np.stack([features[root] for root in train_roots])
    if matrix.shape != (len(train_roots), V9_ROUTER_FEATURE_DIMENSION):
        raise RuntimeError("V10 compact router feature contract was violated")
    router, selection = _fit_selected_ridge_advantage(
        train_roots, matrix, np.asarray([advantage[root] for root in train_roots], dtype=np.float64),
    )
    return router, selection, features, advantage, opportunities


def _evaluate(
    groups: dict[str, list[dict[str, Any]]], costs: dict[str, np.ndarray], *,
    cost_predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    endpoint_predictor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    router: Callable[[np.ndarray], np.ndarray], threshold: float,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for root, rows in sorted(groups.items()):
        action = np.stack([_feature_ordered_action_sequence(row) for row in rows])
        goal = np.asarray([row["goal_raw"] for row in rows], dtype=np.float64)
        cost_prediction, _cost_effect = cost_predictor(action)
        endpoint_prediction, endpoint = endpoint_predictor(action, goal)
        cost_pick, endpoint_pick = _choice(rows, cost_prediction), _choice(rows, endpoint_prediction)
        feature = _opportunity_compact_features(rows, cost_prediction, endpoint_prediction)
        score = float(np.asarray(router(feature[None, :]), dtype=np.float64).reshape(-1)[0])
        eligible = bool(cost_pick != endpoint_pick)
        released = bool(eligible and score >= float(threshold))
        selected = endpoint_pick if released else cost_pick
        realized = np.asarray(costs[root], dtype=np.float64)
        result[root] = {
            "candidate_ids": [str(row["candidate_id"]) for row in rows],
            "realized_candidate_cost": realized.tolist(),
            "cost_predicted_cost": np.asarray(cost_prediction, dtype=np.float64).tolist(),
            "endpoint_predicted_cost": np.asarray(endpoint_prediction, dtype=np.float64).tolist(),
            "endpoint_predicted_terminal": np.asarray(endpoint, dtype=np.float64).tolist(),
            "cost_candidate_id": str(rows[cost_pick]["candidate_id"]),
            "endpoint_candidate_id": str(rows[endpoint_pick]["candidate_id"]),
            "routed_candidate_id": str(rows[selected]["candidate_id"]),
            "cost_regret": _regret(realized, cost_pick),
            "endpoint_regret": _regret(realized, endpoint_pick),
            "routed_regret": _regret(realized, selected),
            "endpoint_beats_cost": bool(_regret(realized, endpoint_pick) + 1.0e-10 < _regret(realized, cost_pick)),
            "candidate_disagreement": eligible,
            "routing_eligible": eligible,
            "released_endpoint": released,
            "router_score_pre_outcome": score,
            "router_feature": feature.tolist(),
        }
    return result


def _summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("summary requires roots")
    eligible = np.asarray([bool(row["routing_eligible"]) for row in rows.values()], dtype=bool)
    released = np.asarray([bool(row["released_endpoint"]) for row in rows.values()], dtype=bool)
    return {
        "root_count": len(rows),
        "cost_endpoint_disagreement_count": int(eligible.sum()),
        "cost_endpoint_disagreement_rate": float(eligible.mean()),
        "endpoint_release_count": int(released.sum()),
        "endpoint_release_coverage": float(released.mean()),
        "mean_regret": {
            arm: float(np.mean([float(row[f"{arm}_regret"]) for row in rows.values()]))
            for arm in ("cost", "endpoint", "routed")
        },
        "endpoint_better_count_on_opportunities": int(sum(
            bool(row["endpoint_beats_cost"]) for row in rows.values() if row["routing_eligible"]
        )),
    }


def _calibration_roles(rows: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    selector = {root: row for root, row in rows.items() if _calibration_role(root) == 0}
    certification = {root: row for root, row in rows.items() if _calibration_role(root) == 1}
    if not selector or not certification:
        raise ValueError("V10 calibration lacks selector or certificate roots")
    return selector, certification


def _apply_threshold(rows: dict[str, dict[str, Any]], threshold: float) -> dict[str, dict[str, Any]]:
    result = {}
    for root, row in rows.items():
        released = bool(row["routing_eligible"] and float(row["router_score_pre_outcome"]) >= float(threshold))
        result[root] = {
            **row,
            "released_endpoint": released,
            "routed_candidate_id": row["endpoint_candidate_id"] if released else row["cost_candidate_id"],
            "routed_regret": float(row["endpoint_regret"] if released else row["cost_regret"]),
        }
    return result


def _calibrate(rows: dict[str, dict[str, Any]], *, seed: int) -> dict[str, Any]:
    selector, certification = _calibration_roles(rows)
    selector_eligible = {root: row for root, row in selector.items() if row["routing_eligible"]}
    certification_eligible = {root: row for root, row in certification.items() if row["routing_eligible"]}
    minimum = V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE
    if len(selector_eligible) < minimum or len(certification_eligible) < minimum:
        raise _CalibrationOpportunityError(
            selector_root_count=len(selector), certification_root_count=len(certification),
            selector_opportunity_count=len(selector_eligible),
            certification_opportunity_count=len(certification_eligible),
            minimum_opportunity_count=minimum,
        )
    thresholds = sorted({float("inf"), *[float(row["router_score_pre_outcome"]) for row in selector_eligible.values()]})
    candidates = []
    for threshold in thresholds:
        evaluated = _apply_threshold(selector, threshold)
        if _summary(evaluated)["endpoint_release_coverage"] >= MINIMUM_RELEASE_COVERAGE:
            candidates.append((float(threshold), evaluated))
    if not candidates:
        raise ValueError("V10 selector cannot achieve its minimum all-root endpoint release coverage")
    threshold, selected = min(candidates, key=lambda item: (
        _summary(item[1])["mean_regret"]["routed"], -item[0],
    ))
    certified = _apply_threshold(certification, threshold)
    delta = np.asarray([row["routed_regret"] - row["cost_regret"] for row in certified.values()], dtype=np.float64)
    upper = _upper_bootstrap_delta(
        np.asarray([row["routed_regret"] for row in certified.values()], dtype=np.float64),
        np.asarray([row["cost_regret"] for row in certified.values()], dtype=np.float64), seed=int(seed) + 701,
    )
    authorized = bool(float(delta.mean()) < 0.0 and upper <= 0.0)
    return {
        "selector_root_count": len(selector), "certification_root_count": len(certification),
        "selector_opportunity_root_count": len(selector_eligible),
        "certification_opportunity_root_count": len(certification_eligible),
        "minimum_opportunity_roots_per_role": minimum,
        "fixed_incumbent": "cost", "threshold": float(threshold),
        "minimum_all_root_release_coverage": MINIMUM_RELEASE_COVERAGE,
        "selector_summary": _summary(selected), "certification_summary": _summary(certified),
        "certification_mean_routed_minus_cost_regret": float(delta.mean()),
        "certification_upper_95_routed_minus_cost_regret": upper,
        "routing_authorized": authorized,
        "deployment_policy": "routed" if authorized else "cost",
    }


def _deployment(rows: dict[str, dict[str, Any]], policy: str) -> dict[str, dict[str, Any]]:
    if policy not in {"cost", "routed"}:
        raise ValueError("V10 deployment policy must be cost or routed")
    if policy == "routed":
        return rows
    return {
        root: {
            **row,
            "released_endpoint": False,
            "routed_candidate_id": row["cost_candidate_id"],
            "routed_regret": float(row["cost_regret"]),
        }
        for root, row in rows.items()
    }


def _write_report(output: Path, report: dict[str, Any]) -> None:
    """Persist one immutable V10 execution or lawful pre-outcome rejection."""

    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(*, source_dataset: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path,
        device: str = "cuda", seed: int = 202610404, width: int = 32, steps: int = 400,
        ensemble_size: int = 4) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if ensemble_size < 1:
        raise ValueError("V10 ensemble_size must be positive")
    source, context_payload = _load(source_dataset), _load(context)
    contract = _validate_v10_contract(source, context_payload)
    train_rows = [dict(row) for row in source.get("records", []) if row.get("split") == "train"]
    calibration_rows = [dict(row) for row in context_payload.get("records", []) if row.get("split") == "calibration"]
    outcome_rows = [dict(row) for row in context_payload.get("records", []) if row.get("split") == "outcome"]
    train_groups, calibration_groups, outcome_groups = map(_partition_rows, (train_rows, calibration_rows, outcome_rows))
    train_roots, calibration_roots, outcome_roots = map(set, (train_groups, calibration_groups, outcome_groups))
    if not train_roots or not calibration_roots or not outcome_roots or train_roots & calibration_roots or train_roots & outcome_roots or calibration_roots & outcome_roots:
        raise ValueError("V10 requires non-empty root-disjoint train, calibration, and outcome groups")
    actual_root_counts = {"train": len(train_roots), "calibration": len(calibration_roots), "outcome": len(outcome_roots)}
    if contract["declared_root_counts"] != actual_root_counts:
        raise ValueError("V10 execution root counts differ from the immutable static-preflight ledger")
    cuda = require_cuda_allocation(device)
    with np.load(source_arrays, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    train_terminal = np.stack([_terminal_from_ref(row, arrays) for row in train_rows])
    terminal_by_root: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, terminal in zip(train_rows, train_terminal):
        terminal_by_root[str(row["instance_id"])].append(np.asarray(terminal, dtype=np.float64))
    train_terminals = {root: np.stack(values) for root, values in terminal_by_root.items()}
    seeds = tuple(int(seed) + offset for offset in range(int(ensemble_size)))
    try:
        router, router_selection, cross_features, cross_advantage, cross_opportunities = _crossfit_router(
            train_groups, train_terminals, seeds=seeds, width=width, steps=steps, device=device,
        )
    except _CrossfitOpportunityError as error:
        # This is a capacity/identifiability result, not a failed training
        # attempt.  No final scorer, calibration evaluation, or outcome
        # successor may be materialized after this train-only stop.
        rejected = {
            "schema": SCHEMA,
            "status": "rejected_pre_calibration_crossfit_opportunity",
            "claim_boundary": "Train-only V10 opportunity-capacity rejection. It establishes neither a fitted endpoint arm nor a calibration/outcome repair consequence.",
            "control_contract": {
                "label": CONTROL_LABEL, "repair_arm": "endpoint_coupled_counterfactual",
                "fallback_arm": "direct_cost_only", "action_representation": ACTION_REPRESENTATION,
                "same_choice_roots_forced_to_cost": True,
                "outcome_execution_deferred_until_router_threshold_and_certificate_frozen": True,
                "outcome_cost_used_for_router_or_threshold_selection": False,
                "shared_candidate_support_verified": contract,
            },
            "split_isolation": {
                "train_root_count": actual_root_counts["train"], "calibration_root_count": actual_root_counts["calibration"],
                "outcome_root_count": actual_root_counts["outcome"], "root_disjoint": True,
            },
            "crossfit": {"status": "rejected_insufficient_opportunity_support", "rejection": error.details},
            "calibration": {"materialized": False, "reason": "no train-only router passed the opportunity-capacity gate"},
            "outcome": {"materialized": False, "reason": "no frozen router, threshold, or calibration certificate"},
            "cuda_preflight": cuda,
            "provenance": {
                "source_dataset": str(source_dataset.resolve()), "source_dataset_sha256": _sha256(source_dataset),
                "source_arrays": str(source_arrays.resolve()), "source_arrays_sha256": _sha256(source_arrays),
                "context": str(context.resolve()), "context_sha256": _sha256(context),
                "calibration_labels": str(calibration_labels.resolve()), "calibration_labels_sha256": _sha256(calibration_labels),
            },
        }
        _write_report(output, rejected)
        return rejected
    cost_contract, cost_predictor, endpoint_contract, endpoint_predictor = _fit_models(
        train_rows, train_terminal, seeds=seeds, width=width, steps=steps, device=device,
    )
    # Do not even decode calibration labels until the train-only cross-fit
    # opportunity gate has admitted a router.  The provenance checksum remains
    # lawful in a rejection artifact, but its successor values stay sealed.
    calibration = _load(calibration_labels)
    calibration_terminal = _calibration_terminal(calibration_rows, calibration)
    raw_calibration = _evaluate(
        calibration_groups, _root_costs(calibration_rows, calibration_terminal), cost_predictor=cost_predictor,
        endpoint_predictor=endpoint_predictor, router=router, threshold=float("inf"),
    )
    try:
        policy = _calibrate(raw_calibration, seed=seed)
    except _CalibrationOpportunityError as error:
        # The threshold cannot be frozen when either independent calibration
        # role lacks the declared opportunity support.  Outcome states must
        # consequently remain sealed: executing them here would turn a valid
        # identifiability stop into an outcome-informed protocol revision.
        rejected = {
            "schema": SCHEMA,
            "status": "rejected_pre_outcome_calibration_opportunity",
            "claim_boundary": "Pre-outcome V10 opportunity-support rejection. It establishes neither endpoint efficacy nor a held-out fallback consequence.",
            "control_contract": {
                "label": CONTROL_LABEL, "repair_arm": "endpoint_coupled_counterfactual",
                "fallback_arm": "direct_cost_only", "action_representation": ACTION_REPRESENTATION,
                "same_choice_roots_forced_to_cost": True,
                "outcome_execution_deferred_until_router_threshold_and_certificate_frozen": True,
                "outcome_cost_used_for_router_or_threshold_selection": False,
                "shared_candidate_support_verified": contract,
            },
            "split_isolation": {
                "train_root_count": actual_root_counts["train"], "calibration_root_count": actual_root_counts["calibration"],
                "outcome_root_count": actual_root_counts["outcome"], "root_disjoint": True,
            },
            "repair_arms": {"cost_only": cost_contract, "endpoint_coupled": endpoint_contract},
            "crossfit": {
                "fold_count": 4, "router_feature_dimension": V9_ROUTER_FEATURE_DIMENSION,
                "router_training_root_count": int(sum(cross_opportunities.values())),
                "scorer_disagreement_rate": float(np.mean(list(cross_opportunities.values()))),
                "mean_cost_minus_endpoint_regret": float(np.mean(list(cross_advantage.values()))),
                "router_training_scope": "cost_endpoint_scorer_disagreement_only",
            },
            "router_train_only_selection": router_selection,
            "calibration": {
                "status": "rejected_insufficient_opportunity_support",
                "rejection": error.details,
                "raw_summary": _summary(raw_calibration),
            },
            "outcome": {"materialized": False, "reason": "no frozen threshold or independent calibration certificate"},
            "cuda_preflight": cuda,
            "provenance": {
                "source_dataset": str(source_dataset.resolve()), "source_dataset_sha256": _sha256(source_dataset),
                "source_arrays": str(source_arrays.resolve()), "source_arrays_sha256": _sha256(source_arrays),
                "context": str(context.resolve()), "context_sha256": _sha256(context),
                "calibration_labels": str(calibration_labels.resolve()), "calibration_labels_sha256": _sha256(calibration_labels),
            },
        }
        _write_report(output, rejected)
        return rejected
    # Outcome successor labels are read exactly once, after every model,
    # cross-fit router, selector threshold, and independent certificate froze.
    outcome_terminal = _fresh_terminal(outcome_rows)
    raw_outcome = _evaluate(
        outcome_groups, _root_costs(outcome_rows, outcome_terminal), cost_predictor=cost_predictor,
        endpoint_predictor=endpoint_predictor, router=router, threshold=float(policy["threshold"]),
    )
    deployed = _deployment(raw_outcome, str(policy["deployment_policy"]))
    report = {
        "schema": SCHEMA, "status": "completed_execution_grounded_endpoint_repairability_router",
        "claim_boundary": "Root-disjoint PointMaze fixed-bank evaluation of one endpoint-coupled repair arm. It is not a general repairability theorem, HTSC certificate, online planner, or closed-loop result.",
        "control_contract": {
            "label": CONTROL_LABEL, "repair_arm": "endpoint_coupled_counterfactual",
            "fallback_arm": "direct_cost_only", "action_representation": ACTION_REPRESENTATION,
            "router_feature_contract": V9_ROUTER_FEATURE_CONTRACT,
            "source_state_raw_forbidden_in_router": True, "crossfit_train_labels_only": True,
            "same_choice_roots_forced_to_cost": True,
            "outcome_execution_deferred_until_router_threshold_and_certificate_frozen": True,
            "outcome_cost_used_for_router_or_threshold_selection": False,
            "shared_candidate_support_verified": contract,
        },
        "split_isolation": {
            "train_root_count": actual_root_counts["train"], "calibration_root_count": actual_root_counts["calibration"],
            "outcome_root_count": actual_root_counts["outcome"], "root_disjoint": True,
        },
        "repair_arms": {"cost_only": cost_contract, "endpoint_coupled": endpoint_contract},
        "crossfit": {
            "fold_count": 4, "router_feature_dimension": V9_ROUTER_FEATURE_DIMENSION,
            "router_training_root_count": int(sum(cross_opportunities.values())),
            "scorer_disagreement_rate": float(np.mean(list(cross_opportunities.values()))),
            "mean_cost_minus_endpoint_regret": float(np.mean(list(cross_advantage.values()))),
            "router_training_scope": "cost_endpoint_scorer_disagreement_only",
        },
        "router_train_only_selection": router_selection,
        "calibration": {"routing": policy, "per_root": raw_calibration},
        "outcome": {"raw_router": _summary(raw_outcome), "authorized_deployment": _summary(deployed), "per_root": deployed},
        "cuda_preflight": cuda,
        "provenance": {
            "source_dataset": str(source_dataset.resolve()), "source_dataset_sha256": _sha256(source_dataset),
            "source_arrays": str(source_arrays.resolve()), "source_arrays_sha256": _sha256(source_arrays),
            "context": str(context.resolve()), "context_sha256": _sha256(context),
            "calibration_labels": str(calibration_labels.resolve()), "calibration_labels_sha256": _sha256(calibration_labels),
        },
    }
    _write_report(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--source-arrays", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--calibration-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=202610404)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--ensemble-size", type=int, default=4)
    args = vars(parser.parse_args())
    report = run(**args)
    if report["status"] == "completed_execution_grounded_endpoint_repairability_router":
        summary = {"status": report["status"], "calibration": report["calibration"]["routing"], "outcome": report["outcome"]["authorized_deployment"]}
    else:
        summary = {"status": report["status"], "calibration": report["calibration"], "outcome": report["outcome"]}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
