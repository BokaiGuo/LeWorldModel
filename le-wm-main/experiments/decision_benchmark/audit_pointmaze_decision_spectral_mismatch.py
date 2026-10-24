"""Audit sealed PointMaze endpoints as task-output response operators.

This is a P1 diagnostic, not a repair algorithm.  It reads endpoint vectors
that were serialized before outcome terminal materialization, fits each root's
response only on its candidate-supported action subspace, and then compares
the fixed mismatch magnitude with realized endpoint-ranking regret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.decision_spectral_mismatch import (
    candidate_pair_leverage,
    decision_spectral_mismatch,
    fit_supported_response,
)


SCHEMA = "pointmaze_decision_spectral_mismatch_audit.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        result[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return result


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _candidate_choice(candidate_ids: list[str], costs: np.ndarray) -> int:
    """Mirror the endpoint runner's declared score/tie-breaking rule."""

    values = np.asarray(costs, dtype=np.float64)
    if values.shape != (len(candidate_ids),) or not np.isfinite(values).all():
        raise ValueError("candidate costs must be one finite value per candidate")
    return min(range(len(candidate_ids)), key=lambda index: (float(values[index]), candidate_ids[index]))


def _pre_outcome_primary_metric(bundle: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Return a hash-bound primary metric declaration, or the legacy default."""

    binding = bundle.get("pre_outcome_diagnostic_spec")
    if binding is None:
        return "spectral_mismatch_norm", None
    if not isinstance(binding, dict):
        raise ValueError("pre-outcome diagnostic spec binding must be an object")
    path = Path(str(binding.get("path", "")))
    if not path.is_file() or _sha256(path) != binding.get("sha256"):
        raise ValueError("pre-outcome diagnostic spec is missing or hash-mismatched")
    spec = _load(path)
    if binding.get("spec") != spec or spec.get("schema") != "pointmaze_p1_pre_outcome_diagnostic_spec.v1":
        raise ValueError("pre-outcome diagnostic spec payload is not bound to its sealed file")
    metric = spec.get("primary_metric")
    if spec.get("selected_before_outcomes") is not True or metric not in {"spectral_mismatch_norm", "frozen_pair_squared_task_contrast_error"}:
        raise ValueError("pre-outcome diagnostic spec is invalid")
    return str(metric), {"path": str(path.resolve()), "sha256": binding["sha256"], "spec": spec}


def _correlation_summary(mismatch: np.ndarray, regret: np.ndarray, *, seed: int) -> dict[str, Any]:
    observed = _spearman(mismatch, regret)
    if observed is None:
        return {"status": "not_identifiable_constant_metric_or_regret", "spearman": None}
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        _spearman(mismatch[index], regret[index])
        for index in rng.integers(0, len(mismatch), size=(4000, len(mismatch)))
    ], dtype=object)
    finite_bootstrap = np.asarray([value for value in bootstrap if value is not None], dtype=np.float64)
    null = np.asarray([
        _spearman(mismatch, regret[rng.permutation(len(regret))])
        for _ in range(10000)
    ], dtype=np.float64)
    return {
        "status": "descriptive_root_level_association",
        "spearman": observed,
        "bootstrap_95_interval": None if not len(finite_bootstrap) else np.quantile(finite_bootstrap, [0.025, 0.975]).tolist(),
        "two_sided_permutation_p": float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null))),
        "root_count": int(len(mismatch)),
        "inference_unit": "root; candidates within a root are not treated as independent observations",
    }


def _root_audit(prediction: dict[str, Any], realized: dict[str, Any]) -> dict[str, Any]:
    candidate_ids = [str(value) for value in prediction.get("candidate_ids", [])]
    if candidate_ids != [str(value) for value in realized.get("candidate_ids", [])] or len(candidate_ids) < 3:
        raise ValueError("frozen and realized candidate identities must match with at least three candidates")
    actions = np.asarray(prediction["candidate_action_sequences"], dtype=np.float64)
    if actions.ndim != 3 or actions.shape[0] != len(candidate_ids):
        raise ValueError("candidate action sequences must be [candidate,time,action]")
    flattened_actions = actions.reshape(len(actions), -1)
    predicted_terminal = np.asarray(prediction["endpoint_predicted_terminal_state"], dtype=np.float64)
    realized_terminal = np.asarray(realized["realized_candidate_terminal_state"], dtype=np.float64)
    if predicted_terminal.shape != realized_terminal.shape or predicted_terminal.shape != (len(candidate_ids), 4):
        raise ValueError("endpoint terminal vectors must be aligned [candidate,4]")
    real_fit = fit_supported_response(flattened_actions, realized_terminal[:, :2])
    predicted_fit = fit_supported_response(flattened_actions, predicted_terminal[:, :2])
    if not np.allclose(real_fit["basis"], predicted_fit["basis"], atol=1.0e-10):
        raise ValueError("real and predicted response fits disagree on the frozen action support basis")
    mismatch = decision_spectral_mismatch(
        real_fit["response"], predicted_fit["response"],
        task_metric=np.eye(2), action_covariance=real_fit["covariance"],
    )
    index = {candidate: position for position, candidate in enumerate(candidate_ids)}
    endpoint = str(prediction["endpoint_candidate_id"])
    cost = str(prediction["cost_candidate_id"])
    if endpoint not in index or cost not in index:
        raise ValueError("frozen selected candidate is absent from the frozen bank")
    goal = np.asarray(prediction["goal_raw"], dtype=np.float64)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("declared endpoint goal must be a finite [x,y] vector")
    endpoint_predicted_cost = np.asarray(prediction.get("endpoint_predicted_cost"), dtype=np.float64)
    predicted_endpoint_cost = np.linalg.norm(predicted_terminal[:, :2] - goal, axis=1)
    if endpoint_predicted_cost.shape != (len(candidate_ids),) or not np.isfinite(endpoint_predicted_cost).all():
        raise ValueError("sealed endpoint predicted costs must contain one finite value per candidate")
    if not np.allclose(endpoint_predicted_cost, predicted_endpoint_cost, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("sealed endpoint predicted costs do not equal Euclidean endpoint-to-goal distance")
    endpoint_index = index[endpoint]
    if endpoint_index != _candidate_choice(candidate_ids, endpoint_predicted_cost):
        raise ValueError("sealed endpoint candidate is not the deterministic argmin of sealed endpoint predicted costs")
    realized_endpoint_cost = np.linalg.norm(realized_terminal[:, :2] - goal, axis=1)
    realized_regret = float(realized_endpoint_cost[endpoint_index] - np.min(realized_endpoint_cost))
    if not np.isclose(float(realized["endpoint_regret"]), realized_regret, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("realized endpoint regret is inconsistent with the sealed candidate ordering")
    predicted_pair_cost = np.square(predicted_terminal[:, :2] - goal).sum(axis=1)
    realized_pair_cost = np.square(realized_terminal[:, :2] - goal).sum(axis=1)
    predicted_pair_delta = float(predicted_pair_cost[index[endpoint]] - predicted_pair_cost[index[cost]])
    realized_pair_delta = float(realized_pair_cost[index[endpoint]] - realized_pair_cost[index[cost]])
    midpoint = 0.5 * (predicted_terminal[index[endpoint], :2] + predicted_terminal[index[cost], :2])
    leverage = candidate_pair_leverage(
        mismatch,
        action_delta=flattened_actions[index[endpoint]] - flattened_actions[index[cost]],
        task_gradient=2.0 * (midpoint - goal),
        action_basis=real_fit["basis"], action_covariance=real_fit["covariance"], task_metric=np.eye(2),
    )
    return {
        "support_rank": int(real_fit["support_rank"]),
        "support_condition_number": float(real_fit["condition_number"]),
        "mismatch_frobenius_norm": float(mismatch["frobenius_norm"]),
        "mismatch_spectral_norm": float(mismatch["spectral_norm"]),
        "mismatch_singular_values": np.asarray(mismatch["singular_values"], dtype=np.float64).tolist(),
        "frozen_decision_pair": {"endpoint_candidate_id": endpoint, "cost_candidate_id": cost},
        "endpoint_readout_binding": {
            "readout": "Euclidean terminal qpos [x,y] to declared goal",
            "endpoint_candidate_is_deterministic_argmin": True,
            "maximum_absolute_cost_error": float(np.max(np.abs(endpoint_predicted_cost - predicted_endpoint_cost))),
            "realized_regret_recomputed_from_same_candidate_ordering": True,
        },
        "frozen_pair_leverage": {
            "first_order_cost_mismatch": leverage["first_order_cost_mismatch"],
            "absolute_modal_sum": leverage["absolute_modal_sum"],
            "modal_absolute_contributions": np.asarray(leverage["modal_absolute_contributions"]).tolist(),
            "claim_boundary": leverage["claim_boundary"],
        },
        "frozen_pair_exact_task_contrast": {
            "predicted_endpoint_minus_cost": predicted_pair_delta,
            "realized_endpoint_minus_cost": realized_pair_delta,
            "absolute_contrast_error": abs(predicted_pair_delta - realized_pair_delta),
            "claim_boundary": "exact squared terminal-task cost discrepancy for the already frozen endpoint-versus-cost pair; diagnostic only, not a fitted repair target",
        },
        "endpoint_regret": float(realized["endpoint_regret"]),
        "cost_regret": float(realized["cost_regret"]),
        # The endpoint runner records Euclidean regret for its deployment
        # comparison.  The pair-contrast specification, however, explicitly
        # declares *squared* terminal cost.  Preserve the runner value above
        # for legacy diagnostics, but derive the confirmatory target in this
        # audit from the same squared task cost as the contrast itself.
        "endpoint_squared_regret": float(realized_pair_cost[endpoint_index] - np.min(realized_pair_cost)),
        "cost_squared_regret": float(realized_pair_cost[index[cost]] - np.min(realized_pair_cost)),
    }


def audit(*, report_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = _load(report_path)
    binding = report.get("pre_outcome_prediction_bundle")
    if not isinstance(binding, dict) or binding.get("schema") != "pointmaze_counterfactual_endpoint_pre_outcome_predictions.v1":
        raise ValueError("endpoint report lacks a sealed pre-outcome prediction bundle")
    bundle_path = Path(str(binding.get("path", "")))
    if not bundle_path.is_file() or _sha256(bundle_path) != binding.get("sha256"):
        raise ValueError("sealed pre-outcome prediction bundle is missing or hash-mismatched")
    bundle = _load(bundle_path)
    if bundle.get("outcome_labels_materialized") is not False:
        raise ValueError("pre-outcome bundle does not attest sealed outcome labels")
    primary_metric, metric_spec = _pre_outcome_primary_metric(bundle)
    per_split: dict[str, dict[str, Any]] = {}
    for split in ("calibration", "outcome"):
        frozen = bundle.get(f"{split}_predictions")
        realized = report.get(split, {}).get("per_root")
        if not isinstance(frozen, dict) or not isinstance(realized, dict) or set(frozen) != set(realized):
            raise ValueError(f"{split}: frozen prediction and realized root sets must match")
        rows = {root: _root_audit(frozen[root], realized[root]) for root in sorted(frozen)}
        mismatch = np.asarray([row["mismatch_spectral_norm"] for row in rows.values()], dtype=np.float64)
        regret = np.asarray([row["endpoint_regret"] for row in rows.values()], dtype=np.float64)
        squared_regret = np.asarray([row["endpoint_squared_regret"] for row in rows.values()], dtype=np.float64)
        pair_leverage = np.asarray([
            abs(row["frozen_pair_leverage"]["first_order_cost_mismatch"])
            for row in rows.values()
        ], dtype=np.float64)
        exact_pair_contrast_error = np.asarray([
            row["frozen_pair_exact_task_contrast"]["absolute_contrast_error"]
            for row in rows.values()
        ], dtype=np.float64)
        per_split[split] = {
            "per_root": rows,
            "spectral_mismatch_vs_endpoint_regret": _correlation_summary(
                mismatch, regret, seed=202610230 + (0 if split == "calibration" else 1),
            ),
            "frozen_pair_leverage_vs_endpoint_regret": _correlation_summary(
                pair_leverage, regret, seed=202610232 + (0 if split == "calibration" else 1),
            ),
            "frozen_pair_exact_task_contrast_error_vs_endpoint_regret": _correlation_summary(
                exact_pair_contrast_error, regret, seed=202610234 + (0 if split == "calibration" else 1),
            ),
            "frozen_pair_exact_task_contrast_error_vs_endpoint_squared_regret": _correlation_summary(
                exact_pair_contrast_error, squared_regret, seed=202610235 + (0 if split == "calibration" else 1),
            ),
        }
        per_split[split]["primary_metric_association"] = per_split[split][
            "spectral_mismatch_vs_endpoint_regret" if primary_metric == "spectral_mismatch_norm"
            else "frozen_pair_exact_task_contrast_error_vs_endpoint_squared_regret"
        ]
    result = {
        "schema": SCHEMA,
        "status": "completed_fixed_metric_audit",
        "protocol": {
            "task_output": "PointMaze terminal qpos [x,y]",
            "task_metric": "identity in physical qpos coordinates",
            "action_support": "root-local centered candidate-bank SVD support only",
            "metric_selected_before_outcomes": True,
            "candidate_pair_for_leverage_selected_before_outcomes": "endpoint choice versus cost-only choice",
            "primary_metric": primary_metric,
            "primary_metric_selected_before_outcomes": True,
            "pair_contrast_primary_outcome": (
                "endpoint_squared_regret under the predeclared squared terminal task cost"
                if primary_metric == "frozen_pair_squared_task_contrast_error" else None
            ),
        },
        "pre_outcome_primary_metric_spec": metric_spec,
        "sealed_prediction_provenance": {"path": str(bundle_path.resolve()), "sha256": binding["sha256"]},
        "splits": per_split,
        "claim_boundary": "A root-level held-out association audit for one PointMaze candidate-bank protocol. Only the hash-bound primary metric selected before outcome materialization is confirmatory; all other associations are exploratory and cannot choose a repair. Neither establishes causal repair value, continuous-action generalization, or RGB-only MPC improvement.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", dest="report_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps({
        "status": result["status"],
        "primary_metric": result["protocol"]["primary_metric"],
        "outcome": result["splits"]["outcome"]["primary_metric_association"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
