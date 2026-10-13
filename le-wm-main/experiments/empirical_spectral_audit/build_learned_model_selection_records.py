"""Build immutable learned-model selection records from enriched exports.

Each candidate is supplied with two *different* canonical exports: a
diagnostic export used to fit the predictor, external task metric, and all
geometry, and an outcome export used only to measure held-out multi-step
transfer risk.  A single export cannot satisfy this interface.  This is the
adapter that turns the common enriched schema into the existing
``real_checkpoint_selection.v1`` input of the strict matched audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.empirical_spectral_audit.enriched_record_schema import load_payload  # noqa: E402
from experiments.task_metric_diagnostics import metric_geometry, task_metric_metadata  # noqa: E402
from experiments.theory_spectral_transfer.finite_horizon_transfer import finite_horizon_response  # noqa: E402


def _ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    gram = x.T @ x / max(len(x), 1)
    scale = max(float(np.trace(gram)) / max(x.shape[1], 1), 1e-12)
    return np.linalg.solve(gram + ridge * scale * np.eye(x.shape[1]), x.T @ y / max(len(x), 1))


def _rows(payload: dict[str, Any], horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source, predicted, target, state_delta = [], [], [], []
    for row in payload["enriched_records"]:
        item = row["latent_by_horizon"][str(horizon)]
        source_latent = np.asarray(row["source_latent"], dtype=np.float64)
        predicted_latent = np.asarray(item["predicted"], dtype=np.float64)
        target_latent = np.asarray(item["target"], dtype=np.float64)
        source_state = np.asarray(row["source_state_raw"], dtype=np.float64)
        target_state = np.asarray(item["target_state_raw"], dtype=np.float64)
        source.append(source_latent)
        predicted.append(predicted_latent)
        target.append(target_latent)
        state_delta.append(target_state - source_state)
    return tuple(np.asarray(item, dtype=np.float64) for item in (source, predicted, target, state_delta))


def _action_rows(payload: dict[str, Any]) -> np.ndarray:
    """Return the declared action intervention features for the first step.

    The export stores a fixed action block sequence.  We retain every block in
    order rather than selecting a hand-picked component; the resulting B
    operator is therefore tied to the export's declared action distribution.
    """
    values = []
    for row in payload["enriched_records"]:
        actions = np.asarray(row["action_blocks_normalized"], dtype=np.float64)
        values.append(actions.reshape(-1))
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError("action_blocks_normalized must produce a non-empty 2D matrix")
    return result


def _fit_transfer_operator(
    source: np.ndarray,
    predicted: np.ndarray,
    action: np.ndarray,
    state_delta: np.ndarray,
    ridge: float,
    metric: np.ndarray,
    horizons: list[int],
) -> dict[str, Any]:
    """Fit the declared column-convention local operator and export G_h.

    The fit is diagnostic-split only.  With centered rows, ``predicted =
    source @ A_row + action @ B_row`` corresponds to column operators
    ``A=A_row.T`` and ``B=B_row.T``.  The task readout is the independent
    external decoder ``state_delta = source @ D_row`` and ``D=D_row.T``.
    """
    centered_source = source - source.mean(0, keepdims=True)
    centered_predicted = predicted - predicted.mean(0, keepdims=True)
    centered_action = action - action.mean(0, keepdims=True)
    features = np.concatenate([centered_source, centered_action], axis=1)
    fit = _ridge(features, centered_predicted, ridge)
    latent_dim = source.shape[1]
    A = fit[:latent_dim].T
    B = fit[latent_dim:].T
    decoder_row = _ridge(centered_source, state_delta - state_delta.mean(0, keepdims=True), ridge)
    D = decoder_row.T
    G = {
        str(int(horizon)): finite_horizon_response(A, B, D, int(horizon)).tolist()
        for horizon in horizons
    }
    return {
        "schema": "learned_transfer_operator.v1",
        "coordinate_convention": "column_state; x_next=A x+B u; y=D x",
        "fit_source": "diagnostic export only",
        "A": A.tolist(),
        "B": B.tolist(),
        "D": D.tolist(),
        "G_h": G,
        "metric_readout": metric.tolist(),
        "identification": {
            "state_feature_count": int(source.shape[1]),
            "action_feature_count": int(action.shape[1]),
            "ridge": float(ridge),
            "centered_intercept": True,
        },
        "risk_envelope": {
            "status": "blocked_missing_independent_components",
            "closure": "available only as diagnostic-split residual after horizon-specific export",
            "off_manifold": "unidentified; independent planner-reachable transition data required",
            "action_effect": "unidentified; independent action-balanced intervention data required",
            "transient": "computable from local A,D and declared external metric, but local-tube validity is unverified",
            "readout": "decoder fitted on diagnostic data; independent task labels required for held-out certification",
            "bound_allowed_for_control": False,
        },
    }


def _fit_metric(source: np.ndarray, state_delta: np.ndarray, ridge: float) -> np.ndarray:
    """Fit an external decoder-pullback metric on diagnostic rows only."""
    decoder = _ridge(source - source.mean(0, keepdims=True), state_delta - state_delta.mean(0, keepdims=True), ridge)
    weights = np.ones(state_delta.shape[1], dtype=np.float64)
    metric = decoder @ np.diag(weights) @ decoder.T
    metric = 0.5 * (metric + metric.T)
    scale = max(float(np.trace(metric)) / max(len(metric), 1), 1e-12)
    return metric + 1e-3 * scale * np.eye(len(metric))


def _risk(payload: dict[str, Any], horizon: int) -> float:
    values = []
    for row in payload["enriched_records"]:
        item = row["latent_by_horizon"][str(horizon)]
        predicted = np.asarray(item["predicted"], dtype=np.float64)
        target = np.asarray(item["target"], dtype=np.float64)
        values.append(float(np.mean(np.square(predicted - target))))
    if not values:
        raise ValueError("outcome export contains no rows")
    return float(np.mean(values))


def _assert_pair(fit: dict[str, Any], outcome: dict[str, Any]) -> None:
    keys = ("architecture", "checkpoint", "environment", "seed", "training_stage", "latent_dimension", "action_distribution")
    for key in keys:
        if fit[key] != outcome[key]:
            raise ValueError(f"fit/outcome metadata mismatch for {key}: {fit[key]!r} != {outcome[key]!r}")
    fit_ids = {str(row["episode_id"]) for row in fit["enriched_records"]}
    outcome_ids = {str(row["episode_id"]) for row in outcome["enriched_records"]}
    overlap = fit_ids & outcome_ids
    if overlap:
        raise ValueError(f"fit/outcome episode leakage for {fit['architecture']}: {sorted(overlap)[:5]}")


def _load_external_evidence(path: Path, checkpoint: str) -> dict[str, Any]:
    """Load independent action/planning evidence without fabricating fields."""
    evidence = json.loads(path.read_text())
    if evidence.get("schema") != "learned_snapshot_external_evidence.v1":
        raise ValueError(f"{path} has incompatible external evidence schema")
    if str(Path(evidence.get("checkpoint", "")).resolve()) != str(Path(checkpoint).resolve()):
        raise ValueError(
            "external evidence checkpoint mismatch: "
            f"{evidence.get('checkpoint')!r} != {checkpoint!r}"
        )
    action = evidence.get("action_risk", {})
    planning = evidence.get("planning_consequence", {})
    required_action = ("action_inverse_risk", "action_identifiability", "paired_roots")
    required_planning = ("planning_risk", "strict_success_rate", "mean_block_xy_distance", "mean_angle_error")
    missing = [f"action_risk.{name}" for name in required_action if name not in action]
    missing.extend(f"planning_consequence.{name}" for name in required_planning if name not in planning)
    if missing:
        raise ValueError(f"{path} lacks independent evidence fields: {missing}")
    if not all(np.isfinite(float(action[name])) for name in required_action):
        raise ValueError(f"{path} has non-finite action evidence")
    if not all(np.isfinite(float(planning[name])) for name in required_planning):
        raise ValueError(f"{path} has non-finite planning evidence")
    protocol = evidence.get("protocol", {})
    action_protocol = action.get("protocol", {})
    planning_protocol = planning.get("protocol", {})
    if planning_protocol.get("runtime_state_access") is not False:
        raise ValueError(f"{path} is not RGB-only at runtime")
    if planning_protocol.get("runtime_successor_verification") is not False:
        raise ValueError(f"{path} uses runtime successor verification")
    if planning_protocol.get("test_planning_used_for_snapshot_selection") is not False:
        raise ValueError(f"{path} used test planning for snapshot selection")
    if action_protocol.get("simulator_state_used_for") != "offline intervention labels only":
        raise ValueError(f"{path} lacks an action-balanced offline-label declaration")
    if not protocol.get("diagnostic_export_not_reused_for_external_metrics", False):
        raise ValueError(f"{path} does not certify independent external metrics")
    return evidence


def build_record(
    fit_path: Path,
    outcome_path: Path,
    *,
    fit_horizon: int = 1,
    outcome_horizon: int = 4,
    ridge: float = 1e-3,
    external_evidence_path: Path | None = None,
) -> dict[str, Any]:
    fit = load_payload(fit_path)
    outcome = load_payload(outcome_path)
    _assert_pair(fit, outcome)
    if fit_horizon not in fit["horizons"] or outcome_horizon not in outcome["horizons"]:
        raise ValueError("requested horizon is absent from fit or outcome export")
    source, predicted, _target, state_delta = _rows(fit, fit_horizon)
    if len(source) < 8:
        raise ValueError(f"insufficient diagnostic rows for identified operator: {len(source)}")
    centered_source = source - source.mean(0, keepdims=True)
    centered_predicted = predicted - predicted.mean(0, keepdims=True)
    covariance = centered_source.T @ centered_source / max(len(source) - 1, 1)
    predictor = _ridge(centered_source, centered_predicted, ridge)
    metric = _fit_metric(source, state_delta, ridge)
    geometry = metric_geometry(
        covariance,
        predictor,
        metric,
        rank=min(16, source.shape[1]),
        horizons=range(1, 5),
    )
    action = _action_rows(fit)
    transfer_operator = _fit_transfer_operator(
        source,
        predicted,
        action,
        state_delta,
        ridge,
        metric,
        sorted(set(int(item) for item in fit["horizons"])),
    )
    fit_mse = float(np.mean(np.square(predicted - _target)))
    external = None if external_evidence_path is None else _load_external_evidence(external_evidence_path, fit["checkpoint"])
    record = {
        "group": f"{fit['environment']}/enriched/{fit['architecture']}/seed_{fit['seed']}",
        "architecture": fit["architecture"],
        "checkpoint": fit["checkpoint"],
        "model_factors": {
            "latent_dimension": int(fit["latent_dimension"]),
            "training_stage": int(fit["training_stage"]),
            "action_distribution": fit["action_distribution"],
        },
        "diagnostics": {
            "prediction_mse": fit_mse,
            "predictor_spectrum": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in np.linalg.eigvals(predictor)
            ],
            **{key: float(value) for key, value in geometry.items()},
            "transfer_operator": transfer_operator,
        },
        "heldout": {"planning_loss": _risk(outcome, outcome_horizon)},
        "provenance": {
            "fit_export": str(fit_path),
            "outcome_export": str(outcome_path),
            "fit_horizon": int(fit_horizon),
            "outcome_horizon": int(outcome_horizon),
            "task_metric": task_metric_metadata(metric),
            "operator_fit": "ridge(source_latent, predicted_latent) on diagnostic export only",
            "outcome": "mean held-out latent transfer MSE on outcome export only",
        },
    }
    if external is not None:
        action = external["action_risk"]
        planning = external["planning_consequence"]
        record["model_factors"]["action_inverse_risk"] = float(action["action_inverse_risk"])
        record["diagnostics"]["action_inverse_risk"] = float(action["action_inverse_risk"])
        record["diagnostics"]["action_identifiability"] = float(action["action_identifiability"])
        record["heldout"].update({
            "planning_risk": float(planning["planning_risk"]),
            "planning_consequence": {
                key: planning[key]
                for key in ("strict_success_rate", "mean_block_xy_distance", "mean_angle_error", "mean_pose_cost")
            },
        })
        record["provenance"]["external_evidence"] = str(external_evidence_path)
        record["provenance"]["action_risk"] = "independent validation matched +a/-a/zero intervention fan"
        record["provenance"]["planning_outcome"] = "fresh fixed RGB-only closed-loop episode panel"
        record["diagnostics"]["transfer_operator"]["risk_envelope"].update({
            "action_effect": "certified by independent matched intervention evidence",
            "readout": "planning consequence externally scored on fresh RGB-only episodes",
        })
    return {
        "record_schema": "real_checkpoint_selection.v1",
        "protocol": {
            "environment": fit["environment"],
            "diagnostic_split": "canonical enriched fit export; episode-disjoint from outcome export",
            "heldout_planning": "canonical enriched outcome export; no metric fitting",
            "coordinate_gauge": "learned_latent_with_external_decoder_pullback_metric",
            "runtime_successor_verification": False,
            "independent_external_evidence": external is not None,
            "checkpoint_immutable": True,
            "prediction_quality": {
                "metric": "diagnostic latent one-step prediction MSE",
                "action_distribution": fit["action_distribution"],
                "fit_horizon": int(fit_horizon),
            },
        },
        "record": record,
        "claim_boundary": (
            "One frozen learned-model candidate with a train-only external metric and a disjoint transfer outcome; "
            "cross-architecture and cross-environment claims require the strict matched audit to identify strata."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-export", action="append", required=True, type=Path)
    parser.add_argument("--outcome-export", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-horizon", type=int, default=1)
    parser.add_argument("--outcome-horizon", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--external-evidence", action="append", type=Path)
    args = parser.parse_args()
    if len(args.fit_export) != len(args.outcome_export):
        parser.error("each --fit-export must have one corresponding --outcome-export")
    if args.external_evidence is not None and len(args.external_evidence) != len(args.fit_export):
        parser.error("each fit/outcome pair must have one corresponding --external-evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (fit_path, outcome_path) in enumerate(zip(args.fit_export, args.outcome_export)):
        payload = build_record(
            fit_path,
            outcome_path,
            fit_horizon=args.fit_horizon,
            outcome_horizon=args.outcome_horizon,
            ridge=args.ridge,
            external_evidence_path=None if args.external_evidence is None else args.external_evidence[index],
        )
        target = args.output / f"candidate_{index:03d}" / "checkpoint_record.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths.append(str(target))
    (args.output / "manifest.json").write_text(json.dumps({"records": paths}, indent=2) + "\n")
    print(json.dumps({"records": paths}, indent=2))


if __name__ == "__main__":
    main()
