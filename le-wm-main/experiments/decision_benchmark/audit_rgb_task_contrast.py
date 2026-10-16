"""Audit task-transfer sufficiency after realized RGB successor verification.

This is the decision-level bridge between ``verify_rgb_successor.py`` and the
PDTC contract.  It joins one immutable candidate bank with predicted and
realized RGB successor latents, fits the task readout on train roots only,
calibrates a root-level contrast envelope on disjoint calibration roots, and
reports selective outcome-root regret under a fixed support gate.

The physical terminal cost remains a post-hoc consequence label.  It is never
used to select the readout, the support threshold, or the release rule.
Consequently a positive RGB successor metric cannot silently become a
closed-loop planning claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.decision_benchmark.task_contrast import (
    calibrate_contrast_bound,
    certify_candidate_bank,
    evaluate_contrast_bound,
    pairwise_task_contrast_audit,
)
from experiments.empirical_spectral_audit.enriched_record_schema import load_payload
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _fit_readouts


def _project_fit_payload(payload: dict[str, Any], metadata_path: Path | None) -> dict[str, Any]:
    """Project train-only fit records into the verifier's frozen PCA space."""

    if metadata_path is None:
        return payload
    if metadata_path.suffix == ".npz":
        metadata = np.load(metadata_path, allow_pickle=False)
        pca_mean = np.asarray(metadata["pca_mean"], dtype=np.float64)
        pca_basis = np.asarray(metadata["pca_basis"], dtype=np.float64)
    else:
        import torch

        checkpoint = torch.load(metadata_path, map_location="cpu", weights_only=False)
        pca_mean = np.asarray(checkpoint["pca_mean"], dtype=np.float64)
        pca_basis = np.asarray(checkpoint["pca_basis"], dtype=np.float64)
    projected = json.loads(json.dumps(payload))
    for row in projected["enriched_records"]:
        source = np.asarray(row["source_latent"], dtype=np.float64)
        if source.shape[-1] == pca_mean.shape[-1]:
            row["source_latent"] = ((source - pca_mean) @ pca_basis.T).tolist()
        elif source.shape[-1] != pca_basis.shape[0]:
            raise ValueError(
                f"fit source latent dimension {source.shape[-1]} matches neither raw {pca_mean.shape[-1]} "
                f"nor projected {pca_basis.shape[0]}"
            )
        for item in row["latent_by_horizon"].values():
            for field in ("predicted", "target"):
                value = np.asarray(item[field], dtype=np.float64)
                if value.shape[-1] == pca_mean.shape[-1]:
                    item[field] = ((value - pca_mean) @ pca_basis.T).tolist()
                elif value.shape[-1] != pca_basis.shape[0]:
                    raise ValueError(
                        f"fit {field} latent dimension {value.shape[-1]} matches neither raw "
                        f"{pca_mean.shape[-1]} nor projected {pca_basis.shape[0]}"
                    )
    projected["latent_dimension"] = int(pca_basis.shape[0])
    return projected


def _decode_cost(latent: np.ndarray, source_state: np.ndarray, readout: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    decoder, center, target_center = readout
    displacement = (np.asarray(latent, dtype=np.float64).reshape(1, -1) - center.reshape(1, -1)) @ decoder
    displacement = displacement[0] + target_center
    predicted_position = np.asarray(source_state, dtype=np.float64)[:2] + displacement
    return float(np.sum((predicted_position - np.asarray([1.2, 1.2])) ** 2))


def _override_predicted_rows(
    rgb_rows: dict[tuple[str, int], dict[str, Any]],
    prediction_report: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Replace only predictions with an independently trained adapter report."""

    payload = json.loads(prediction_report.read_text(encoding="utf-8"))
    overrides = {
        (str(row["instance_id"]), int(row["candidate_id"])): row
        for row in payload["rows"]
    }
    missing = sorted(set(rgb_rows) - set(overrides))
    if missing:
        raise ValueError(f"prediction report lacks {len(missing)} RGB candidate rows")
    merged = {}
    for key, base in rgb_rows.items():
        replacement = dict(base)
        replacement["predicted_terminal_projection"] = overrides[key]["predicted_terminal_projection"]
        replacement["prediction_projection_status"] = overrides[key].get(
            "prediction_projection_status", "external_prediction_report"
        )
        merged[key] = replacement
    return merged


def _loo_rms_distance_threshold(values: np.ndarray, scale: np.ndarray) -> float:
    """Fit a train-only joint-support radius in standardized RMS distance."""

    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float("inf")
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1.0e-6)
    standardized = (values - np.mean(values, axis=0, keepdims=True)) / scale
    differences = standardized[:, None, :] - standardized[None, :, :]
    distances = np.sqrt(np.mean(differences**2, axis=-1))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(max(np.quantile(nearest, 0.95), 1.0e-6))


def _fit_support(fit_rows: list[dict[str, Any]]) -> dict[str, np.ndarray | float]:
    states = np.asarray([row["source_state_raw"] for row in fit_rows], dtype=np.float64)
    action_values = np.asarray([row["action_blocks_raw"] for row in fit_rows], dtype=np.float64).reshape(len(fit_rows), -1)
    state_scale = np.maximum(np.std(states, axis=0), 1.0e-3)
    action_scale = np.maximum(np.std(action_values, axis=0), 1.0e-3)
    return {
        "state_min": np.min(states, axis=0),
        "state_max": np.max(states, axis=0),
        "action_min": np.min(action_values, axis=0),
        "action_max": np.max(action_values, axis=0),
        "state_train": states,
        "action_train": action_values,
        "state_scale": state_scale,
        "action_scale": action_scale,
        "state_joint_threshold": _loo_rms_distance_threshold(states, state_scale),
        "action_joint_threshold": _loo_rms_distance_threshold(action_values, action_scale),
    }


def _support_for_root(rows: list[dict[str, Any]], support: dict[str, Any], *, expansion: float = 0.05) -> dict[str, Any]:
    if not rows:
        raise ValueError("support check requires at least one candidate")
    source = np.asarray(rows[0]["source_state_raw"], dtype=np.float64)
    candidate_actions = np.asarray([row["action_sequence"] for row in rows], dtype=np.float64).reshape(len(rows), -1)
    actions = candidate_actions.reshape(len(rows), -1, 2)
    state_span = np.maximum(support["state_max"] - support["state_min"], 1.0e-6)
    action_span = np.maximum(support["action_max"] - support["action_min"], 1.0e-6)
    state_low = support["state_min"] - expansion * state_span
    state_high = support["state_max"] + expansion * state_span
    # Keep the old coordinatewise fields visible for auditability, but do not
    # use them as the operational gate when the train-only joint support model
    # is available.  Coordinate bounds accept impossible temporal combinations.
    action_low = np.min(support["action_min"]) - expansion * max(float(np.ptp(support["action_min"])), 1.0)
    action_high = np.max(support["action_max"]) + expansion * max(float(np.ptp(support["action_max"])), 1.0)
    state_passed = bool(np.all(source >= state_low) and np.all(source <= state_high))
    coordinate_action_passed = bool(np.all(actions >= action_low) and np.all(actions <= action_high))
    joint_available = all(
        key in support for key in ("state_train", "action_train", "state_scale", "action_scale", "state_joint_threshold", "action_joint_threshold")
    )
    if joint_available:
        state_scale_joint = np.maximum(np.asarray(support["state_scale"], dtype=np.float64), 1.0e-6)
        action_scale_joint = np.maximum(np.asarray(support["action_scale"], dtype=np.float64), 1.0e-6)
        state_train = np.asarray(support["state_train"], dtype=np.float64)
        action_train = np.asarray(support["action_train"], dtype=np.float64)
        state_distance = float(np.min(np.sqrt(np.mean(((state_train - source) / state_scale_joint) ** 2, axis=1))))
        action_distance = np.sqrt(np.min(
            np.mean(((action_train[None, :, :] - candidate_actions[:, None, :]) / action_scale_joint) ** 2, axis=-1),
            axis=1,
        ))
        state_joint_passed = bool(state_distance <= float(support["state_joint_threshold"]))
        action_joint_passed = bool(np.all(action_distance <= float(support["action_joint_threshold"])))
    else:
        state_distance = None
        action_distance = None
        state_joint_passed = state_passed
        action_joint_passed = coordinate_action_passed
    return {
        "passed": bool(state_joint_passed and action_joint_passed),
        "state_passed": state_passed,
        "action_passed": coordinate_action_passed,
        "coordinate_action_passed": coordinate_action_passed,
        "joint_support_available": bool(joint_available),
        "state_joint_passed": state_joint_passed,
        "action_joint_passed": action_joint_passed,
        "state_joint_distance": state_distance,
        "action_joint_distance_max": None if action_distance is None else float(np.max(action_distance)),
        "action_joint_distance_mean": None if action_distance is None else float(np.mean(action_distance)),
        "state_joint_threshold": None if not joint_available else float(support["state_joint_threshold"]),
        "action_joint_threshold": None if not joint_available else float(support["action_joint_threshold"]),
        "expansion": float(expansion),
        "source_state": source.tolist(),
        "action_min_observed": np.min(actions, axis=(0, 1)).tolist(),
        "action_max_observed": np.max(actions, axis=(0, 1)).tolist(),
    }


def _pairwise_upper(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.empty(0, dtype=np.float64)
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    return distances[np.triu_indices(len(values), k=1)]


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _candidate_collapse_diagnostics(
    rows: list[dict[str, Any]],
    rgb_rows: dict[tuple[str, int], dict[str, Any]],
    *,
    spread_ratio_threshold: float = 0.25,
    variation_correlation_threshold: float = 0.30,
) -> dict[str, dict[str, Any]]:
    """Measure whether the RGB successor preserves within-root action variation."""

    diagnostics: dict[str, dict[str, Any]] = {}
    for root in sorted({str(row["instance_id"]) for row in rows}):
        root_rows = sorted(
            [row for row in rows if str(row["instance_id"]) == root],
            key=lambda row: int(row["candidate_id"]),
        )
        predicted = np.asarray([
            rgb_rows[(root, int(row["candidate_id"]))]["predicted_terminal_projection"]
            for row in root_rows
        ], dtype=np.float64)
        realized = np.asarray([
            rgb_rows[(root, int(row["candidate_id"]))]["realized_terminal_projection"]
            for row in root_rows
        ], dtype=np.float64)
        actions = np.asarray([row["action_sequence"] for row in root_rows], dtype=np.float64).reshape(len(root_rows), -1)
        predicted_spread = float(np.sqrt(np.mean(np.sum((predicted - predicted.mean(axis=0)) ** 2, axis=1))))
        realized_spread = float(np.sqrt(np.mean(np.sum((realized - realized.mean(axis=0)) ** 2, axis=1))))
        spread_ratio = float(predicted_spread / max(realized_spread, 1.0e-12))
        predicted_pairwise = _pairwise_upper(predicted)
        realized_pairwise = _pairwise_upper(realized)
        action_pairwise = _pairwise_upper(actions)
        variation_corr = _safe_correlation(predicted_pairwise, realized_pairwise)
        action_predicted_corr = _safe_correlation(action_pairwise, predicted_pairwise)
        action_realized_corr = _safe_correlation(action_pairwise, realized_pairwise)
        passed = bool(
            spread_ratio >= float(spread_ratio_threshold)
            and variation_corr is not None
            and variation_corr >= float(variation_correlation_threshold)
        )
        diagnostics[root] = {
            "instance_id": root,
            "split": str(root_rows[0]["split"]),
            "candidate_count": int(len(root_rows)),
            "predicted_candidate_spread": predicted_spread,
            "realized_candidate_spread": realized_spread,
            "spread_ratio": spread_ratio,
            "predicted_realized_variation_correlation": variation_corr,
            "action_predicted_variation_correlation": action_predicted_corr,
            "action_realized_variation_correlation": action_realized_corr,
            "action_candidate_spread": float(np.sqrt(np.mean(np.sum((actions - actions.mean(axis=0)) ** 2, axis=1)))),
            "candidate_collapse_gate": {
                "passed": passed,
                "spread_ratio_threshold": float(spread_ratio_threshold),
                "variation_correlation_threshold": float(variation_correlation_threshold),
                "reason": "predicted candidate variation is not jointly supported by realized RGB variation" if not passed else "passed",
            },
        }
    return diagnostics


def _root_metrics(
    rows: list[dict[str, Any]],
    rgb_rows: dict[tuple[str, int], dict[str, Any]],
    readout: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[dict[str, Any]]:
    metrics = []
    for row in rows:
        key = (str(row["instance_id"]), int(row["candidate_id"]))
        rgb = rgb_rows.get(key)
        if rgb is None:
            raise ValueError(f"missing realized RGB row for {key}")
        predicted_cost = _decode_cost(np.asarray(rgb["predicted_terminal_projection"]), row["source_state_raw"], readout)
        realized_rgb_cost = _decode_cost(np.asarray(rgb["realized_terminal_projection"]), row["source_state_raw"], readout)
        metrics.append({
            "instance_id": str(row["instance_id"]),
            "candidate_id": int(row["candidate_id"]),
            "split": str(row["split"]),
            "predicted_cost": predicted_cost,
            "realized_rgb_cost": realized_rgb_cost,
            "real_task_cost": float(row["real_task_cost"]),
            "rgb_projection_rmse": float(rgb["predicted_vs_realized_projection_rmse"]),
        })
    return metrics


def _selective_outcome(
    metrics: list[dict[str, Any]],
    supports: dict[str, dict[str, Any]],
    collapse: dict[str, dict[str, Any]],
    bound: float | None,
) -> dict[str, Any]:
    released = []
    all_roots = []
    for root in sorted({row["instance_id"] for row in metrics}):
        group = [row for row in metrics if row["instance_id"] == root]
        group.sort(key=lambda row: row["predicted_cost"])
        best, *competitors = group
        support = supports[root]
        margins = [float(row["predicted_cost"] - best["predicted_cost"]) for row in competitors]
        margin = float(min(margins)) if margins else 0.0
        collapse_passed = bool(collapse[root]["candidate_collapse_gate"]["passed"])
        certificate = certify_candidate_bank(
            np.asarray([row["predicted_cost"] for row in group], dtype=np.float64),
            bound,
            support_passed=bool(support["passed"] and collapse_passed),
            candidate_ids=[int(row["candidate_id"]) for row in group],
        )
        permit = certificate["status"] == "permit"
        physical_oracle = min(group, key=lambda row: row["real_task_cost"])
        rgb_oracle = min(group, key=lambda row: row["realized_rgb_cost"])
        chosen = best
        physical_regret = float(chosen["real_task_cost"] - physical_oracle["real_task_cost"])
        rgb_regret = float(chosen["realized_rgb_cost"] - rgb_oracle["realized_rgb_cost"])
        record = {
            "instance_id": root,
            "support_passed": bool(support["passed"]),
            "candidate_collapse_passed": collapse_passed,
            "predicted_margin": margin,
            "all_competitors_separated": bool(certificate["status"] == "permit"),
            "competitor_count": len(competitors),
            "bound": None if bound is None else float(bound),
            "status": (
                "permit" if permit else
                ("insufficient_calibration" if bound is None else
                 ("probe" if support["passed"] and collapse_passed else "abstain"))
            ),
            "chosen_candidate": int(chosen["candidate_id"]),
            "oracle_candidate": int(physical_oracle["candidate_id"]),
            "physical_oracle_candidate": int(physical_oracle["candidate_id"]),
            "rgb_oracle_candidate": int(rgb_oracle["candidate_id"]),
            "physical_ranking_correct": bool(chosen["candidate_id"] == physical_oracle["candidate_id"]),
            "rgb_ranking_correct": bool(chosen["candidate_id"] == rgb_oracle["candidate_id"]),
            "chosen_real_task_cost": float(chosen["real_task_cost"]),
            "oracle_real_task_cost": float(physical_oracle["real_task_cost"]),
            "regret": physical_regret,
            "physical_regret": physical_regret,
            "chosen_realized_rgb_cost": float(chosen["realized_rgb_cost"]),
            "oracle_realized_rgb_cost": float(rgb_oracle["realized_rgb_cost"]),
            "rgb_regret": rgb_regret,
        }
        all_roots.append(record)
        if permit:
            released.append(record)
    return {
        "roots": all_roots,
        "root_count": len(all_roots),
        "released_root_count": len(released),
        "coverage": None if not all_roots else float(len(released) / len(all_roots)),
        "release_rate": None if not all_roots else float(len(released) / len(all_roots)),
        "useful_coverage": None if not all_roots else float(sum(
            row["physical_ranking_correct"] for row in released
        ) / len(all_roots)),
        "released_mean_regret": None if not released else float(np.mean([row["regret"] for row in released])),
        "all_root_mean_regret": None if not all_roots else float(np.mean([row["regret"] for row in all_roots])),
        "released_false_release_rate": None if not released else float(np.mean([
            row["chosen_candidate"] != row["oracle_candidate"] for row in released
        ])),
        "released_physical_false_release_rate": None if not released else float(np.mean([
            not row["physical_ranking_correct"] for row in released
        ])),
        "released_rgb_false_release_rate": None if not released else float(np.mean([
            not row["rgb_ranking_correct"] for row in released
        ])),
        "coverage_field_note": "coverage is retained as a backward-compatible alias for release_rate; conformal outcome coverage is reported separately",
    }


def audit(
    dataset: Path,
    fit_export: Path,
    rgb_report: Path,
    projection_metadata: Path | None = None,
    prediction_report: Path | None = None,
    *,
    support_expansion: float = 0.05,
    alpha: float = 0.1,
    collapse_spread_ratio_threshold: float = 0.25,
    collapse_variation_correlation_threshold: float = 0.30,
) -> dict[str, Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    rows = list(payload["records"])
    rgb_payload = json.loads(rgb_report.read_text(encoding="utf-8"))
    rgb_rows = {(str(row["instance_id"]), int(row["candidate_id"])): row for row in rgb_payload["rows"]}
    if prediction_report is not None:
        rgb_rows = _override_predicted_rows(rgb_rows, prediction_report)
    fit_payload = load_payload(fit_export)
    fit_payload = _project_fit_payload(fit_payload, projection_metadata)
    readouts = _fit_readouts(fit_payload["enriched_records"])
    metrics = _root_metrics(rows, rgb_rows, readouts[4])
    fit_rows = fit_payload["enriched_records"]
    support = _fit_support(fit_rows)
    supports = {
        root: _support_for_root([row for row in rows if str(row["instance_id"]) == root], support, expansion=support_expansion)
        for root in sorted({str(row["instance_id"]) for row in rows})
    }
    collapse = _candidate_collapse_diagnostics(
        rows,
        rgb_rows,
        spread_ratio_threshold=collapse_spread_ratio_threshold,
        variation_correlation_threshold=collapse_variation_correlation_threshold,
    )
    calibration = [row for row in metrics if row["split"] == "calibration"]
    outcome = [row for row in metrics if row["split"] == "outcome"]
    calibration_groups = np.asarray([row["instance_id"] for row in calibration])
    calibration_bound = calibrate_contrast_bound(
        np.asarray([row["realized_rgb_cost"] for row in calibration]),
        np.asarray([row["predicted_cost"] for row in calibration]),
        calibration_groups,
        np.unique(calibration_groups),
        alpha=alpha,
        minimum_decision_margin=1.0e-3,
    )
    outcome_groups = np.asarray([row["instance_id"] for row in outcome])
    outcome_rgb = pairwise_task_contrast_audit(
        np.asarray([row["realized_rgb_cost"] for row in outcome]),
        np.asarray([row["predicted_cost"] for row in outcome]),
        outcome_groups,
        minimum_decision_margin=1.0e-3,
    )
    outcome_physical = pairwise_task_contrast_audit(
        np.asarray([row["real_task_cost"] for row in outcome]),
        np.asarray([row["predicted_cost"] for row in outcome]),
        outcome_groups,
        minimum_decision_margin=1.0e-3,
    )
    outcome_coverage = evaluate_contrast_bound(
        np.asarray([row["realized_rgb_cost"] for row in outcome]),
        np.asarray([row["predicted_cost"] for row in outcome]),
        outcome_groups,
        calibration_bound["bound"],
    )
    outcome_support = {root: supports[root] for root in sorted({row["instance_id"] for row in outcome})}
    selective = _selective_outcome(outcome, outcome_support, collapse, calibration_bound["bound"])
    return {
        "schema": "rgb_task_transfer_sufficiency_audit.v1",
        "dataset": str(dataset.resolve()),
        "fit_export": str(fit_export.resolve()),
        "rgb_report": str(rgb_report.resolve()),
        "prediction_report": None if prediction_report is None else str(prediction_report.resolve()),
        "projection_metadata": None if projection_metadata is None else str(projection_metadata.resolve()),
        "record_count": len(rows),
        "rgb_record_count": len(rgb_rows),
        "calibration_root_count": len(set(calibration_groups.tolist())),
        "outcome_root_count": len(set(outcome_groups.tolist())),
        "rgb_contrast_outcome": outcome_rgb,
        "physical_contrast_outcome": outcome_physical,
        "conformal_outcome_coverage": outcome_coverage,
        "calibration": calibration_bound,
        "support": {
            "fit_row_count": len(fit_rows),
            "declared_gate": "source-state and full candidate-action joint kNN support fitted on train roots only; coordinate bounds retained as diagnostics",
            "expansion": float(support_expansion),
            "outcome_root_pass_count": int(sum(item["passed"] for item in outcome_support.values())),
            "outcome_root_count": len(outcome_support),
            "roots": outcome_support,
        },
        "candidate_collapse": {
            "declared_gate": "predicted within-root candidate variation must track independently realized RGB variation",
            "thresholds": {
                "spread_ratio": float(collapse_spread_ratio_threshold),
                "variation_correlation": float(collapse_variation_correlation_threshold),
            },
            "root_pass_count": int(sum(item["candidate_collapse_gate"]["passed"] for item in collapse.values())),
            "root_count": len(collapse),
            "roots": collapse,
        },
        "selective_outcome": selective,
        "gates": {
            "train_only_readout": True,
            "root_disjoint_rgb_calibration": True,
            "rgb_successor_available": len(rgb_rows) == len(rows),
            "prediction_source": "external_prediction_report" if prediction_report is not None else "candidate_bank_rgb_report",
            "support_gate_independent_of_outcome": True,
            "closed_loop_online_planning": False,
            "candidate_bank_mode": "fixed_predeclared",
            "adaptive_cem_coverage_certified": False,
        },
        "claim_boundary": "selective fixed-bank RGB/task transfer audit; not closed-loop online planning or universal transfer",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fit-export", type=Path, required=True)
    parser.add_argument("--rgb-report", type=Path, required=True)
    parser.add_argument("--projection-metadata", type=Path)
    parser.add_argument("--prediction-report", type=Path)
    parser.add_argument("--support-expansion", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--collapse-spread-ratio-threshold", type=float, default=0.25)
    parser.add_argument("--collapse-variation-correlation-threshold", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.dataset,
        args.fit_export,
        args.rgb_report,
        args.projection_metadata,
        args.prediction_report,
        support_expansion=args.support_expansion,
        alpha=args.alpha,
        collapse_spread_ratio_threshold=args.collapse_spread_ratio_threshold,
        collapse_variation_correlation_threshold=args.collapse_variation_correlation_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "rgb_contrast_inversion_rate": report["rgb_contrast_outcome"]["inversion_rate"],
        "physical_contrast_inversion_rate": report["physical_contrast_outcome"]["inversion_rate"],
        "released_root_count": report["selective_outcome"]["released_root_count"],
        "coverage": report["selective_outcome"]["coverage"],
        "released_mean_regret": report["selective_outcome"]["released_mean_regret"],
    }, indent=2))


if __name__ == "__main__":
    main()
