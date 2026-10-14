"""Audit task contrasts on an existing shared PointMaze candidate bank.

This runner uses a train-only latent-to-task readout and offline candidate
outcomes.  It is useful for wiring the PDTC protocol, but it is not RGB-only
closed-loop evidence because the candidate bank was generated without fresh
RGB rollouts for every simulated sequence.
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

from experiments.empirical_spectral_audit.enriched_record_schema import load_payload
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _fit_readouts
from experiments.decision_benchmark.task_contrast import (
    calibrate_contrast_bound,
    certify_pair,
    pairwise_task_contrast_audit,
)


def _decode_cost(row: dict[str, Any], arrays: Any, readouts: dict[int, Any]) -> float:
    horizon = int(row["predicted_horizon"])
    decoder, center, target_center = readouts[horizon]
    key = str(row["predicted_latent_ref"]).split("::", 1)[-1]
    latent = np.asarray(arrays[key], dtype=np.float64).reshape(1, -1)
    displacement = (latent - center.reshape(1, -1)) @ decoder + target_center.reshape(1, -1)
    predicted_position = np.asarray(row["source_state_raw"], dtype=np.float64)[:2] + displacement[0]
    goal = np.asarray(row["goal_raw"], dtype=np.float64)[:2]
    return float(np.sum((predicted_position - goal) ** 2))


def _project_fit_payload(payload: dict[str, Any], metadata_path: Path | None) -> dict[str, Any]:
    if metadata_path is None:
        return payload
    metadata = np.load(metadata_path, allow_pickle=False) if metadata_path.suffix == ".npz" else None
    if metadata is not None:
        pca_mean = np.asarray(metadata["pca_mean"], dtype=np.float64)
        pca_basis = np.asarray(metadata["pca_basis"], dtype=np.float64)
    else:
        import torch
        checkpoint = torch.load(metadata_path, map_location="cpu", weights_only=False)
        pca_mean = np.asarray(checkpoint["pca_mean"], dtype=np.float64)
        pca_basis = np.asarray(checkpoint["pca_basis"], dtype=np.float64)
    projected = json.loads(json.dumps(payload))
    for row in projected["enriched_records"]:
        row["source_latent"] = ((np.asarray(row["source_latent"], dtype=np.float64) - pca_mean) @ pca_basis.T).tolist()
        for item in row["latent_by_horizon"].values():
            item["predicted"] = ((np.asarray(item["predicted"], dtype=np.float64) - pca_mean) @ pca_basis.T).tolist()
            item["target"] = ((np.asarray(item["target"], dtype=np.float64) - pca_mean) @ pca_basis.T).tolist()
    projected["latent_dimension"] = int(pca_basis.shape[0])
    return projected


def audit(dataset: Path, arrays_path: Path, fit_export: Path, projection_metadata: Path | None = None) -> dict[str, Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    rows = list(payload["records"])
    arrays = np.load(arrays_path, allow_pickle=False)
    fit_payload = load_payload(fit_export)
    if projection_metadata is not None:
        fit_payload = _project_fit_payload(fit_payload, projection_metadata)
    readouts = _fit_readouts(fit_payload["enriched_records"])
    real = np.asarray([float(row["real_task_cost"]) for row in rows], dtype=np.float64)
    predicted = np.asarray([_decode_cost(row, arrays, readouts) for row in rows], dtype=np.float64)
    groups = np.asarray([str(row["instance_id"]) for row in rows])
    split = np.asarray([str(row["split"]) for row in rows])
    outcome = split == "outcome"
    calibration = split == "calibration"
    outcome_report = pairwise_task_contrast_audit(real[outcome], predicted[outcome], groups[outcome], minimum_decision_margin=1.0e-3)
    calibration_report = calibrate_contrast_bound(
        real[calibration], predicted[calibration], groups[calibration], np.unique(groups[calibration]),
        alpha=0.1, minimum_decision_margin=1.0e-3,
    )
    margin_separated = 0
    root_count = 0
    for group in np.unique(groups[outcome]):
        indices = np.flatnonzero(outcome & (groups == group))
        order = indices[np.argsort(predicted[indices])]
        if len(order) < 2:
            continue
        root_count += 1
        best, runner_up = int(order[0]), int(order[1])
        if float(predicted[runner_up] - predicted[best]) > float(calibration_report["bound"]):
            margin_separated += 1
        # Support coverage has not been independently calibrated for this
        # bank, so the operational status must remain abstain.
        certify_pair(
            float(predicted[best]), float(predicted[runner_up]), calibration_report["bound"],
            support_passed=False, candidate_ids=(int(best), int(runner_up)),
        )
    return {
        "schema": "offline_candidate_task_contrast_audit.v1",
        "dataset": str(dataset.resolve()),
        "arrays": str(arrays_path.resolve()),
        "fit_export": str(fit_export.resolve()),
        "projection_metadata": None if projection_metadata is None else str(projection_metadata.resolve()),
        "record_count": len(rows),
        "outcome_root_count": int(len(np.unique(groups[outcome]))),
        "calibration_root_count": int(len(np.unique(groups[calibration]))),
        "outcome": outcome_report,
        "calibration": calibration_report,
        "certificate_preview": {
            "status": "abstain_support_gate_unavailable",
            "root_count": root_count,
            "margin_separated_root_count": margin_separated,
            "potential_margin_separated_fraction": None if not root_count else float(margin_separated / root_count),
            "released_root_count": 0,
            "reason": "candidate support coverage and RGB realized successor are not independently calibrated",
        },
        "protocol": {
            "train_only_readout": True,
            "root_disjoint_calibration_outcome": True,
            "same_candidate_bank": True,
            "rgb_realized_latent_available": False,
            "runtime_privileged_state": False,
            "claim_boundary": "offline candidate contrast and certificate wiring; not RGB-only closed-loop planning",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--fit-export", type=Path, required=True)
    parser.add_argument("--projection-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset, args.arrays, args.fit_export, args.projection_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": report["outcome"], "calibration": report["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
