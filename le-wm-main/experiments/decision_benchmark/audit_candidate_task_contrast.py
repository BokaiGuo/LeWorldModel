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
    certify_candidate_bank,
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
    # `real_task_cost` is Euclidean goal distance.  Squared distance has the
    # same within-root ordering, but changes pairwise margins and invalidates
    # calibration bounds that are reported in the real-cost scale.
    return float(np.linalg.norm(predicted_position - goal))


def _train_only_fit_rows(
    fit_rows: list[dict[str, Any]], dataset_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject a readout export contaminated by candidate calibration/outcome roots.

    The old audit called its decoder train-only while accepting an arbitrary
    enriched export.  In particular, an export could contain the very roots
    later used for calibration or outcome scoring.  This helper makes root
    identity an executable contract, rather than a report-only claim.
    """
    forbidden: set[str] = set()
    for row in dataset_rows:
        if str(row.get("split")) in {"calibration", "outcome"}:
            root = row.get("start_episode_id", row.get("instance_id"))
            if root is None:
                raise ValueError("candidate dataset row lacks start_episode_id/instance_id")
            forbidden.add(str(root))
    if not forbidden:
        raise ValueError("candidate dataset has no calibration/outcome roots to protect")
    fit_ids: set[str] = set()
    for row in fit_rows:
        if "episode_id" not in row:
            raise ValueError("fit export lacks episode_id required for train-only root audit")
        fit_ids.add(str(row["episode_id"]))
    overlap = sorted(forbidden & fit_ids)
    if overlap:
        raise ValueError(
            "fit export contains candidate calibration/outcome episode IDs: "
            + ", ".join(overlap)
        )
    return fit_rows, {
        "fit_episode_count": len(fit_ids),
        "protected_nontrain_root_count": len(forbidden),
        "fit_nontrain_root_overlap": overlap,
    }


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
    fit_rows, fit_contract = _train_only_fit_rows(fit_payload["enriched_records"], rows)
    readouts = _fit_readouts(fit_rows)
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
        bound = calibration_report["bound"]
        potential_bank_certificate = certify_candidate_bank(
            predicted[indices], bound, support_passed=True,
            candidate_ids=[int(value) for value in indices],
        )
        if potential_bank_certificate["status"] == "permit":
            margin_separated += 1
        # Support coverage has not been independently calibrated for this
        # bank, so the operational status must remain abstain.
        certify_candidate_bank(
            predicted[indices], bound, support_passed=False,
            candidate_ids=[int(value) for value in indices],
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
            **fit_contract,
            "cost_scale": "euclidean_goal_distance_matches_real_task_cost",
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
