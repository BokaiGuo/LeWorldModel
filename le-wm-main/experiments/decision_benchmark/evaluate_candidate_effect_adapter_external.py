"""Evaluate a frozen PointMaze candidate-effect protocol on external roots.

The adapter is fitted only on the development bank's train roots.  Its risk
bound and support gate use only development calibration/train roots.  Every
record in the external bank must be labeled ``external_outcome``; those labels
are read only after fitting and calibration are complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.task_contrast import calibrate_root_score_bound
from experiments.decision_benchmark.train_candidate_effect_adapter import (
    SEEDS,
    _cyclic_within_root,
    _features,
    _fit_predict,
    _matched_order_pair_audit,
    _paired_regret_gain,
    _root_selection,
    _train_only_root_support,
)


def _load_bank(dataset: Path, arrays: Path) -> tuple[list[dict[str, Any]], Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"candidate bank has no records: {dataset}")
    return records, np.load(arrays, allow_pickle=False)


def _validate_fixed_bank_protocol(
    development_records: list[dict[str, Any]],
    outcome_records: list[dict[str, Any]],
) -> list[int]:
    """Fail closed unless the external panel is root-disjoint and bank-matched."""

    def roots(records: list[dict[str, Any]]) -> set[str]:
        return {str(row["instance_id"]) for row in records}

    development_roots = roots(development_records)
    outcome_roots = roots(outcome_records)
    overlap = sorted(development_roots & outcome_roots)
    if overlap:
        raise ValueError(f"development and external roots overlap: {overlap[:3]}")
    if {str(row.get("split")) for row in outcome_records} != {"external_outcome"}:
        raise ValueError("external bank must contain only external_outcome records")

    train_candidate_ids = {
        int(row["candidate_id"])
        for row in development_records
        if row.get("split") == "train"
    }
    if not train_candidate_ids:
        raise ValueError("development bank requires train candidates")
    expected = sorted(train_candidate_ids)
    for panel, records in (("development", development_records), ("external", outcome_records)):
        by_root: dict[str, set[int]] = {}
        for row in records:
            by_root.setdefault(str(row["instance_id"]), set()).add(int(row["candidate_id"]))
        mismatched = [root for root, candidates in by_root.items() if sorted(candidates) != expected]
        if mismatched:
            raise ValueError(
                f"{panel} roots do not share the declared candidate IDs: {mismatched[:3]}"
            )

    reference: dict[int, np.ndarray] = {}
    for row in development_records + outcome_records:
        candidate = int(row["candidate_id"])
        sequence = np.asarray(row["action_sequence"], dtype=np.float64)
        if candidate not in reference:
            reference[candidate] = sequence
        elif reference[candidate].shape != sequence.shape or not np.allclose(
            reference[candidate], sequence, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError(
                "development and external panels do not use one shared world-action bank"
            )
    return expected


def _bounded_regret_report(
    calibration_selection: dict[str, Any],
    outcome_selection: dict[str, Any],
    support: dict[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    supported_calibration = [
        root for root in sorted(calibration_selection["roots"])
        if support["roots"][root]["passed"]
    ]
    calibration = calibrate_root_score_bound(
        np.asarray([
            calibration_selection["roots"][root]["regret"]
            for root in supported_calibration
        ]),
        alpha=alpha,
    )
    supported_outcome = [
        root for root in sorted(outcome_selection["roots"])
        if support["roots"][root]["passed"]
    ]
    bound = calibration["bound"]
    reports = {}
    for tolerance in (0.05, 0.1, 0.25):
        permitted = (
            supported_outcome
            if bound is not None and float(bound) <= tolerance
            else []
        )
        reports[str(tolerance)] = {
            "tolerance": tolerance,
            "permitted_root_count": len(permitted),
            "release_rate": float(len(permitted) / len(outcome_selection["roots"])),
            "empirical_violation_rate": None if not permitted else float(np.mean([
                outcome_selection["roots"][root]["regret"] > tolerance
                for root in permitted
            ])),
            "mean_released_regret": None if not permitted else float(np.mean([
                outcome_selection["roots"][root]["regret"] for root in permitted
            ])),
        }
    return {
        "support": support,
        "supported_calibration_root_count": len(supported_calibration),
        "supported_outcome_root_count": len(supported_outcome),
        "calibration": calibration,
        "tolerance_reports": reports,
        "certificate_type": "external marginal epsilon-optimal fixed-bank regret",
    }


def run(
    development_dataset: Path,
    development_arrays: Path,
    outcome_dataset: Path,
    outcome_arrays: Path,
    output: Path,
    *,
    epochs: int = 300,
    bootstrap_samples: int = 2000,
    feature_mode: str = "frozen_latent_action_basis",
    device: str = "cuda",
    alpha: float = 0.1,
) -> dict[str, Any]:
    development_records, development_array_payload = _load_bank(
        development_dataset, development_arrays
    )
    outcome_records, outcome_array_payload = _load_bank(outcome_dataset, outcome_arrays)
    candidate_ids = _validate_fixed_bank_protocol(development_records, outcome_records)

    development_x, development_y, development_roots, _ = _features(
        development_records, development_array_payload, feature_mode
    )
    outcome_x, outcome_y, outcome_roots, _ = _features(
        outcome_records, outcome_array_payload, feature_mode
    )
    train_mask = np.asarray([row.get("split") == "train" for row in development_records])
    calibration_mask = np.asarray([
        row.get("split") == "calibration" for row in development_records
    ])
    if not train_mask.any() or not calibration_mask.any():
        raise ValueError("development bank requires train and calibration roots")

    combined_x = np.concatenate([development_x, outcome_x], axis=0)
    predictions = []
    training = []
    for seed in SEEDS:
        prediction, metadata = _fit_predict(
            development_x[train_mask], development_y[train_mask], combined_x,
            seed, epochs, device,
        )
        predictions.append(prediction)
        training.append({"seed": int(seed), **metadata})
    ensemble = np.mean(np.stack(predictions), axis=0)
    development_effect = ensemble[: len(development_records)]
    outcome_effect = ensemble[len(development_records) :]

    development_sources = np.asarray([
        row["source_state_raw"] for row in development_records
    ], dtype=np.float64)[:, :2]
    development_goals = np.asarray([
        row["goal_raw"] for row in development_records
    ], dtype=np.float64)
    outcome_sources = np.asarray([
        row["source_state_raw"] for row in outcome_records
    ], dtype=np.float64)[:, :2]
    outcome_goals = np.asarray([
        row["goal_raw"] for row in outcome_records
    ], dtype=np.float64)
    development_cost = np.linalg.norm(
        development_sources + development_effect - development_goals, axis=1
    )
    outcome_cost = np.linalg.norm(outcome_sources + outcome_effect - outcome_goals, axis=1)

    calibration_records = [
        row for row, keep in zip(development_records, calibration_mask) if keep
    ]
    calibration_selection = _root_selection(
        calibration_records, development_cost[calibration_mask]
    )
    outcome_selection = _root_selection(outcome_records, outcome_cost)

    combined_records = development_records + outcome_records
    combined_train_mask = np.concatenate([
        train_mask, np.zeros(len(outcome_records), dtype=bool)
    ])
    combined_calibration_mask = np.concatenate([
        calibration_mask, np.zeros(len(outcome_records), dtype=bool)
    ])
    combined_outcome_mask = np.concatenate([
        np.zeros(len(development_records), dtype=bool),
        np.ones(len(outcome_records), dtype=bool),
    ])
    combined_cost = np.concatenate([development_cost, outcome_cost])
    support = _train_only_root_support(combined_records, combined_train_mask, quantile=0.9)
    bounded_regret = _bounded_regret_report(
        calibration_selection, outcome_selection, support, alpha=alpha
    )

    train_candidate_cost = {}
    for candidate in candidate_ids:
        mask = train_mask & np.asarray([
            int(row["candidate_id"]) == candidate for row in development_records
        ])
        train_candidate_cost[candidate] = float(np.mean([
            float(row["real_task_cost"])
            for row, keep in zip(development_records, mask) if keep
        ]))
    bank_only = _root_selection(
        outcome_records,
        np.asarray([train_candidate_cost[int(row["candidate_id"])] for row in outcome_records]),
    )
    shuffled = _root_selection(
        outcome_records, _cyclic_within_root(outcome_cost, outcome_records)
    )
    no_dynamics = _root_selection(outcome_records, np.zeros(len(outcome_records)))
    paired_controls = {
        "bank_only": _paired_regret_gain(
            outcome_selection, bank_only, seed=20261017, samples=bootstrap_samples
        ),
        "shuffled_prediction": _paired_regret_gain(
            outcome_selection, shuffled, seed=20261016, samples=bootstrap_samples
        ),
        "no_dynamics": _paired_regret_gain(
            outcome_selection, no_dynamics, seed=20261018, samples=bootstrap_samples
        ),
    }

    matched = _matched_order_pair_audit(
        combined_records, combined_cost,
        train_mask=combined_train_mask,
        outcome_mask=combined_outcome_mask,
        samples=bootstrap_samples,
    )
    matched_calibration = _matched_order_pair_audit(
        combined_records, combined_cost,
        train_mask=combined_train_mask,
        outcome_mask=combined_calibration_mask,
        samples=bootstrap_samples,
    )
    matched_calibration_selection = {
        "roots": {
            root: {"regret": regret}
            for root, regret in matched_calibration["method"]["root_regret"].items()
        }
    }
    matched_outcome_selection = {
        "roots": {
            root: {"regret": regret}
            for root, regret in matched["method"]["root_regret"].items()
        }
    }
    matched["bounded_regret_certificate"] = _bounded_regret_report(
        matched_calibration_selection, matched_outcome_selection, support, alpha=alpha
    )

    heldout_candidate_ids = [
        candidate for index, candidate in enumerate(candidate_ids) if index % 4 == 3
    ]
    heldout_set = set(heldout_candidate_ids)
    new_train_mask = train_mask & np.asarray([
        int(row["candidate_id"]) not in heldout_set for row in development_records
    ])
    new_predictions = []
    for seed in SEEDS:
        prediction, _ = _fit_predict(
            development_x[new_train_mask], development_y[new_train_mask], outcome_x,
            seed, epochs, device,
        )
        new_predictions.append(prediction)
    new_effect = np.mean(np.stack(new_predictions), axis=0)
    new_cost = np.linalg.norm(outcome_sources + new_effect - outcome_goals, axis=1)
    heldout_mask = np.asarray([
        int(row["candidate_id"]) in heldout_set for row in outcome_records
    ])
    heldout_records = [row for row, keep in zip(outcome_records, heldout_mask) if keep]
    new_selection = _root_selection(heldout_records, new_cost[heldout_mask])
    new_shuffled = _root_selection(
        heldout_records,
        _cyclic_within_root(new_cost[heldout_mask], heldout_records),
    )
    new_gain = _paired_regret_gain(
        new_selection, new_shuffled, seed=20261019, samples=bootstrap_samples
    )

    report = {
        "schema": "pointmaze_external_candidate_effect_evaluation.v1",
        "development_dataset": str(development_dataset.resolve()),
        "outcome_dataset": str(outcome_dataset.resolve()),
        "summary": {
            "external_root_count": len(set(outcome_roots)),
            "effect_rmse": float(np.sqrt(np.mean((outcome_effect - outcome_y) ** 2))),
            "outcome_regret": outcome_selection["mean_regret"],
            "bank_only_regret": bank_only["mean_regret"],
            "shuffled_regret": shuffled["mean_regret"],
            "no_dynamics_regret": no_dynamics["mean_regret"],
        },
        "controls": {
            "paired_regret_gain": paired_controls,
            "matched_order": matched,
            "new_candidate": {
                "heldout_candidate_ids": heldout_candidate_ids,
                "selection": new_selection,
                "shuffled_prediction": new_shuffled,
                "paired_regret_gain_over_shuffled": new_gain,
            },
        },
        "bounded_regret_certificate": bounded_regret,
        "training": training,
        "protocol": {
            "feature_mode": feature_mode,
            "adapter_seeds": list(SEEDS),
            "epochs": int(epochs),
            "alpha": float(alpha),
            "development_training_split": "train roots only",
            "development_calibration_split": "calibration roots only",
            "outcome_split": "external_outcome roots only",
            "outcome_used_for_training_calibration_or_selection": False,
            "device": device,
        },
        "claim_boundary": (
            "External fixed-bank realized-state consequence for an intervention-supervised "
            "source-latent/action adapter; not frozen predictor-dynamics or adaptive-CEM evidence."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--development-arrays", type=Path, required=True)
    parser.add_argument("--outcome-dataset", type=Path, required=True)
    parser.add_argument("--outcome-arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--feature-mode",
        choices=("frozen_latent_action", "frozen_latent_action_basis"),
        default="frozen_latent_action_basis",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
