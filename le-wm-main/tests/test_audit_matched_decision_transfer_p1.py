import json
from pathlib import Path

import numpy as np

from experiments.audit_matched_decision_transfer_p1 import (
    Candidate,
    audit_candidates,
    finite_horizon_action_conditioned_innovation,
    load_candidate,
)


def _candidate(seed: int, *, architecture: str = "arch", stage: int = 10, mse: float = 1.0, atm: float = 1.0, p1: float = 0.1, risk: float = 0.2) -> Candidate:
    return Candidate(
        source_path=f"record-{seed}", checkpoint=f"checkpoint-{seed}", environment="PointMaze",
        architecture=architecture, seed=str(seed), stage=stage, latent_dimension=4,
        action_distribution="fixed-v1", horizon_support=(1, 2), action_support_hash="support",
        mse=mse, atm=atm, p1=p1, p1_by_horizon={"1": p1, "2": p1}, planning_risk=risk,
    )


def test_p1_is_bounded_and_uses_only_fit_rows():
    rows = []
    for index in range(10):
        action = float(index - 5)
        rows.append({
            "action_blocks_normalized": [[action], [0.5 * action]],
            "latent_by_horizon": {
                "1": {"predicted": [0.0, 0.0], "target": [action, 0.5 * action]},
                "2": {"predicted": [0.0, 0.0], "target": [0.5 * action, action]},
            },
        })
    value, by_horizon = finite_horizon_action_conditioned_innovation({"enriched_records": rows}, (1, 2))
    assert 0.0 <= value <= 1.0
    assert set(by_horizon) == {"1", "2"}
    assert value > 0.1


def test_no_unmatched_pool_statistics_are_emitted():
    report = audit_candidates([
        _candidate(1, mse=1.0, atm=1.0), _candidate(2, mse=1.4, atm=1.4), _candidate(3, mse=1.8, atm=1.8),
    ], mse_tolerance=0.05, atm_tolerance=0.05, min_candidates=3)
    assert report["summary"]["status"] == "structural_deficiency_no_valid_matched_strata"
    assert report["heldout_incremental_risk"]["mean"] is None
    assert report["strata"][0]["selection"] is None


def test_p1_selector_is_evaluated_only_after_strict_matching():
    report = audit_candidates([
        _candidate(1, p1=0.5, risk=0.6), _candidate(2, p1=0.1, risk=0.1), _candidate(3, p1=0.4, risk=0.4),
        _candidate(4, architecture="other", p1=0.0, risk=0.0),
    ], min_candidates=3)
    row = next(item for item in report["strata"] if item["architecture"] == "arch")
    assert row["identifiable"] is True
    assert row["selection"]["p1_augmented_mse_atm"]["checkpoint"] == "checkpoint-2"
    assert row["selection"]["incremental_delta_risk_baseline_minus_p1"] >= 0.0


def test_loader_rejects_records_without_external_certificate(tmp_path: Path):
    fit = {
        "horizons": [1],
        "enriched_records": [
            {"action_blocks_normalized": [[float(index)]], "latent_by_horizon": {"1": {"predicted": [0.0], "target": [float(index)]}}}
            for index in range(8)
        ],
    }
    fit_path = tmp_path / "results" / "fit.json"
    fit_path.parent.mkdir()
    fit_path.write_text(json.dumps(fit))
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps({
        "record_schema": "real_checkpoint_selection.v1",
        "protocol": {"environment": "PointMaze", "independent_external_evidence": False},
        "record": {
            "architecture": "arch", "checkpoint": "x", "group": "PointMaze/seed_1",
            "model_factors": {"latent_dimension": 1, "training_stage": 1, "action_distribution": "a"},
            "diagnostics": {"prediction_mse": 0.1, "action_inverse_risk": 0.1},
            "heldout": {"planning_risk": 0.1}, "provenance": {"fit_export": "results/fit.json"},
        },
    }))
    try:
        load_candidate(record_path)
    except ValueError as exc:
        assert "independent_external_evidence" in str(exc)
    else:
        raise AssertionError("missing external certificate must fail closed")


def test_loader_uses_fit_export_without_opening_outcome_export(tmp_path: Path):
    fit = {
        "horizons": [1],
        "seed": "1",
        "enriched_records": [
            {"action_blocks_normalized": [[float(index)]], "latent_by_horizon": {"1": {"predicted": [0.0], "target": [float(index)]}}}
            for index in range(8)
        ],
    }
    fit_path = tmp_path / "results" / "fit.json"
    fit_path.parent.mkdir()
    fit_path.write_text(json.dumps(fit))
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps({
        "record_schema": "real_checkpoint_selection.v1",
        "protocol": {"environment": "PointMaze", "independent_external_evidence": True},
        "record": {
            "architecture": "arch", "checkpoint": "x", "group": "PointMaze/seed_1",
            "model_factors": {"latent_dimension": 1, "training_stage": 1, "action_distribution": "a"},
            "diagnostics": {"prediction_mse": 0.1, "action_inverse_risk": 0.1},
            "heldout": {"planning_risk": 0.1}, "provenance": {"fit_export": "results/fit.json"},
        },
    }))
    candidate = load_candidate(record_path)
    assert candidate.p1 > 0.0
    assert candidate.horizon_support == (1,)
