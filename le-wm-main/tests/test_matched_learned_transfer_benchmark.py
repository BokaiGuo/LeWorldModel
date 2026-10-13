from __future__ import annotations

import numpy as np

from experiments.empirical_spectral_audit.run_matched_learned_transfer_benchmark import audit


def _record(
    architecture: str,
    seed: int,
    environment: str,
    stage: int,
    mse: float,
    spectrum_shift: float,
    risk: float,
    alignment: float,
    schur: float | None = 0.2,
) -> dict:
    diagnostics = {
        "prediction_mse": mse,
        "predictor_spectrum": [
            {"real": 0.8 + spectrum_shift, "imag": 0.0},
            {"real": 0.4, "imag": 0.0},
        ],
        "task_metric_alignment": alignment,
        "task_metric_commutator": alignment,
        "task_metric_schur_offdiagonal_energy": schur,
        "task_metric_transient_amplification": alignment,
    }
    return {
        "checkpoint": f"{architecture}/seed_{seed}/epoch_{stage}",
        "group": f"{environment}/in_distribution/{architecture}/seed_{seed}",
        "model_factors": {
            "latent_dimension": 2,
            "training_stage": stage,
            "action_distribution": "fixed-action-v1",
        },
        "diagnostics": diagnostics,
        "heldout": {"planning_loss": risk},
        "protocol": {"environment": environment},
        "_architecture": architecture,
        "_seed": str(seed),
        "_environment": environment,
        "_stage": stage,
        "_action_distribution": "fixed-action-v1",
        "_latent_dimension": 2,
        "_spectrum": np.asarray([0.8 + spectrum_shift, 0.4], dtype=np.complex128),
        "_missing_metrics": [] if schur is not None else ["task_metric_schur_offdiagonal_energy"],
    }


def test_primary_audit_requires_spectrum_matched_cross_model_pool() -> None:
    records = [
        _record("jepa", 1, "PushT", 50, 1.00, 0.0, 0.9, 0.9),
        _record("dino", 2, "PushT", 50, 1.02, 0.0, 0.2, 0.1),
        _record("jepa", 3, "PointMaze", 50, 1.01, 0.0, 0.8, 0.8),
        _record("dino", 4, "PointMaze", 50, 1.02, 0.0, 0.3, 0.1),
    ]
    report = audit(records)
    assert report["summary"]["status"] == "gated_positive_delta_not_established"
    assert report["summary"]["claim_ready_for_positive_delta"] is False
    assert report["summary"]["primary_delta_risk_mse_minus_full"]["positive_gate"] is False
    assert report["summary"]["identifiable_strata"] == 2
    assert report["summary"]["environment_count"] == 2


def test_spectrum_mismatch_is_fail_closed() -> None:
    report = audit([
        _record("jepa", 1, "PushT", 50, 1.00, 0.0, 0.9, 0.9),
        _record("dino", 2, "PushT", 50, 1.02, 0.2, 0.2, 0.1),
    ])
    assert report["summary"]["status"] == "gated_no_identifiable_matched_stratum"
    assert report["strata"][0]["reason_if_gated"] == "fewer_than_two_quality_and_spectrum_matched_candidates"


def test_missing_schur_is_not_promoted() -> None:
    report = audit([
        _record("jepa", 1, "PushT", 50, 1.00, 0.0, 0.9, 0.9, schur=None),
        _record("dino", 2, "PushT", 50, 1.02, 0.0, 0.2, 0.1, schur=None),
    ])
    assert report["summary"]["status"] == "gated_missing_schur_metric"
    assert report["strata"][0]["identifiable"] is False


def test_pairwise_matching_recovers_pair_outside_global_best_mse_anchor() -> None:
    # The lowest-MSE candidate is deliberately incompatible with both
    # candidates in the valid pair.  A global-anchor rule would therefore
    # report no matched stratum, while the declared pairwise rule must retain
    # the cross-architecture, cross-seed pair.
    report = audit([
        _record("anchor", 1, "PushT", 75, 1.00, 0.00, 0.9, 0.9),
        _record("phase", 2, "PushT", 75, 1.10, 0.06, 0.2, 0.2),
        _record("scalar", 3, "PushT", 75, 1.14, 0.10, 0.3, 0.3),
    ])
    row = report["strata"][0]
    assert row["eligible_count"] == 2
    assert row["eligible_architectures"] == ["phase", "scalar"]
    assert row["eligible_seeds"] == ["2", "3"]
    assert row["identifiable"] is True
    assert report["summary"]["identifiable_strata"] == 1


def test_diverse_pair_beats_larger_single_architecture_clique() -> None:
    # Three same-architecture checkpoints form a larger clique, but the
    # smaller phase/scalar pair is the only clique satisfying the declared
    # cross-architecture and cross-seed estimand.
    report = audit([
        _record("scalar", 1, "PushT", 75, 1.00, 0.00, 0.9, 0.9),
        _record("scalar", 2, "PushT", 75, 1.01, 0.00, 0.8, 0.8),
        _record("scalar", 3, "PushT", 75, 0.99, 0.00, 0.7, 0.7),
        _record("phase", 4, "PushT", 75, 1.10, 0.06, 0.2, 0.2),
        _record("scalar", 5, "PushT", 75, 1.14, 0.10, 0.3, 0.3),
    ])
    row = report["strata"][0]
    assert row["eligible_count"] == 2
    assert row["eligible_architectures"] == ["phase", "scalar"]
    assert row["eligible_seeds"] == ["4", "5"]
    assert row["identifiable"] is True


def test_cross_environment_status_requires_a_matched_stratum_in_each_environment() -> None:
    report = audit([
        _record("jepa", 1, "PushT", 50, 1.00, 0.0, 0.9, 0.9),
        _record("dino", 2, "PushT", 50, 1.02, 0.0, 0.2, 0.1),
        _record("jepa", 3, "PointMaze", 50, 1.00, 0.0, 0.8, 0.8),
        _record("dino", 4, "PointMaze", 50, 1.40, 0.2, 0.3, 0.1),
    ])
    assert report["summary"]["identifiable_environment_count"] == 1
    assert report["summary"]["status"] == "gated_insufficient_environment_replication"
