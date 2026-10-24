import pytest

from experiments.spectral_reachability.select_reacher_d2p_model_physical_admission import select


def _report(physical_deltas):
    return {
        "schema": "d2p_reacher_p0_goal_progress.v1",
        "status": "calibration_only",
        "protocol": {"calibration_only": True, "outcome_only": False},
        "gates": {"cuda_only": True, "frozen": True},
        "rows": [
            {"seed": index, "split": "calibration", "methods": {
                "d2p_goal_progress": {"predicted_terminal_cost": 0.0, "physical": {"final_qpos_distance": 1.0 + delta}},
                "cem": {"predicted_terminal_cost": 1.0, "physical": {"final_qpos_distance": 1.0}},
            }}
            for index, delta in enumerate(physical_deltas)
        ],
    }


def test_calibration_admission_requires_exact_sign_evidence_before_outcome():
    result = select(_report([-0.1] * 7 + [0.1]))

    assert result["status"] == "pass"
    assert result["strict_order_agreement_count"] == 7
    assert result["one_sided_exact_sign_p"] < 0.05


def test_calibration_admission_fails_closed_for_insufficient_agreement():
    result = select(_report([-0.1] * 6 + [0.1] * 2))

    assert result["status"] == "model_physical_calibration_failed"
    assert result["gates"]["model_physical_admitted"] is False


def test_calibration_admission_rejects_ties_and_non_calibration_rows():
    with pytest.raises(ValueError, match="strict paired"):
        select(_report([-0.1] * 7 + [0.0]))
    report = _report([-0.1] * 8)
    report["rows"][0]["split"] = "outcome"
    with pytest.raises(ValueError, match="insufficient or non-calibration"):
        select(report)
