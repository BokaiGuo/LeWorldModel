import pytest

from experiments.spectral_reachability import analyze_pusht_d2p_live_action_units as a
from tests.test_audit_pusht_d2p_live_action_units_cuda import fixture as rows_fixture


def fixture():
    rows = rows_fixture()
    return {"schema": "pusht_d2p_live_action_units.v1", "status": "completed", "gates": dict.fromkeys(a.GATES, True),
        "actual_predict_samples": 64, "sealed_protocol": {"checkpoint_training_statistics_certified": False,
            "physical_actions_unchanged_between_normalizations": True}, "summary": a.audit.summarize(rows), "rows": rows}


def test_full_coordinate_arithmetic_preserves_admission_boundary():
    result = a.summarize(fixture())
    assert result["p1_admitted"] is False
    assert result["training_stats_provenance_certified"] is False


@pytest.mark.parametrize("fault", ["gate", "partial", "summary", "stats", "actions", "samples"])
def test_partial_or_overclaimed_reports_rejected(fault):
    report = fixture()
    if fault == "gate": report["gates"]["cuda_only"] = False
    elif fault == "partial": report["rows"].pop()
    elif fault == "summary": report["summary"]["improved"] = 0
    elif fault == "stats": report["sealed_protocol"]["checkpoint_training_statistics_certified"] = True
    elif fault == "actions": report["sealed_protocol"]["physical_actions_unchanged_between_normalizations"] = False
    else: report["actual_predict_samples"] = 63
    with pytest.raises(ValueError): a.summarize(report)
