"""NumPy/AST fixtures only, not model or physical evidence."""
import pytest

from experiments.spectral_reachability.audit_pusht_d2p_metric_program_arrays import audit
from test_pusht_d2p_metric_program_compiler import fixture_caller, inputs


def fixture():
    compile_programs, _ = fixture_caller()
    return compile_programs(**inputs())


def test_augmented_lstsq_independently_matches_caller_arrays():
    result = audit(fixture())
    assert result["independent_augmented_lstsq_fits"] == 2
    assert result["P1_admission"] is False


@pytest.mark.parametrize("field", ["stage_cost", "own_reach_value", "objective", "ridge_penalty"])
def test_wrong_score_rejected(field):
    report = fixture()
    report["scoring"]["rows"][0][field] += .1
    with pytest.raises(ValueError, match="mismatch"):
        audit(report)


def test_wrong_live_action_rejected():
    report = fixture()
    report["receipts"][0]["program_live_commands"][0][0] = .2
    with pytest.raises(ValueError, match="live commands"):
        audit(report)


def test_nonlinear_diagnostic_cannot_be_selection():
    report = fixture()
    report["selected_candidate_id"] = "prefix_1"
    with pytest.raises(ValueError, match="selection"):
        audit(report)
