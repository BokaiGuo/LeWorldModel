import pytest

from experiments.spectral_reachability import p1_second_feedback_prediction_audit_20261102 as subject


@pytest.mark.parametrize("values,expected", [
    ((1., 2., 3., 4.), "same"),
    ((1., 2., 4., 3.), "opposite"),
    ((1., 1., 4., 3.), "tie"),
])
def test_distance_ranking(values, expected):
    assert subject.classify(*values)["ranking"] == expected


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        subject.classify(1, 2, float("nan"), 3)


def test_fixed_development_panel():
    spec = subject.protocol()
    assert len(spec["rows"]) == 8
    assert spec["per_arm"]["predictor_calls"] == 2
    assert spec["no_selection_or_physical_execution"]
    assert not spec["P1_admission"]
