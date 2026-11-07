import pytest

from experiments.pc_tworoom_render_input_decision_audit_20261107_codex import score_summary


def test_score_summary_records_stable_order_and_top_overlap():
    result = score_summary([1., 0., 0., 4.], [1., 0., 2., 0.], n=2)
    assert (result["left_best"], result["right_best"], result["top_intersection"]) == (1, 1, 1)
    assert result["left_gap"] == 0
    assert result["max_abs_difference"] == 4


def test_score_summary_fails_closed_on_nonfinite_or_shape():
    with pytest.raises(ValueError):
        score_summary([1., float("nan")], [1., 2.], n=1)
    with pytest.raises(ValueError):
        score_summary([1.], [1., 2.], n=1)
