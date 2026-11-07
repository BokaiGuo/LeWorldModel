import pytest

from experiments.pc_pusht_effect_triple_alignment_20261107_codex import classify, count


def test_classify_ties_and_both_failure_modes():
    assert classify(0., 1., -1.)["category"] == "physical_tie"
    assert classify(1., 0.00001, -1.)["category"] == "score_tie"
    assert classify(1., 1., -1.)["category"] == "encoded_only_correct"
    assert classify(1., -1., 1.)["category"] == "predicted_only_correct"
    assert classify(1., -1., -1.)["category"] == "both_wrong"
    assert classify(1., 1., 1.)["category"] == "both_correct"


def test_counts_keep_physical_ties_out_of_decisive_denominator():
    rows = [{"classification": classify(0., 1., 1.)}, {"classification": classify(1., 1., -1.)}]
    totals = count(rows)
    assert totals["total"] == 2
    assert totals["physical_decisive"] == 1
    assert totals["encoded_sign_correct"] == 1
    assert totals["predicted_sign_correct"] == 0


def test_nonfinite_is_rejected():
    with pytest.raises(ValueError):
        classify(float("nan"), 1., 1.)
