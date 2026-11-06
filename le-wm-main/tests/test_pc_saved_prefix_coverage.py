import pytest

from experiments.pc_saved_prefix_coverage import coverage


def test_generated_reference_can_be_pruned():
    layers = [{"depth": 0, "candidates": [{"chain": [[0., 0.]], "score": 0.},
                                           {"chain": [[1., 1.]], "score": 1.}],
               "retained": [0]}]
    row = coverage(layers, [[1., 1.]], 1)[0]
    assert row["nearest_rms"] == 0
    assert not row["nearest_was_retained"]
    assert row["nearest_retained_rms"] == 1


def test_rejects_duplicate_retention():
    layer = {"depth": 0, "candidates": [{"chain": [[0.]], "score": 0.}], "retained": [0, 0]}
    with pytest.raises(ValueError, match="retention"):
        coverage([layer], [[0.]], 1)
