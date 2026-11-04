import pytest

from experiments.spectral_reachability.p1_residual_metric_alignment_20261102 import pairwise


def test_pairwise_direction_and_ties():
    rows = [{"arm": "a", "p": 1, "a": 3, "y": 1},
            {"arm": "b", "p": 2, "a": 2, "y": 2},
            {"arm": "c", "p": 2, "a": 2, "y": 0}]
    pairs = pairwise(rows, "p", "a", "y")
    assert len(pairs) == 3
    assert pairs[0]["predicted_actual_inversion"]
    assert pairs[0]["actual_physical_inversion"]
    assert pairs[-1]["contains_tie"]


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        pairwise([{"arm": "a", "p": float("nan"), "a": 0, "y": 1},
                  {"arm": "b", "p": 1, "a": 1, "y": 1}], "p", "a", "y")
