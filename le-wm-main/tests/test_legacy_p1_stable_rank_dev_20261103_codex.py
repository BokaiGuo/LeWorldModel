import itertools

import pytest

from experiments.empirical_spectral_audit import legacy_p1_stable_rank_dev_20261103_codex as dev


def record(name, values):
    return {"checkpoint": name, "diagnostics": dict(zip(dev.METRICS, values))}


def test_all_permutations_average_ties_and_composite_tie():
    records = [record("b", [1, 2, 1, 1, 1, 1]),
               record("a", [2, 1, 1, 1, 1, 1]),
               record("c", [3, 3, 2, 2, 2, 2])]
    result = dev.select(records)
    assert result["full"] == "a"
    assert result["mse"] == "b"
    assert result["ranks"]["a"][2:] == [0.5] * 4
    assert result["scores"]["a"] == result["scores"]["b"]
    for permutation in itertools.permutations(records):
        assert dev.select(permutation) == result


def test_mse_tie_uses_identity():
    assert dev.select([record("z", [1] * 6), record("a", [1] * 6)])["mse"] == "a"


@pytest.mark.parametrize("records", [
    [], [record("", [1] * 6)], [record(" x", [1] * 6)],
    [record("a", [1] * 6), record("a", [2] * 6)],
    [{"diagnostics": dict.fromkeys(dev.METRICS, 1)}],
    [record("a", [float("nan")] * 6)], [record("a", [float("inf")] * 6)],
    [record("a", [1] * 5)],
])
def test_fail_closed(records):
    with pytest.raises(ValueError):
        dev.select(records)
