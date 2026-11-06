import pytest

from experiments.pc_prefix_heldout_physical_dev_20261106_codex import summarize


def test_evaluates_only_frozen_retained_prefixes():
    physical = {"B": [True] + [False] * 11, "optimized": [False] * 11 + [True]}
    result = summarize(physical, {"probe": {"retained": [0, 1, 2, 3]}})
    assert result["all"] == {"B": 1, "optimized": 1}
    assert result["probe"] == {"B": 1, "optimized": 0}
    with pytest.raises(ValueError):
        summarize(physical, {"invalid": {"retained": [0, 0, 1, 2]}})
