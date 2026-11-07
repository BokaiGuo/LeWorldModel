import pytest

from experiments.pc_tworoom_cem_prefix_feedback_20261107_codex import choose


def test_choice_and_stable_tie():
    assert choose([1, 2]) == 0
    assert choose([2, 1]) == 1
    assert choose([1, 1]) == 0


def test_invalid_scores():
    with pytest.raises(ValueError):
        choose([1])
    with pytest.raises(ValueError):
        choose([1, float("nan")])
