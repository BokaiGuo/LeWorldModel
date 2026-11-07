import numpy as np
import pytest

from experiments.pc_tworoom_cem16_gpu_replay_20261107_codex import check_score_record, rebuild_round


def test_score_check_detects_wrong_selection_and_score_drift():
    values = np.arange(16, dtype=float)
    good = check_score_record(values, values[1:], 0, 0)
    assert good["selection_agrees"] and good["within_tolerance"]
    bad = check_score_record(values, values[1:] + 0.1, 1, 1)
    assert not bad["selection_agrees"] and not bad["within_tolerance"]
    with pytest.raises(ValueError):
        check_score_record(values, [np.nan] * 15, 0, 0)


def test_rebuild_is_seeded_and_clipped():
    sample = rebuild_round(np.random.default_rng(31), np.zeros((5, 2)), np.ones((5, 2)), -0.4, 0.4)
    other = rebuild_round(np.random.default_rng(31), np.zeros((5, 2)), np.ones((5, 2)), -0.4, 0.4)
    np.testing.assert_array_equal(sample, other)
    assert sample.shape == (5, 5, 2) and np.max(np.abs(sample)) <= 0.4
