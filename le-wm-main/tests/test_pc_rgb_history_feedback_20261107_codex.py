import numpy as np
import pytest

from experiments.pc_rgb_history_feedback_20261107_codex import ObservedHistory


def test_history_evicts_matching_oldest_packet_and_never_fills():
    history = ObservedHistory(np.zeros((2, 2, 3), dtype=np.uint8))
    for i in range(1, 4):
        history.append(np.full((5, 2), i / 10), np.full((2, 2, 3), i, dtype=np.uint8))
    frames, packets = history.view("history")
    assert [x[0, 0, 0] for x in frames] == [1, 2, 3]
    np.testing.assert_allclose([x[0, 0] for x in packets], [.2, .3])
    assert len(history.view("single")[0]) == 1
    assert history.view("single")[1] == []


def test_rejects_incomplete_or_illegal_actions():
    history = ObservedHistory(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        history.append(np.zeros((4, 2)), np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        history.append(np.full((5, 2), 2), np.zeros((2, 2, 3), dtype=np.uint8))
    assert len(history.frames) == 1
