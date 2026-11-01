import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_source_temporal_alignment import (
    _packetize,
    _valid_clip_starts,
)


def test_packetize_declared_alignment_preserves_three_consecutive_time_major_packets():
    actions = np.arange(80, dtype=np.float32).reshape(40, 2)
    actual = _packetize(actions, start=5, packet_shift=0)
    expected = actions[5:20].reshape(3, 10)
    np.testing.assert_array_equal(actual, expected)


def test_packetize_neighbour_shift_moves_by_exactly_one_five_step_packet():
    actions = np.arange(80, dtype=np.float32).reshape(40, 2)
    np.testing.assert_array_equal(
        _packetize(actions, start=5, packet_shift=-1), actions[0:15].reshape(3, 10),
    )
    np.testing.assert_array_equal(
        _packetize(actions, start=5, packet_shift=1), actions[10:25].reshape(3, 10),
    )


def test_valid_clips_exclude_episode_boundaries_needed_by_all_alignment_probes():
    clips = _valid_clip_starts(np.asarray([25], dtype=np.int64), packet_shifts=(-1, 0, 1))
    assert clips == [(0, 5)]
    with pytest.raises(ValueError):
        _valid_clip_starts(np.asarray([24], dtype=np.int64), packet_shifts=(-1, 0, 1))
