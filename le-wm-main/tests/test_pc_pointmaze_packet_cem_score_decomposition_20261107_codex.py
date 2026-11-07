import pytest

from experiments.pc_pointmaze_packet_cem_score_decomposition_20261107_codex import relation


def test_relation_preserves_ties_and_direction():
    assert relation([1, 2], [1, 2]) is True
    assert relation([1, 2], [2, 1]) is False
    assert relation([1, 1], [2, 1]) is None
    with pytest.raises(ValueError):
        relation([1], [2])
