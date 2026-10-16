import numpy as np

from experiments.audit_reacher_sequence_candidate_ranking import _sequence_bank


def test_sequence_bank_has_fixed_shared_shape_and_three_families():
    mean = np.zeros(10, dtype=np.float32)
    std = np.ones(10, dtype=np.float32)
    raw, normalized, labels = _sequence_bank(mean, std, horizon=5, action_block=5)
    assert raw.shape == (161, 5, 2)
    assert tuple(normalized.shape) == (161, 5, 10)
    assert len(labels) == 161
    assert {label["family"] for label in labels} == {"constant", "pulse", "pulse_recovery"}


def test_compact_sequence_bank_preserves_continuous_exploration_budget():
    mean = np.zeros(10, dtype=np.float32)
    std = np.ones(10, dtype=np.float32)
    raw, normalized, labels = _sequence_bank(
        mean, std, horizon=5, action_block=5, bank_family="compact",
    )
    assert raw.shape == (49, 5, 2)
    assert tuple(normalized.shape) == (49, 5, 10)
    assert len(labels) == 49


def test_support4_bank_preserves_nearly_all_continuous_exploration_budget():
    mean = np.zeros(10, dtype=np.float32)
    std = np.ones(10, dtype=np.float32)
    raw, normalized, labels = _sequence_bank(
        mean, std, horizon=5, action_block=5, bank_family="support4",
    )
    assert raw.shape == (4, 5, 2)
    assert tuple(normalized.shape) == (4, 5, 10)
    assert len(labels) == 4
    assert {label["family"] for label in labels} == {"pulse_recovery"}


def test_sequence_bank_normalizes_each_action_block():
    mean = np.full(10, 0.5, dtype=np.float32)
    std = np.full(10, 2.0, dtype=np.float32)
    raw, normalized, _ = _sequence_bank(mean, std, horizon=5, action_block=5)
    constant_zero = int(np.flatnonzero(np.all(np.isclose(raw, 0.0), axis=(1, 2)))[0])
    assert np.allclose(normalized[constant_zero], -0.25)
