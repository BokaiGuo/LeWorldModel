import numpy as np

from experiments.empirical_spectral_audit.run_frozen_predictor_bootstrap import (
    _action_labels,
    _cosine_rows,
    _summary_for_rows,
)


def test_cosine_and_action_labels_are_well_defined():
    true = np.eye(4, dtype=np.float32)
    predicted = true.copy()
    assert np.allclose(_cosine_rows(true, predicted), 1.0)
    labels = _action_labels(np.arange(20, dtype=np.float32).reshape(10, 2), 2)
    assert set(labels.tolist()) == {0, 1}


def test_paired_alignment_beats_permutation_null_on_constructed_data():
    rng = np.random.default_rng(7)
    true = rng.normal(size=(256, 8)).astype(np.float32)
    predicted = (true + 0.01 * rng.normal(size=true.shape)).astype(np.float32)
    actions = rng.normal(size=(256, 2)).astype(np.float32)
    labels = _action_labels(actions, 2)
    report = _summary_for_rows(true, predicted, labels, np.random.default_rng(9), 40, 40)
    assert report["paired_cosine"]["mean"] > 0.99
    assert report["cosine_gap_paired_minus_null_mean"] > 0.5
    assert report["row_permutation_null_cosine"]["p_value_mean_null_ge_paired"] < 0.1
