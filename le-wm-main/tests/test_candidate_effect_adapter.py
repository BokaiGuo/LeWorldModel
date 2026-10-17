from __future__ import annotations

import numpy as np
import pytest

from experiments.decision_benchmark.train_candidate_effect_adapter import (
    _features,
    action_sequence_basis,
)


def test_action_sequence_basis_is_fixed_width_and_finite() -> None:
    actions = np.linspace(-0.75, 0.75, 40, dtype=np.float64).reshape(20, 2)
    basis = action_sequence_basis(actions)

    assert basis.shape == (16,)
    assert np.isfinite(basis).all()


def test_action_sequence_basis_distinguishes_temporal_order() -> None:
    first = np.tile(np.asarray([0.75, 0.0]), (10, 1))
    second = np.tile(np.asarray([0.0, 0.5]), (10, 1))
    left = np.concatenate([first, second], axis=0)
    right = np.concatenate([second, first], axis=0)

    assert np.allclose(left.sum(axis=0), right.sum(axis=0))
    assert not np.allclose(action_sequence_basis(left), action_sequence_basis(right))


def test_action_sequence_basis_rejects_malformed_or_short_sequences() -> None:
    with pytest.raises(ValueError, match="shape"):
        action_sequence_basis(np.zeros((20, 3)))
    with pytest.raises(ValueError, match="one step per bin"):
        action_sequence_basis(np.zeros((3, 2)))


def test_frozen_latent_action_basis_does_not_include_privileged_state() -> None:
    records = [{
        "instance_id": "root",
        "source_latent_model": [1.0, 2.0, 3.0],
        "source_state_raw": [100.0, 200.0, 300.0, 400.0],
        "goal_raw": [0.0, 0.0],
        "action_sequence": np.zeros((20, 2)).tolist(),
        "realized_terminal_state_ref": "arrays.npz::terminal",
    }]
    arrays = {"terminal": np.asarray([101.0, 201.0, 300.0, 400.0])}

    features, effects, _, _ = _features(
        records, arrays, feature_mode="frozen_latent_action_basis"
    )

    assert features.shape == (1, 3 + 16)
    assert np.allclose(features[0, :3], [1.0, 2.0, 3.0])
    assert not np.isin([100.0, 200.0, 300.0, 400.0], features).any()
    assert np.allclose(effects, [[1.0, 1.0]])
