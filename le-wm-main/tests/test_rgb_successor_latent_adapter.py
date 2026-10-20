from __future__ import annotations

import numpy as np
import pytest

from experiments.gdd_wm.train_rgb_successor_latent_adapter import _features


def _row() -> dict:
    return {
        "source_latent_model": [0.1, -0.2],
        "_realized_latent": [0.3, 0.4],
        "_realized_state": [0.7, 0.8],
        "source_state_raw": [0.0, 0.0, 0.0, 0.0],
        "action_sequence": [[0.2, -0.1]] * 20,
        "instance_id": "root-0",
    }


def test_successor_features_use_realized_terminal_state_for_readout_targets() -> None:
    x, y, source, states, roots = _features([_row()], "basis")
    assert x.shape[0] == y.shape[0] == 1
    assert np.allclose(y[0], np.asarray([0.2, 0.6]))
    assert np.allclose(states[0], np.asarray([0.7, 0.8]))
    assert roots == ["root-0"]


def test_successor_features_reject_unknown_action_chart() -> None:
    with pytest.raises(ValueError, match="feature_mode"):
        _features([_row()], "unknown")
