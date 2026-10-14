from __future__ import annotations

import json

import numpy as np

from experiments.decision_benchmark.train_rgb_candidate_successor_adapter import _features


def test_features_use_source_plus_complete_action_and_residual_target():
    records = [{
        "instance_id": "r0",
        "candidate_id": 0,
        "split": "train",
        "action_sequence": [[1.0, 2.0], [3.0, 4.0]],
    }]
    rgb = {
        ("r0", 0): {
            "source_projection": [0.5, 1.5],
            "realized_terminal_projection": [2.5, 4.5],
        }
    }
    x, residual, source = _features(records, rgb)
    np.testing.assert_allclose(x, [[0.5, 1.5, 1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(residual, [[2.0, 3.0]])
    np.testing.assert_allclose(source, [[0.5, 1.5]])


def test_features_can_use_complete_causal_rgb_history():
    records = [{
        "instance_id": "r0",
        "candidate_id": 0,
        "split": "train",
        "action_sequence": [[1.0, 2.0]],
    }]
    rgb = {
        ("r0", 0): {
            "source_projection": [3.0, 4.0],
            "source_history_projection": [[1.0, 2.0], [3.0, 4.0]],
            "realized_terminal_projection": [5.0, 8.0],
        }
    }
    x, residual, source = _features(records, rgb, history_windows=2)
    np.testing.assert_allclose(x, [[1.0, 2.0, 3.0, 4.0, 1.0, 2.0]])
    np.testing.assert_allclose(residual, [[2.0, 4.0]])
    np.testing.assert_allclose(source, [[3.0, 4.0]])
