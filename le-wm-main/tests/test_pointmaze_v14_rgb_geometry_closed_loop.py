from __future__ import annotations

import numpy as np
import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v12_decoupled_belief as v12
from experiments.run_pointmaze_v14_rgb_geometry_closed_loop import _costs


def test_v14_cost_boundary_casts_json_float64_to_frozen_model_dtype() -> None:
    estimator = v12.VisualStateEstimator(history_dim=4, hidden=8)
    ranker = v12.StateTaskRanker(action_dim=2, hidden=8)
    action = v12.ActionOnly(action_dim=2, hidden=8)
    fit = {
        "history_mean": np.zeros(4),
        "history_scale": np.ones(4),
        "candidate_mean": np.zeros((20, 2)),
        "candidate_scale": np.ones((20, 2)),
        "target_mean": np.asarray(0.0),
        "target_scale": np.asarray(1.0),
    }
    geometry = np.zeros((8, 4), dtype=np.float64)
    bank = np.zeros((16, 20, 2), dtype=np.float64)

    belief_cost, action_cost = _costs(estimator, ranker, action, fit, geometry, bank)

    assert belief_cost.shape == (16,)
    assert action_cost.shape == (16,)
    assert next(estimator.parameters()).dtype is torch.float32
