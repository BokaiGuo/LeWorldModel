from __future__ import annotations

import numpy as np
import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v12_decoupled_belief as v12


def test_decoupled_estimator_and_rankers_have_deployment_shapes() -> None:
    estimator = v12.VisualStateEstimator(history_dim=5, hidden=8)
    ranker = v12.StateTaskRanker(action_dim=2, hidden=8)
    action = v12.ActionOnly(action_dim=2, hidden=8)
    belief = estimator(torch.randn(3, 8, 5))
    assert belief.shape == (3, 4)
    assert ranker(belief, torch.randn(3, 20, 2)).shape == (3,)
    assert action(torch.randn(3, 20, 2)).shape == (3,)


def test_state_metrics_reward_exact_readout() -> None:
    value = np.asarray([[0.1, 0.2, 0.3, -0.4], [0.3, -0.1, -0.2, 0.5]])
    assert v12._state_metrics(value, value)["state_rmse"] == 0.0


def test_rgb_geometry_chart_requires_eight_visual_state_observations() -> None:
    row = {"source_history_visual_state": np.zeros((8, 4)).tolist()}
    geometry = v12._history_chart([row], "rgb_geometry")
    assert geometry.shape == (1, 8, 4)
