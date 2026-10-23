from __future__ import annotations

import numpy as np
import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v8_belief_state as v8


def test_belief_ranker_exposes_scalar_cost_and_four_state_coordinates() -> None:
    model = v8.VisualBeliefStateTaskRanker(history_dim=5, action_dim=2, hidden=8)
    cost, belief = model(torch.randn(3, 8, 5), torch.randn(3, 20, 2))
    assert cost.shape == (3,)
    assert belief.shape == (3, 4)


def test_belief_film_starts_without_unearned_state_effect() -> None:
    model = v8.VisualBeliefStateTaskRanker(history_dim=5, action_dim=2, hidden=8)
    assert torch.count_nonzero(model.film.weight) == 0
    assert torch.count_nonzero(model.film.bias) == 0


def test_policy_cost_selects_only_declared_arm() -> None:
    belief, direct, action = np.asarray([1.0]), np.asarray([2.0]), np.asarray([3.0])
    assert np.array_equal(v8._policy_cost("belief_state", belief, direct, action), belief)
    assert np.array_equal(v8._policy_cost("direct_visual", belief, direct, action), direct)
    assert np.array_equal(v8._policy_cost("action_only", belief, direct, action), action)


def test_state_metrics_identify_exact_belief() -> None:
    state = np.asarray([[0.2, -0.1, 0.3, -0.4], [0.1, 0.5, -0.2, 0.1]])
    metrics = v8._state_metrics(state, state.copy())
    assert metrics["state_rmse"] == 0.0
    assert metrics["velocity_sign_accuracy"] == 1.0
