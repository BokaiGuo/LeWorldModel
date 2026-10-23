from __future__ import annotations

import torch

from experiments.decision_benchmark.audit_pointmaze_v7_privileged_state_ceiling import StateTaskRanker


def test_privileged_state_ranker_has_scalar_cost_for_each_candidate() -> None:
    model = StateTaskRanker(action_dim=2, hidden=8)
    scores = model(torch.randn(3, 4), torch.randn(3, 20, 2))
    assert scores.shape == (3,)


def test_privileged_state_ranker_film_is_zero_initialized() -> None:
    model = StateTaskRanker(action_dim=2, hidden=8)
    assert torch.count_nonzero(model.film.weight) == 0
    assert torch.count_nonzero(model.film.bias) == 0
