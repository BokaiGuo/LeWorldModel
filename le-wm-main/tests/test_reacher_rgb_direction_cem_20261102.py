import pytest
import torch

from experiments.spectral_reachability import reacher_response_svd_rgb_direction_cem_20261102 as method


def test_direction_costs_and_batch_order():
    goal = torch.zeros(2)
    observed = torch.tensor([1.0, 0.0])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert torch.allclose(method.direction_costs(candidates, goal, observed), torch.tensor([0.6, 0.5, 0.0]))


def test_direction_fails_closed_for_zero_and_nonfinite():
    goal = torch.zeros(2)
    with pytest.raises(ValueError):
        method.direction_costs(goal[None], goal, goal)
    with pytest.raises(ValueError):
        method.direction_costs(torch.tensor([[float("nan"), 0.0]]), goal, torch.ones(2))


def test_direction_invariant_to_observation_scaling():
    goal = torch.zeros(2)
    predicted = torch.tensor([[0.5, 0.2], [-0.1, 0.3]])
    observed = torch.tensor([1.0, 2.0])
    assert torch.allclose(method.direction_costs(predicted, goal, observed),
                          method.direction_costs(predicted, goal, observed * 3))


def test_penalty_scale_against_192_dimensional_mean_l2():
    goal = torch.zeros(192)
    observed = torch.zeros(192)
    observed[0] = 1
    predictions = torch.zeros(2, 192)
    predictions[0, 0] = 1
    predictions[1, 1] = 1
    scores = method.direction_costs(predictions, goal, observed)
    assert torch.allclose(scores[0] / scores[1], torch.tensor(20.2), atol=1e-5)


def test_saved_run_has_full_pairing():
    import json

    path = method.base.ROOT / "results/reacher_response_svd_rgb_direction_SAVED_20261102_v2.json"
    if not path.exists():
        pytest.skip("development physical run not present")
    result = json.loads(path.read_text())
    assert result["status"] == "complete_saved_consistency"
    assert len(result["paired_rows"]) == 26
    assert all("direction_final" in row and "q_final" not in row for row in result["paired_rows"])
