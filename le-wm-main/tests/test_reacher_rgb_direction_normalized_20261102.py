import pytest
import torch

from experiments.spectral_reachability import reacher_rgb_direction_normalized_20261102 as method


def test_relative_direction_weight_is_dimension_independent():
    for size in (2, 192):
        goal = torch.zeros(size)
        observed = torch.zeros(size)
        observed[0] = 1
        predictions = torch.zeros(2, size)
        predictions[0, 0] = predictions[1, 1] = 1
        scores = method.direction_costs(predictions, goal, observed)
        assert torch.allclose(scores[0] / scores[1], torch.tensor(1.1))


def test_invalid_direction_fails_closed():
    with pytest.raises(ValueError):
        method.direction_costs(torch.zeros(1, 2), torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError):
        method.direction_costs(torch.tensor([[float("nan"), 0.]]), torch.zeros(2), torch.ones(2))


def test_saved_pairing_recomputes():
    import json
    from experiments.spectral_reachability import audit_reacher_rgb_direction_normalized_20261102 as audit

    root = method.base.ROOT / "results"
    path = root / "reacher_rgb_direction_normalized_SAVED_20261102_v1.json"
    if not path.exists():
        pytest.skip("development physical run not present")
    result = audit.consume(
        root / "reacher_rgb_direction_normalized_PLAN_20261102_v1.json",
        "d9e505458e5a825c90b3c1c619ca3a2530723f3e828f881afa9ba887dcea2bbe",
        root / "reacher_response_svd_cp5_rgb_direction_normalized_RUN_20261102_v1/events.jsonl",
        "2a69e3047366a19914a5f6c59a672a40bfe1569f0cc3346c8741f86a83d3b15b")
    assert result == json.loads(path.read_text())
    assert result["suffix_encode_calls"] == 104
    assert result["suffix_predict_samples"] == 1664
