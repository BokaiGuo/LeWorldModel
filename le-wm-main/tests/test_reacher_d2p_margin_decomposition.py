import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_margin_decomposition import decompose_margin


def test_exact_decomposition_distinguishes_common_bias_from_response_error():
    generator = torch.Generator().manual_seed(21)
    values = [torch.randn(192, generator=generator) for _ in range(5)]
    row = decompose_margin(*values)
    assert row["decomposition_abs_error"] < 1e-12
    assert row["margin_error"] == pytest.approx(row["zero_successor_bias_contribution"]
        + row["response_linear_error_contribution"] + row["response_quadratic_error_contribution"])


def test_identical_response_with_shared_bias_can_reverse_cost_order():
    pz, p = torch.tensor([0.]), torch.tensor([1.])
    oz, o, goal = torch.tensor([2.]), torch.tensor([3.]), torch.tensor([1.])
    row = decompose_margin(p, pz, o, oz, goal)
    assert row["predicted_margin"] == -1
    assert row["observed_margin"] == 3
    assert row["zero_successor_bias_contribution"] == 4
    assert row["response_linear_error_contribution"] == 0
    assert row["response_quadratic_error_contribution"] == 0
    assert row["oracle_zero_anchored_margin"] == 3
    assert row["action_response_cosine"] == 1
    assert row["oracle_future_zero_successor_used"] is True


def test_anchor_cannot_correct_wrong_response():
    row = decompose_margin(torch.tensor([1.]), torch.tensor([0.]),
        torch.tensor([-1.]), torch.tensor([0.]), torch.tensor([2.]))
    assert row["zero_successor_bias_contribution"] == 0
    assert row["oracle_zero_anchored_margin"] < 0 < row["observed_margin"]
    assert row["action_response_cosine"] == -1


@pytest.mark.parametrize("bad", [torch.ones(1, 2), torch.ones(3), torch.tensor([float("nan"), 0.])])
def test_bad_latent_contract_fails_closed(bad):
    with pytest.raises(ValueError):
        decompose_margin(bad, torch.ones(2), torch.ones(2), torch.ones(2), torch.ones(2))
