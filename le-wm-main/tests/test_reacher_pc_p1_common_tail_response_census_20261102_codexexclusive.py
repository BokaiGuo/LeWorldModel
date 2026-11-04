import pytest
import torch

from experiments.spectral_reachability import reacher_pc_p1_common_tail_response_census_20261102_codexexclusive as census
from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as generator
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric


def q():
    return GoalEncoderMetric(torch.zeros((2, 2), dtype=torch.float64))


def endpoints(gain):
    return torch.tensor([[0., 0.], [gain, 0.], [-gain, 0.], [0., gain], [0., -gain]],
                        dtype=torch.float64)


def test_known_gain_and_symmetric_spectra():
    result = census.census(q(), endpoints(1.), endpoints(.5))
    assert result["response_frobenius_ratio"] == pytest.approx(.5)
    assert result["singular_value_ratios"] == pytest.approx([.5, .5])
    assert result["changes_existing_selector"] is False


def test_real_counterexample_selector_can_reward_weaker_response():
    # All scores and the response census come from the same Q-scored endpoints.
    metric = GoalEncoderMetric(torch.zeros((3, 3), dtype=torch.float64))
    def predicted_endpoints(x):
        gain = 1 + x[1]
        z = torch.sqrt(2 + .1*x[1])
        zero = torch.zeros_like(gain)
        return torch.stack([torch.stack([zero, zero, z]), torch.stack([gain, zero, z]),
                            torch.stack([-gain, zero, z]), torch.stack([zero, gain, z]),
                            torch.stack([zero, -gain, z])])
    def evaluate(x):
        return 1+x[0], generator.q_costs(metric, predicted_endpoints(x), torch.zeros(3))
    bank, result = generator.generate(torch.zeros(10, dtype=torch.float64), evaluate)
    chosen = result["selected_proposal_index"]
    assert chosen is not None
    result = census.census(metric, predicted_endpoints(bank[0]), predicted_endpoints(bank[chosen]))
    assert result["response_frobenius_ratio"] < 1
    assert all(v < 1 for v in result["singular_value_ratios"])


def test_zero_reference_and_shape_fail_closed():
    result = census.census(q(), endpoints(0.), endpoints(1.))
    assert result["response_frobenius_ratio"] is None
    assert result["singular_value_ratios"] == [None, None]
    with pytest.raises(ValueError, match="aligned"):
        census.census(q(), endpoints(1.)[:4], endpoints(1.))
    broken = endpoints(1.)
    broken[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        census.census(q(), broken, endpoints(1.))
