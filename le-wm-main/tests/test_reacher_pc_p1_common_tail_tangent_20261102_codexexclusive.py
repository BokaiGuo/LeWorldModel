import pytest
import torch

from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as tangent
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric


def test_analytic_non_immediate_paired_tail_and_budget():
    calls = []
    def evaluate(x):
        calls.append(x.detach().clone())
        return x[0], torch.stack([-x[1], -2*x[1], -3*x[1], -4*x[1], -5*x[1]])
    bank, receipt = tangent.generate(torch.zeros(10, dtype=torch.float64), evaluate)
    assert bank.shape == (7, 10)
    assert len(calls) == receipt["logical_costs"]["evaluation_calls"] == 9
    assert receipt["logical_costs"]["suffix_predict_samples"] == 90
    assert receipt["logical_costs"]["prefix_gradient_vjp_calls"] == 2
    assert receipt["selected_proposal_index"] is not None
    chosen = receipt["proposal_rows"][receipt["selected_proposal_index"]]
    assert chosen["immediate_cost"] >= tangent.MARGIN
    assert all(value <= 0 for value in chosen["per_tail_delta"])
    assert all(value <= 0 for value in chosen["active_tail_neutral_adjusted_delta"])
    assert abs(receipt["gradient_check"]["immediate_tangent_dot"]) < 1e-12
    assert all(r["parameter_delta_l2"] <= .2 + 1e-12 for r in receipt["proposal_rows"])


def test_pareto_common_tail_rule_forbids_average_only_and_no_fallback():
    def evaluate(x):
        return x[0], torch.stack([-3*x[1], x[1]])
    _, receipt = tangent.generate(torch.zeros(10, dtype=torch.float64), evaluate)
    assert receipt["selected_proposal_index"] is None
    assert not any(row["eligible"] for row in receipt["proposal_rows"])


def test_uniform_terminal_gain_without_suffix_specific_gain_rejected():
    def evaluate(x):
        return x[0], torch.stack([-x[1], -x[1]+1, -x[1]+2])
    with pytest.raises(ValueError, match="no first-order"):
        tangent.generate(torch.zeros(10, dtype=torch.float64), evaluate)


def test_no_tangent_rejected_instead_of_fake_winner():
    def evaluate(x):
        return x[0], torch.stack([2*x[0], 3*x[0]])
    with pytest.raises(ValueError, match="no first-order"):
        tangent.generate(torch.zeros(10, dtype=torch.float64), evaluate)


def test_nonfinite_and_changing_tail_count_rejected():
    with pytest.raises(ValueError, match="finite"):
        tangent.common_tails(torch.full((10,), float("nan")))
    count = 0
    def evaluate(x):
        nonlocal count
        count += 1
        values = torch.stack([-x[1], -2*x[1]])
        return x[0], values if count == 1 else values[:1]
    with pytest.raises(ValueError, match="invalid immediate"):
        tangent.generate(torch.zeros(10, dtype=torch.float64), evaluate)


def test_real_q_metric_column_batch_abi():
    q = GoalEncoderMetric(torch.eye(3, dtype=torch.float64))
    predicted = torch.tensor([[1., 2., 3.], [0., 3., 1.]], requires_grad=True)
    target = torch.tensor([0., 0., 0.])
    result = tangent.q_costs(q, predicted, target)
    expected = torch.stack([q.apply(row.double()).square().mean() for row in predicted])
    assert torch.allclose(result, expected)
    result.sum().backward()
    assert torch.isfinite(predicted.grad).all()
    with pytest.raises(ValueError, match="shape"):
        tangent.q_costs(q, predicted, torch.zeros(4))


def test_fixed_tail_packets_and_cuda_boundary():
    tails = tangent.common_tails(torch.zeros(10))
    assert tails.shape == (5, 2, 10)
    assert torch.equal(tails[:, 0], tails[:, 1])
    assert torch.equal(tails[0], torch.zeros(2, 10))
    model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="host CUDA"):
        tangent.compile_reacher(model, torch.zeros(1), torch.zeros(1), torch.zeros(1, 3, 10),
                                None, lower=torch.zeros(10), upper=torch.ones(10))
