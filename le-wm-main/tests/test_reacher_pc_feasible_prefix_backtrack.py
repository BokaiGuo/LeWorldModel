import pytest
import torch

from experiments.spectral_reachability.reacher_pc_feasible_prefix_backtrack import choose


def test_feasible_interpolation_and_tie_order():
    def evaluate(x):
        return x[0], (x - 0.5).square()
    result, receipt = choose(torch.tensor([1.0]), torch.tensor([0.0]), evaluate, 0.3)
    assert torch.equal(result, torch.tensor([0.5]))
    assert receipt["chosen_alpha"] == 0.5
    assert len(receipt["trials"]) == 5


def test_no_feasible_candidate_and_nonfinite_rejected():
    result, receipt = choose(torch.tensor([0.0]), torch.tensor([1.0]),
                             lambda x: (x[0], x), 2.0)
    assert result is None and receipt["chosen_alpha"] is None
    with pytest.raises(ValueError):
        choose(torch.zeros(1), torch.ones(1), lambda x: (x[0], x * float("nan")), 0.0)
