import torch

from experiments.spectral_reachability.reacher_d2p_suffix_step_rescue import accept_with_fine_rescue


def test_fine_rescue_finds_descent_hidden_by_one_sixteenth_grid():
    # Optimum scale .01: all legacy nonzero scales overshoot it.
    terminal = lambda value: 100 * value
    delta, accepted, receipt = accept_with_fine_rescue(terminal, torch.tensor([1.]), torch.tensor([0.]),
        torch.tensor([1.]), step_rule="backtracking")
    assert receipt["coarse_stop"]
    assert receipt["fine_rescue_accepted"]
    assert 0. < receipt["accepted_step_scale"] < 1 / 16
    assert float((accepted - 1).square()) < 1.
    assert not delta.requires_grad


def test_no_extra_work_when_legacy_accepts():
    calls = []
    def terminal(value):
        calls.append(1)
        return value
    _, _, receipt = accept_with_fine_rescue(terminal, torch.tensor([1.]), torch.tensor([0.]),
        torch.tensor([1.]), step_rule="backtracking")
    assert len(calls) == 5
    assert not receipt["fine_rescue_attempted"]
    assert receipt["accepted_step_scale"] == 1.


def test_real_no_descent_preserves_exact_zero():
    delta, accepted, receipt = accept_with_fine_rescue(lambda value: value.square(), torch.tensor([1.]),
        torch.tensor([0.]), torch.tensor([0.]), step_rule="backtracking")
    assert receipt["fine_rescue_attempted"]
    assert not receipt["fine_rescue_accepted"]
    assert torch.equal(delta, torch.zeros(1))
    assert torch.equal(accepted, torch.zeros(1))
