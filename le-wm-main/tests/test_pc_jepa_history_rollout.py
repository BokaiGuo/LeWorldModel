import pytest
import torch

from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history


class TracedModel:
    def __init__(self):
        self.calls = []

    def encode_act(self, actions):
        return actions

    def forward_pred(self, visual, actions, proprio):
        self.calls.append((visual.clone(), actions.clone(), proprio.clone()))
        return visual + actions, None, proprio + actions


def test_three_step_windows_and_batch_causality():
    model = TracedModel()
    visual = torch.tensor([[[0.], [10.], [20.]]])
    proprio = visual + 100
    history = torch.tensor([[[1.], [2.]]])
    suffix = torch.tensor([[[3.], [30.]], [[4.], [40.]], [[5.], [50.]]])
    saved = [x.clone() for x in (visual, proprio, history, suffix)]
    result = unroll_history(model, visual, proprio, history, suffix, 3)
    assert [c[1][0, :, 0].tolist() for c in model.calls] == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert [c[1][1, :, 0].tolist() for c in model.calls] == [[1, 2, 30], [2, 30, 40], [30, 40, 50]]
    assert result["visual"][:, 0, 0].tolist() == [0, 10, 20, 23, 27, 32]
    assert result["visual"][:, 1, 0].tolist() == [0, 10, 20, 50, 90, 140]
    assert all(torch.equal(a, b) for a, b in zip((visual, proprio, history, suffix), saved))


def test_context_window_one_and_no_history():
    model = TracedModel()
    result = unroll_history(model, torch.ones(1, 1, 1), torch.ones(1, 1, 1),
                            torch.empty(1, 0, 1), torch.tensor([[[2.]], [[3.]]]), 1)
    assert [c[1][0, :, 0].tolist() for c in model.calls] == [[2], [3]]
    assert result["visual"][:, 0, 0].tolist() == [1, 3, 6]


@pytest.mark.parametrize("bad_history", [torch.zeros(1, 1, 1), torch.full((1, 2, 1), float("nan"))])
def test_invalid_history_fails_before_prediction(bad_history):
    model = TracedModel()
    with pytest.raises(ValueError):
        unroll_history(model, torch.zeros(1, 3, 1), torch.zeros(1, 3, 1),
                       bad_history, torch.zeros(1, 1, 1), 3)
    assert not model.calls
