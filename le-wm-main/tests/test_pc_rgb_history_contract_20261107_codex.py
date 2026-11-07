import pytest
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, freeze_audit
from experiments.pc_rgb_history_contract_20261107_codex import score_future


class Capture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.seen = None
        self.eval()

    def get_cost(self, context, actions):
        self.seen = actions
        return actions.sum(dim=(-1, -2))


@pytest.fixture
def bound(tmp_path):
    path = tmp_path / "binding"
    path.write_text("fixture")
    audit = freeze_audit(RGB, "lewm", [path], preprocessing="fixture", clock="fixture")
    model = Capture()
    return model, RGBLeWM(model, audit)


def test_future_precedes_no_history_and_preserves_gradients(bound):
    model, wrapper = bound
    future = torch.tensor([[[[3.0]], [[5.0]]]], requires_grad=True)
    score, _ = score_future(wrapper, pixels=torch.ones(1, 1, 1, 3, 2, 2),
                            goal=torch.ones(1, 1, 1, 3, 2, 2),
                            history=torch.empty(1, 1, 0, 1), future=future)
    assert torch.equal(model.seen, future)
    score.sum().backward()
    assert torch.equal(future.grad, torch.ones_like(future))


def test_multi_frame_history_cannot_be_replaced_by_candidate(bound):
    model, wrapper = bound
    history = torch.tensor([[[[1.0], [2.0]]]])
    future = torch.tensor([[[[3.0]], [[4.0]]]])
    score_future(wrapper, pixels=torch.ones(1, 1, 3, 3, 2, 2),
                 goal=torch.ones(1, 1, 1, 3, 2, 2), history=history, future=future)
    assert torch.equal(model.seen, torch.tensor([[[[1.0], [2.0], [3.0]],
                                                 [[1.0], [2.0], [4.0]]]]))
    assert torch.equal(history, torch.tensor([[[[1.0], [2.0]]]]))


@pytest.mark.parametrize("history_shape,future_shape", [
    ((1, 1, 1, 1), (1, 2, 1, 1)),
    ((1, 2, 2, 1), (1, 2, 1, 1)),
    ((1, 1, 2, 2), (1, 2, 1, 1)),
    ((1, 1, 2, 1), (1, 2, 0, 1)),
])
def test_rejects_wrong_clock_or_shared_history(bound, history_shape, future_shape):
    _, wrapper = bound
    with pytest.raises(ValueError):
        score_future(wrapper, pixels=torch.ones(1, 1, 3, 3, 2, 2),
                     goal=torch.ones(1, 1, 1, 3, 2, 2),
                     history=torch.zeros(history_shape), future=torch.zeros(future_shape))
