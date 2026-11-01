import numpy as np
import pytest
import torch

from experiments.spectral_reachability.build_cube_d2p_same_root_ranking_pool import candidate_bank
from experiments.spectral_reachability.train_cube_d2p_same_root_ranking_repair import _terminal


def test_positive_packet_and_uniform_candidates_are_physical():
    positive = np.full((5, 5), 0.7, np.float32)
    bank = candidate_bank(positive, candidates=9, generator=np.random.default_rng(4))
    np.testing.assert_array_equal(bank[0], positive)
    assert bank.shape == (9, 5, 5)
    assert np.all(np.abs(bank[1:]) <= 1)


def test_candidate_bank_rejects_nan():
    with pytest.raises(ValueError, match='invalid'):
        candidate_bank(np.full((5, 5), np.nan), candidates=3, generator=np.random.default_rng(4))


class _Stub:
    """No learned model or physics: test only singleton dispatch invariants."""
    def __init__(self, train):
        self.predictor = torch.nn.Identity().train(train)
        self.batch_sizes = []

    def rollout(self, info, actions, history_size):
        self.batch_sizes.append(info['pixels'].shape[0])
        assert info['pixels'].is_contiguous()
        assert actions.is_contiguous()
        return {'predicted_emb': actions[..., :2]}


def test_validation_dispatch_uses_deployment_singleton_shape():
    model = _Stub(False)
    context = torch.zeros((4, 4, 3, 2, 2))
    actions = torch.arange(4 * 3 * 25, dtype=torch.float32).reshape(4, 3, 25)
    terminal = _terminal(model, context, actions)
    torch.testing.assert_close(terminal, actions[:, -1, :2])
    assert model.batch_sizes == [1, 1, 1, 1]


def test_training_preserves_mini_batch_gradient_path():
    model = _Stub(True)
    context = torch.zeros((4, 4, 3, 2, 2))
    actions = torch.ones((4, 3, 25), requires_grad=True)
    _terminal(model, context, actions).sum().backward()
    assert actions.grad is not None
    assert model.batch_sizes == [4]
