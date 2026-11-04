import numpy as np
import pytest
import torch

from experiments.spectral_reachability.run_reacher_pc_guarded_prediction_replay import direct_rollout, validate_actions


class Dummy:
    def action_encoder(self, actions):
        return actions

    def predict(self, embeddings, actions):
        return embeddings + actions


def test_direct_rollout_uses_each_new_action_at_its_boundary():
    embeddings = torch.tensor([[[0.], [1.], [2.]]])
    history = torch.tensor([[[9.], [8.], [7.]]])
    actions = torch.tensor([[[3.], [4.], [5.]]])
    assert torch.equal(direct_rollout(Dummy(), embeddings, history, actions),
                       torch.tensor([[[5.], [9.], [14.]]]))


def test_saved_raw_and_normalized_binding():
    normalized = np.arange(30, dtype=np.float32).reshape(3, 10)
    raw = normalized.reshape(15, 2)
    method = {"action_witness_raw": raw.tolist(), "action_witness_normalized": normalized.tolist()}
    validate_actions(method, np.zeros(10), np.ones(10))
    method["action_witness_raw"][0][0] = 3
    with pytest.raises(ValueError, match="differ"):
        validate_actions(method, np.zeros(10), np.ones(10))
