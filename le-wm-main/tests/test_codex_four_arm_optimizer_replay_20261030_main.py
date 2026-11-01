import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_optimizer_replay_20261030_main as mod


class AnalyticFixture:
    def action_encoder(self, context):
        return context

    def predict(self, states, actions):
        result = states.clone()
        result[:, -1, 0] = states[:, -1, 0] + actions[:, -1].sum(dim=1)
        return result


def test_replay_normalizes_each_packet_and_advances_history():
    task = {"root_context": {"latent_history": [[0.] * 192] * 3,
                             "action_history": [[0.] * 10] * 2},
            "goal": [0.] * 192, "action_mean": [1., 1.], "action_std": [2., 2.]}
    actions = np.zeros((3, 5, 2), dtype=np.float32)
    assert mod.replay(AnalyticFixture(), torch.device("cpu"), task, actions) == pytest.approx(225 / 192)


def test_replay_rejects_zero_normalizer():
    task = {"root_context": {"latent_history": [[0.] * 192] * 3,
                             "action_history": [[0.] * 10] * 2},
            "goal": [0.] * 192, "action_mean": [0., 0.], "action_std": [0., 1.]}
    with pytest.raises(ValueError, match="normalizer"):
        mod.replay(AnalyticFixture(), torch.device("cpu"), task, np.zeros((3, 5, 2)))
