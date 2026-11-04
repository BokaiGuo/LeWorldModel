"""Analytic CPU contracts only: no checkpoint, neural inference or simulator."""
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.a0_pointmaze_cem_feedback_same_action_20261102 import suffix
from experiments.a1_pointmaze_native_spectral_20261102 import packet
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history


def test_pc0_pc1_packet_normalization_matches_multibatch_time_order():
    # Different coordinates, batches and times expose axis swaps and repeated first packet.
    raw = torch.tensor([[[.2, -.3], [.8, .4]], [[-.7, .9], [.1, -.2]]], dtype=torch.float64)
    mean = torch.tensor([.13, -.27], dtype=torch.float64)
    std = torch.tensor([.4, 1.7], dtype=torch.float64)
    packed = suffix(raw, mean, std)
    assert packed.shape == (2, 2, 10)
    for b in range(2):
        for t in range(2):
            torch.testing.assert_close(packed[t, b], packet(raw[b, t], mean, std)[0, 0])
            reconstructed = packed[t, b].reshape(5, 2) * std + mean
            torch.testing.assert_close(reconstructed, raw[b, t].expand(5, 2))


def test_pc1_jvp_and_centered_fd_use_raw_action_and_normalizer_chain_rule():
    # A known analytic map of all ten action slots; no neural model is evaluated.
    mean = torch.tensor([.13, -.27], dtype=torch.float64)
    std = torch.tensor([.4, 1.7], dtype=torch.float64)
    weights = torch.arange(1, 11, dtype=torch.float64)

    def score(u):
        token = packet(u, mean, std).reshape(10)
        return torch.stack(((weights * token).sum(), (token * token).sum()))

    origin = torch.tensor([.21, -.35], dtype=torch.float64)
    basis = torch.eye(2, dtype=torch.float64)
    jac = torch.stack([torch.autograd.functional.jvp(score, origin, d)[1] for d in basis], dim=1)
    eps = 1e-5
    fd = torch.stack([(score(origin + eps*d) - score(origin - eps*d)) / (2*eps)
                      for d in basis], dim=1)
    normalized = (origin - mean) / std
    expected = torch.stack((torch.tensor([25., 30.], dtype=torch.float64) / std,
                            10 * normalized / std))
    torch.testing.assert_close(jac, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(jac, fd, rtol=1e-9, atol=1e-9)
    # A test that omits the normalization chain rule would not pass this case.
    assert not torch.allclose(jac[0], torch.tensor([25., 30.], dtype=torch.float64))


class AnalyticPredictor:
    proprio_mode = "predict_proprio"

    def __init__(self):
        self.model = self
        self.seen = []

    def encode_act(self, action):
        self.seen.append(action.detach().clone())
        return action

    def forward_pred(self, visual, action, proprio):
        # Last observation plus its outgoing control; the observed action enters
        # a separate historical slot so accidental time shifts are observable.
        v = action[..., :1, None, None, None]
        return visual + v, None, proprio + action[..., :1, None]


def test_pc0_action_history_time_index_and_ad_fd_agree():
    model = AnalyticPredictor()
    context = {"visual": torch.zeros(1, 3, 1, 1, 1, 1, dtype=torch.float64),
               "proprio": torch.zeros(1, 3, 1, 1, dtype=torch.float64)}
    history = torch.tensor([[[2.]], [[3.]]], dtype=torch.float64)

    def terminal(values):
        return unroll_with_action_history(model, context, history, values.reshape(3, 1, 1), 3)["visual"][-1].sum()

    controls = torch.tensor([5., 7., 11.], dtype=torch.float64, requires_grad=True)
    actual = terminal(controls)
    assert actual.item() == 23.
    assert [v.flatten().tolist() for v in model.seen] == [[2., 3., 5.], [3., 5., 7.], [5., 7., 11.]]
    grad = torch.autograd.grad(actual, controls)[0]
    eps = 1e-5
    fd = torch.stack([(terminal(controls.detach() + eps * d) -
                       terminal(controls.detach() - eps * d)) / (2 * eps)
                      for d in torch.eye(3, dtype=torch.float64)])
    torch.testing.assert_close(grad, torch.ones_like(grad))
    torch.testing.assert_close(grad, fd, rtol=1e-9, atol=1e-9)


def test_training_slicer_binds_outgoing_five_actions_to_observation(monkeypatch):
    vendor = Path(__file__).resolve().parents[1] / "external/jepa-wms"
    monkeypatch.syspath_prepend(str(vendor))
    from app.plan_common.datasets.traj_dset import TrajSlicerDataset

    class FakeTrajectory:
        action_dim = 2
        proprio_dim = 1
        state_dim = 1

        def __len__(self):
            return 1

        def get_seq_length(self, idx):
            return 40

        def __getitem__(self, idx):
            frame = torch.arange(40, dtype=torch.float32)
            act = torch.stack((frame, -frame), dim=1)
            return {"visual": frame[:, None]}, act, frame[:, None], None, {}

    dataset = TrajSlicerDataset(FakeTrajectory(), num_frames=3, frameskip=5, action_skip=1)
    dataset.slices = [(0, 7, 22)]
    observation, actions, state, reward = dataset[0]
    torch.testing.assert_close(observation["visual"].flatten(), torch.tensor([7., 12., 17.]))
    for t in range(3):
        expected_time = torch.arange(7 + 5*t, 12 + 5*t, dtype=torch.float32)
        torch.testing.assert_close(actions[t].reshape(5, 2),
                                   torch.stack((expected_time, -expected_time), dim=1))
