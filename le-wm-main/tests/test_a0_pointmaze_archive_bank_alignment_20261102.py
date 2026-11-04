import numpy as np
import torch

from experiments import a0_pointmaze_archive_bank_alignment_20261102 as audit


def test_chunking_preserves_first_major_order(monkeypatch):
    def fake_cost(model, context, suffix, goal):
        return suffix[0, :, 0] * 10 + suffix[1, :, 0]

    monkeypatch.setattr(audit, "terminal_cost", fake_cost)
    a = np.arange(8, dtype=np.float32)
    pairs = np.stack([np.repeat(a, 8), np.tile(a, 8)])
    packets = torch.from_numpy(pairs[:, :, None])
    actual = audit.score(None, None, None, packets, 7).reshape(8, 8)
    expected = np.arange(8)[:, None] * 10 + np.arange(8)[None, :]
    np.testing.assert_array_equal(actual, expected)


def test_action_normalization_then_five_step_packet():
    bank = np.array([[0.25, -0.25], [-0.1, 0.2]], dtype=np.float32)
    mean = np.array([0.1, -0.2], dtype=np.float32)
    std = np.array([0.2, 0.3], dtype=np.float32)
    normalized = (bank - mean) / std
    packets = np.repeat(normalized[:, None, :], 5, axis=1).reshape(2, 10)
    np.testing.assert_allclose(packets.reshape(2, 5, 2) * std + mean,
                               np.repeat(bank[:, None, :], 5, axis=1), atol=1e-7)
