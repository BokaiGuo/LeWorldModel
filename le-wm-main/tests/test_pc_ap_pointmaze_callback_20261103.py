"""CUDA analytic checks only; not a neural or physical planning result."""
import pytest
import torch

from experiments.spectral_reachability.pc_ap_official_cem_kernel import official_cem_kernel
from experiments.spectral_reachability.pc_ap_pointmaze_callback_20261103 import PointMazeAPCallback


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host CUDA required")
def test_ap_callback_preserves_normalized_packet_and_accounts_kernel():
    device = torch.device("cuda:0")

    class FakeModel:
        def __init__(self):
            self.last = None

        def unroll(self, context, act_suffix):
            assert context == "fixed-context"
            self.last = act_suffix.clone()
            return {"proprio": [act_suffix[0, :, :4]]}

    model = FakeModel()
    target = torch.tensor([0.2, -0.1, 0.3, -0.2], device=device)
    callback = PointMazeAPCallback(model, "fixed-context", target, device)
    assert PointMazeAPCallback(model, "fixed-context", target, "cuda").device == target.device
    packets = torch.arange(30, device=device, dtype=torch.float32).reshape(1, 3, 1, 10) / 10
    actual = callback(packets)
    torch.testing.assert_close(model.last, packets[:, :, 0, :])
    torch.testing.assert_close(actual[0], (packets[0, :, 0, :4] - target).square().mean(1))
    assert (callback.batches, callback.samples) == (1, 3)
    result = official_cem_kernel(
        callback, torch.zeros((1, 1, 10), device=device),
        torch.full((10,), -1., device=device), torch.ones(10, device=device),
        torch.Generator(device=device).manual_seed(11), population=8, iterations=2, elites=2,
    )
    assert (result.prediction_batches, result.predicted_blocks) == (3, 18)
    assert (callback.batches, callback.samples) == (4, 21)
    assert result.normalized_block.shape == (1, 1, 10)
    with pytest.raises(ValueError):
        callback(torch.zeros((1, 2, 10), device=device))


def test_callback_rejects_cpu_neural_path():
    with pytest.raises(ValueError, match="host CUDA"):
        PointMazeAPCallback(None, None, torch.zeros(4), "cpu")
