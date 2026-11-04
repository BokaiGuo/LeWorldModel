import numpy as np
import pytest
import torch

from experiments.spectral_reachability.pc_ap_reacher_cuda_adapter import ReacherCudaPredictor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; no CPU fallback")
def test_actual_history_order_and_attempted_accounting():
    class Stub(torch.nn.Module):
        def action_encoder(self, actions):
            return actions
        def predict(self, embeddings, actions):
            self.actions = actions
            return embeddings + actions[..., :2]

    model = Stub().cuda().eval()
    embeddings = torch.zeros((1, 3, 2), device="cuda")
    history = torch.arange(30., device="cuda").reshape(1, 3, 10)
    adapter = ReacherCudaPredictor(model, embeddings, history)
    blocks = np.ones((4, 5, 2), dtype=np.float32)
    result = adapter(blocks)
    np.testing.assert_array_equal(result, np.ones((4, 2)))
    torch.testing.assert_close(model.actions[0, :2], history[0, 1:])
    torch.testing.assert_close(model.actions[0, 2], torch.ones(10, device="cuda"))
    assert (adapter.predicted_blocks, adapter.prediction_batches) == (4, 1)
    with pytest.raises(ValueError):
        adapter(np.zeros((4, 1, 10)))
    assert adapter.predicted_blocks == 4
