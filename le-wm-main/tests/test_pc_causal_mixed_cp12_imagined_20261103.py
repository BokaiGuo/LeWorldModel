import pytest
import torch
from tensordict import TensorDict

from experiments.pc_causal_mixed_cp12_imagined_20261103 import last_context


def test_last_context_handles_unbatched_prediction_container():
    pred = TensorDict({"visual": torch.arange(12.).reshape(2, 1, 2, 3),
                       "proprio": torch.arange(8.).reshape(2, 1, 2, 2)}, batch_size=[])
    context = last_context(pred, torch.device("cpu"))
    assert context.batch_size == torch.Size([1, 1])
    torch.testing.assert_close(context["visual"], pred["visual"][-1:])
    torch.testing.assert_close(context["proprio"], pred["proprio"][-1:])
    with pytest.raises(ValueError):
        last_context({"visual": torch.zeros(1, 1, 2), "proprio": torch.zeros(1, 1, 2)},
                     torch.device("cpu"))
