import numpy as np
import pytest
import torch

from experiments.spectral_reachability.d2p_cuda_sampling_cost import CudaSamplingCost, _require_frozen_eval, _require_real_cost_dtype
from types import SimpleNamespace


class MetadataModel:
    def __init__(self):
        self.child = SimpleNamespace(training=False)
        self.parameter = SimpleNamespace(requires_grad=False, device="cuda:0")

    def modules(self):
        return [SimpleNamespace(training=False), self.child]

    def parameters(self):
        return [self.parameter]


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128, torch.bool, torch.int64])
def test_nonreal_or_nonfloating_cost_dtype_rejected_before_conversion(dtype):
    # Tensor metadata validation only: no model or neural computation.
    with pytest.raises(ValueError, match="real floating-point"):
        _require_real_cost_dtype(torch.zeros(2, dtype=dtype))


def test_real_float_cost_dtype_accepted_as_metadata():
    _require_real_cost_dtype(torch.zeros(2, dtype=torch.float32))


def test_partial_training_and_parameter_drift_rejected_without_inference():
    model = MetadataModel()
    _require_frozen_eval(model, "cuda:0")
    model.child.training = True
    with pytest.raises(ValueError, match="every model module"):
        _require_frozen_eval(model, "cuda:0")
    model.child.training = False
    model.parameter.requires_grad = True
    with pytest.raises(ValueError, match="frozen parameters"):
        _require_frozen_eval(model, "cuda:0")
    model.parameter.requires_grad = False
    model.parameter.device = "cpu"
    with pytest.raises(ValueError, match="CUDA device"):
        _require_frozen_eval(model, "cuda:0")


def test_cpu_rejected_before_cost_callable():
    def forbidden(*args, **kwargs):
        raise AssertionError("not called")
    with pytest.raises(ValueError, match="CUDA"):
        CudaSamplingCost(None, torch.zeros(1), torch.zeros(1), torch.zeros(1),
                         action_lower=torch.zeros(1), action_upper=torch.ones(1),
                         future_horizon=1, cost_fn=forbidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host CUDA only")
def test_cuda_candidate_bridge_counts_batches_and_rejects_bad_shape():
    model = torch.nn.Identity().cuda().eval().requires_grad_(False)
    context = torch.zeros(1, 3, 2, device="cuda")
    def cost(model, pixels, goal, history, params, **kwargs):
        assert params.device.type == "cuda" and not torch.is_grad_enabled()
        return params.square().sum((1, 2))
    oracle = CudaSamplingCost(model, context, context, context,
        action_lower=-torch.ones(2, device="cuda"), action_upper=torch.ones(2, device="cuda"),
        future_horizon=2, cost_fn=cost)
    np.testing.assert_array_equal(oracle(np.ones((3, 2, 2))), [4., 4., 4.])
    with pytest.raises(ValueError):
        oracle(np.ones((3, 1, 2)))
    assert oracle.receipt()["candidate_cost_samples"] == 3
    assert oracle.receipt()["candidate_cost_calls"] == 1
