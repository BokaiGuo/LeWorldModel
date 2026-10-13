import math

import numpy as np
import torch

from experiments.empirical_spectral_audit.run_real_latent_alignment_adapter import (
    AdapterConfig,
    ResidualAdapter,
    _split_indices,
    _tensor_dataset,
    _run_one,
)


def _arrays(n=40, dim=6, action_dim=4):
    rng = np.random.default_rng(0)
    source = rng.normal(size=(n, dim)).astype("float32")
    base = rng.normal(scale=0.1, size=(n, dim)).astype("float32")
    target = (base + 0.05 * source).astype("float32")
    actions = rng.normal(size=(n, action_dim)).astype("float32")
    return {"latents": source, "predicted_dynamics": base, "dynamics": target, "actions": actions}


def test_residual_adapter_is_identity_correction_at_initialization():
    model = ResidualAdapter(6, 4)
    source = torch.randn(3, 6)
    delta = torch.randn(3, 6)
    action = torch.randn(3, 4)
    assert torch.equal(model(source, delta, action), source + delta)


def test_real_adapter_smoke_emits_finite_held_out_metrics():
    arrays = _arrays()
    train, val, test = _split_indices(40, 0.1, 0.1, seed=1)
    result = _run_one(
        arrays,
        train,
        val,
        test,
        "conditional",
        0.01,
        0,
        AdapterConfig(epochs=1, batch_size=8, alignment_min_samples=4),
        torch.device("cpu"),
    )
    assert math.isfinite(result["test"]["mse"])
    assert math.isfinite(result["test"]["conditional_alignment"]["operator_alignment"])
    assert result["num_train"] + result["num_val"] + result["num_test"] == 40
