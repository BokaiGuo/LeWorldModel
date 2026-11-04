"""CPU analytic fixture checks the extracted official kernel, not a model run."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest
import torch

from experiments.spectral_reachability.pc_ap_official_cem_kernel import official_cem_kernel


UPSTREAM = (Path(__file__).resolve().parents[2] / "external" /
            "anchored-planning-984327424ca7" / "code" / "ap" /
            "anchored_planning" / "gaussian_cem.py")


class IdentityScaler:
    def inverse_transform(self, x):
        return x

    def transform(self, x):
        return x


class AnalyticNative:
    device = torch.device("cpu")
    action_dim = 2
    scaler = IdentityScaler()

    def __init__(self):
        self.latent_cost = self
        self.seen = []

    def pixels(self, _):
        return torch.zeros(1, 1, 3, 2, 2)

    def get_cost(self, info, actions):
        self.seen.append(actions.clone())
        return (actions - 0.17).square().sum(dim=(-1, -2))


def test_same_seed_and_inputs_match_pinned_official_kernel():
    assert UPSTREAM.is_file(), "pinned official AP source is required for parity test"
    spec = importlib.util.spec_from_file_location("ap_upstream_cem", UPSTREAM)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    native = AnalyticNative()
    generator = torch.Generator().manual_seed(101)
    raw, receipt = upstream.cem(
        native, None, torch.zeros(1, 2), torch.zeros(1, 2), generator,
        np.zeros((5, 2), np.float32),
        (np.full(2, -0.6, np.float32), np.full(2, 0.6, np.float32)),
        (np.full(2, -0.6, np.float32), np.full(2, 0.6, np.float32)),
    )

    seen = []

    def evaluate(actions):
        seen.append(actions.clone())
        return (actions - 0.17).square().sum(dim=(-1, -2))

    result = official_cem_kernel(
        evaluate, torch.zeros(1, 1, 10), torch.full((10,), -0.6),
        torch.full((10,), 0.6), torch.Generator().manual_seed(101),
    )
    assert len(native.seen) == len(seen) == 31
    assert all(torch.equal(a, b) for a, b in zip(native.seen, seen))
    np.testing.assert_array_equal(result.normalized_block.numpy().reshape(5, 2), raw)
    assert result.predicted_blocks == receipt["predictor_candidate_transitions"] == 9002
    assert result.initial_cost == receipt["prior_cost"]
    assert result.refined_cost == receipt["refined_cost"]
    assert result.refined_selected == receipt["refined_mean_selected"]


def test_rejects_prior_outside_action_domain_and_single_elite():
    prior = torch.zeros(1, 1, 2)
    bounds = torch.ones(2)
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="sample standard deviation"):
        official_cem_kernel(lambda x: x.sum((-1, -2)), prior, -bounds, bounds,
                            generator, population=2, elites=1)
    with pytest.raises(ValueError, match="invalid bounds or prior"):
        official_cem_kernel(lambda x: x.sum((-1, -2)), prior + 2, -bounds, bounds,
                            generator)
