import pytest
import torch

from experiments.spectral_reachability.pc_spectral_seeded_feedback_cem_20261103_codex import (
    seeded_cem, spectral_candidates,
)
from experiments.pc_pointmaze_cp12_rgb_history_planning_20261103_codex import cem


def test_normalizer_chain_rule_and_spectral_candidates():
    dtype = torch.float64
    mean = torch.tensor([.25, -.4], dtype=dtype)
    std = torch.tensor([.2, 2.], dtype=dtype)
    matrix = torch.arange(1, 21, dtype=dtype).reshape(2, 10) / 10
    response = lambda normalized: matrix @ (normalized * std.repeat(5) + mean.repeat(5))
    result = spectral_candidates(response, mean, std)
    torch.testing.assert_close(result["jacobian_raw"], matrix)
    raw = result["raw"]
    torch.testing.assert_close(raw.abs().amax(dim=1), torch.tensor([.25, .25, .5, .5], dtype=dtype))
    torch.testing.assert_close(result["normalized"] * std.repeat(5) + mean.repeat(5), raw)
    eps = 1e-6
    for col in range(10):
        d = torch.zeros(10, dtype=dtype)
        d[col] = eps
        z0 = -mean.repeat(5) / std.repeat(5)
        finite = (response(z0 + d / std.repeat(5)) - response(z0 - d / std.repeat(5))) / (2 * eps)
        torch.testing.assert_close(finite, result["jacobian_raw"][:, col], atol=1e-9, rtol=1e-9)
    assert result["structure_counts"] == {
        "response_forward_calls": 1, "reverse_vector_calls": 2, "svd_calls": 1
    }


def test_degenerate_and_detached_responses_fail_closed():
    mean, std = torch.zeros(2), torch.ones(2)
    with pytest.raises(ValueError, match="degenerate"):
        spectral_candidates(lambda x: torch.stack((x.sum() * 0, x.sum() * 0)), mean, std)
    with pytest.raises(ValueError, match="differentiable"):
        spectral_candidates(lambda x: torch.zeros(2), mean, std)
    with pytest.raises(ValueError, match="normalization"):
        spectral_candidates(lambda x: x[:2], mean, torch.tensor([1., 0.]))


def test_no_injection_matches_historical_cem_and_injection_is_bound():
    generator = torch.Generator().manual_seed(734)
    noise = torch.randn((3, 16, 10), generator=generator)
    prior, lower, upper = torch.zeros(10), torch.full((10,), -1.), torch.ones(10)
    score = lambda x: (x - .2).square().sum(dim=1)
    old = cem(score, prior, lower, upper, noise)
    new = seeded_cem(score, prior, lower, upper, noise)
    for old_value, new_value in zip((old[0], old[2], old[3]), (new["chosen"], new["final"], new["final_scores"])):
        torch.testing.assert_close(old_value, new_value, rtol=0, atol=0)
    assert old[4] == new["winner"]
    for old_iteration, new_iteration in zip(old[1], new["trace"]):
        for a, b in zip(old_iteration, new_iteration):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    injected = torch.full((4, 10), .9)
    seeded = seeded_cem(score, prior, lower, upper, noise, injected)
    torch.testing.assert_close(seeded["trace"][0][0][1:5], injected, rtol=0, atol=0)
    assert seeded["cem_counts"] == {"samples": 50, "batches": 4}
    with pytest.raises(ValueError, match="injected"):
        seeded_cem(score, prior, lower, upper, noise, torch.full((4, 10), 2.))
