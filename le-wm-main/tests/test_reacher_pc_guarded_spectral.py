import pytest
import torch

from experiments.spectral_reachability.reacher_pc_guarded_spectral import guarded_candidates


def test_linear_modes_pass_at_largest_scale():
    bank, receipt = guarded_candidates(lambda x: torch.stack((2 * x[0], x[1])),
                                       torch.zeros(2, dtype=torch.float64),
                                       torch.ones(2, dtype=torch.float64), modes=1)
    assert bank.shape == (6, 2)
    assert len(receipt["probes"]) == 4
    assert all(p["accepted"] and p["scale"] == 0.5 for p in receipt["probes"])


def test_curved_direction_backtracks_and_rejection_is_not_zero_candidate():
    def nonlinear(x):
        return torch.stack((x[0] + 10 * x[0] ** 2, x[1] + 10 * x[1] ** 2))
    bank, receipt = guarded_candidates(nonlinear, torch.zeros(2, dtype=torch.float64),
                                       torch.ones(2, dtype=torch.float64), modes=1)
    assert any(p["scale"] != 0.5 and p["accepted"] for p in receipt["probes"])
    assert all(not label.startswith("mode") or torch.count_nonzero(bank[i])
               for i, label in enumerate(receipt["candidate_labels"]))
    assert receipt["nonlinear_guard_primal_rollouts"] == len(receipt["probes"])
    with pytest.raises(ValueError, match="contract"):
        guarded_candidates(nonlinear, torch.zeros(2), torch.ones(2), scales=(0.1, 0.5))
