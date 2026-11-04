import pytest
import torch

from experiments.spectral_reachability.reacher_pc_spectral_candidates import spectral_candidates


def test_spectral_candidates_use_actual_response_and_symmetric_modes():
    matrix = torch.diag(torch.tensor([3.0, 1.0, 0.2], dtype=torch.float64))
    bank, receipt = spectral_candidates(lambda x: matrix @ x, torch.zeros(3, dtype=torch.float64),
                                        torch.ones(3, dtype=torch.float64),
                                        radius=0.4, modes=1, damping=0.1)
    assert receipt["mode_indices"] == [0, 2]
    assert bank.shape == (6, 3)
    assert torch.allclose(bank[2] + bank[3], torch.zeros(3, dtype=bank.dtype))
    assert torch.allclose(bank[4] + bank[5], torch.zeros(3, dtype=bank.dtype))
    assert torch.linalg.norm(bank[1]) <= 0.40000001
    assert torch.linalg.norm(matrix @ bank[1] - 1) < torch.linalg.norm(torch.ones(3, dtype=bank.dtype))


def test_degenerate_or_bad_contract_fails_closed():
    with pytest.raises(ValueError, match="degenerate"):
        spectral_candidates(lambda x: x * 0, torch.zeros(2), torch.ones(2),
                            radius=1, modes=1, damping=0.1)
    with pytest.raises(ValueError, match="contract"):
        spectral_candidates(lambda x: x, torch.zeros(2), torch.ones(2),
                            radius=0, modes=1, damping=0.1)
