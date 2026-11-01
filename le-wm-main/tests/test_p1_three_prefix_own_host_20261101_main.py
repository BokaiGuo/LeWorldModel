import pytest
import torch
from experiments.spectral_reachability import p1_three_prefix_own_host_20261101_main as m


def test_prefix_roles_are_explicit_and_greedy_is_stage_based():
    bank = torch.arange(8, dtype=torch.float32)[:, None, None, None].expand(8, 3, 5, 2) / 8
    result = dict(optimized_raw=bank, selected_raw=bank[7])
    scores = [3., 2., 1., 1., 4., 5., 6., 7.]
    assert torch.equal(m.choose_prefix(result, scores, "greedy"), bank[2])
    assert torch.equal(m.choose_prefix(result, scores, "selected"), bank[7])
    assert torch.count_nonzero(m.choose_prefix(result, scores, "zero")) == 0
    with pytest.raises(ValueError):
        m.choose_prefix(result, scores, "unknown")
