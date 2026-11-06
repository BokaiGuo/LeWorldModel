import torch

from experiments.pc_jepa_multitask_unroll_parity_20261106_codex import decision_parity


def test_near_tie_elite_swap_is_not_hidden_by_tolerance():
    reference = torch.tensor([1.0, 2.0, 2.00001, 4.0])
    split = torch.tensor([1.0, 2.00001, 2.0, 4.0])
    result = decision_parity(reference, split, 2)
    assert result["within_tolerance"]
    assert not result["full_stable_order_equal"]
    assert not result["elite_set_equal"]
    assert result["reference_elite_boundary_gap"] > 0


def test_non_elite_swap_does_not_claim_elite_swap():
    result = decision_parity(torch.tensor([0., 1., 2., 3.]),
                             torch.tensor([0., 1., 3., 2.]), 2)
    assert result["elite_set_equal"]
    assert not result["full_stable_order_equal"]
    assert not result["within_tolerance"]
