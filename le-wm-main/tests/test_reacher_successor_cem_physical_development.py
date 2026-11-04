import pytest
import torch

from experiments.spectral_reachability.run_reacher_successor_cem_physical_development import bound_indices


def test_bound_indices_uses_sealed_selection_and_same_bank_greedy():
    bank = torch.tensor([[[0.0]], [[2.0]], [[3.0]]])
    row = {"selected_params": [[[3.0]]], "selection": {
        "selected_proposal_index": 2, "eligible_count": 1,
        "proposal_rows": [{"proposal_index": i, "immediate_goal_cost": c}
                          for i, c in enumerate([2.0, 0.5, 1.0])]}}
    assert bound_indices(row, bank) == {"selected": 2, "zero": 0, "same_bank_greedy": 1}
    row["selected_params"] = [[[2.0]]]
    with pytest.raises(ValueError, match="reconstructed bank"):
        bound_indices(row, bank)


def test_ineligible_root_retained_as_zero_fallback():
    bank = torch.tensor([[[0.0]], [[1.0]]])
    row = {"selected_params": None, "selection": {
        "selected_proposal_index": None, "eligible_count": 0,
        "proposal_rows": [{"proposal_index": 0, "immediate_goal_cost": 1.0},
                          {"proposal_index": 1, "immediate_goal_cost": 0.0}]}}
    assert bound_indices(row, bank) == {"selected": 0, "zero": 0, "same_bank_greedy": 1}
    row["selection"]["eligible_count"] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        bound_indices(row, bank)
