import pytest
import torch

from experiments.spectral_reachability.run_reacher_pc_feasible_prefix_physical import physical_indices


def test_all_appended_candidates_are_included_without_outcome_ranking():
    bank = torch.arange(50, dtype=torch.float32).reshape(5, 1, 10)
    row = {"original_bank_size": 3,
           "selected_params": bank[1:2].tolist(),
           "selection": {"proposal_rows": [{"proposal_index": i, "immediate_goal_cost": float(i)}
                                            for i in range(5)],
                         "selected_proposal_index": 1, "eligible_count": 1},
           "refinement": {"backtracking": [{"appended": True}, {"appended": False},
                                            {"appended": True}, {"appended": False}]}}
    assert physical_indices(row, bank) == {
        "selected": 1, "zero": 0, "same_bank_greedy": 0, "refined_0": 3, "refined_1": 4}
    row["refinement"]["backtracking"][0]["appended"] = False
    with pytest.raises(ValueError):
        physical_indices(row, bank)
