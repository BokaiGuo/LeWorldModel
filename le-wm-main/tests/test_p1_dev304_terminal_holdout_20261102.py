import torch

from experiments.spectral_reachability import p1_dev304_terminal_holdout_20261102 as subject
from experiments.spectral_reachability import p1_dev304_terminal_saved_20261102 as saved


def test_terminal_objective_distinguishes_different_second_packet_states():
    goal = torch.zeros((1, 2))
    final = torch.tensor([[1., 1.], [2., 2.]])
    assert torch.equal(subject.terminal_cost(final, goal), torch.tensor([1., 4.]))


def test_revision_is_not_independent_confirmation():
    spec = subject.specification()
    assert spec["selection_objective"] == "two-packet terminal latent squared error"
    assert spec["classification"].startswith("post_outcome_development")
    assert not spec["p1_admission"] and not spec["independent_confirmation"]


def test_saved_physical_execution():
    result = saved.analyze()
    assert result["frames_verified"] == 20
    assert result["same_selected_actions"]
    assert all(row["status"] == "fixed_horizon_completed" and not row["success"]
               for row in result["rows"])
    assert not result["independent_confirmation"]
