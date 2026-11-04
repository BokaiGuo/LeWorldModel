import pytest
import torch
from copy import deepcopy

from experiments.spectral_reachability import run_reacher_pc_spectral_tail_carryover as m


def test_saved_suffix_preserves_parameters_and_rejects_wrong_index():
    row = {"selection": {"proposal_rows": [
        {"proposal_index": 0, "suffix_parameters": [[0.2, -0.3], [0.4, 0.1]]}]}}
    result = m.saved_suffix(row, 0, horizon=2, action_dim=2, device="cpu")
    torch.testing.assert_close(result, torch.tensor([[[0.2, -0.3], [0.4, 0.1]]]))
    row["selection"]["proposal_rows"][0]["proposal_index"] = 1
    with pytest.raises(ValueError, match="index"):
        m.saved_suffix(row, 0, horizon=2, action_dim=2, device="cpu")


def test_legacy_fallback_and_invalid_witness():
    row = {"selection": {"proposal_rows": [{"proposal_index": 0}]}}
    assert m.saved_suffix(row, 0, horizon=2, action_dim=2, device="cpu") is None
    row["selection"]["proposal_rows"][0]["suffix_parameters"] = [[float("nan"), 0.0]]
    with pytest.raises(ValueError, match="invalid"):
        m.saved_suffix(row, 0, horizon=2, action_dim=2, device="cpu")


def test_saved_witness_scored_from_successor_context(monkeypatch):
    seen = []
    def terminal(model, embeddings, actions, suffix, **kwargs):
        seen.append((embeddings.clone(), actions.clone(), suffix.clone()))
        return torch.tensor([[2.0, 3.0]])
    monkeypatch.setattr(m.selector.p1, "_terminal_from_state", terminal)
    monkeypatch.setattr(m.setup.p0, "_planning_coordinate", lambda model, value, role: value)
    embeddings, actions, suffix = torch.ones(1, 2, 2), torch.zeros(1, 2, 2), torch.ones(1, 2, 2)
    cost = m.score_saved_suffix(None, torch.tensor([[1.0, 1.0]]), embeddings, actions,
                                suffix, lower=-torch.ones(2), upper=torch.ones(2))
    assert cost == pytest.approx(2.5)
    assert len(seen) == 1
    assert all(torch.equal(a, b) for a, b in zip(seen[0], (embeddings, actions, suffix)))


def test_cross_report_binding_rejects_changed_proposal():
    row = {"seed": 1, "root_rgb_sha256": "root", "goal_rgb_sha256": "goal",
           "bank_params": [[0.0]], "selected_params": [[0.0]],
           "selection": {"selected_proposal_index": 0, "proposal_rows": [
               {"proposal_index": 0, "immediate_goal_cost": 1.0, "cem_future_goal_cost": 2.0}]}}
    source = {key: "same" for key in ("source_report_sha256", "weights_sha256",
              "action_stats_sha256", "parameter_hash_before", "parameter_hash_after")}
    source.update(protocol={"samples": 16}, rows=[row])
    old = deepcopy(source)
    m.check_physical_source_equivalence(source, old)
    old["rows"][0]["selection"]["proposal_rows"][0]["cem_future_goal_cost"] = 3.0
    with pytest.raises(ValueError, match="proposal"):
        m.check_physical_source_equivalence(source, old)
