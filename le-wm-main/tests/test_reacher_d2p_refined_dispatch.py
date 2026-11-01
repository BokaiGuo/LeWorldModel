import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_refined_dispatch as compiler
from experiments.spectral_reachability.reacher_d2p_refined_dispatch import select_refined_bank


def rows():
    return [{"proposal_index": 0, "immediate_goal_cost": 1., "refined_goal_cost": .5},
        {"proposal_index": 1, "immediate_goal_cost": .3, "refined_goal_cost": .2},
        {"proposal_index": 2, "immediate_goal_cost": 1.2, "refined_goal_cost": .1},
        {"proposal_index": 3, "immediate_goal_cost": .5, "refined_goal_cost": None}]


def test_full_bank_refines_all_eligible_candidates_and_both_controls():
    decision = select_refined_bank(rows())
    assert decision["selected_program_index"] == 2
    assert decision["refined_proposal_indices"] == [0, 1, 2]
    assert decision["reconfiguration_witness_found"]
    assert not decision["physical_labels_used"]


def test_missing_eligible_score_cannot_silently_prune_candidate():
    value = rows()
    value[2]["refined_goal_cost"] = None
    with pytest.raises(ValueError):
        select_refined_bank(value)


def test_no_winner_falls_back_to_same_bank_greedy():
    value = rows()
    value[2]["refined_goal_cost"] = .3
    decision = select_refined_bank(value)
    assert decision["selected_program_index"] == 1
    assert not decision["reconfiguration_witness_found"]


def test_predicted_immediate_progress_not_used_as_reconfiguration():
    value = rows()
    value[3]["refined_goal_cost"] = 0.
    assert select_refined_bank(value)["selected_program_index"] == 2


def test_nonfinite_control_rejected():
    value = rows()
    value[0]["refined_goal_cost"] = float("nan")
    with pytest.raises(ValueError):
        select_refined_bank(value)


def test_compiler_visits_every_required_candidate_and_accounts_suffix_work(monkeypatch):
    visits = []
    monkeypatch.setattr(compiler.p0, "_cost_fn", lambda *a, **kw: torch.tensor([1., .3, 1.2, .5]))
    monkeypatch.setattr(compiler.p0, "_goal_latent", lambda *a: torch.zeros(1))
    def state(_model, _pixels, _history, params, **kw):
        index = int(params[0, 0, 0])
        visits.append(index)
        return torch.full((1, 3, 1), float(index)), torch.zeros(1, 3, 1)
    monkeypatch.setattr(compiler.p1, "_state_after_prefix", state)
    def snapshots(_model, _goal, emb, _actions, **kw):
        index = int(emb[0, 0, 0])
        receipt = {"trace": [{"cost_after": [.5, .2, .1][index]}],
            "backward_vjp_rows": 8, "model_predict_sample_evaluations": 14}
        return {8: (torch.zeros(1, 2, 1), receipt)}
    monkeypatch.setattr(compiler.solver, "compile_snapshots", snapshots)
    bank = torch.arange(4.).reshape(4, 1, 1)
    chosen, receipt = compiler.compile_refined_dispatch(object(), torch.zeros(1), torch.zeros(1),
        torch.zeros(1, 3, 1), proposal_bank=bank, action_lower=torch.tensor([-1.]), action_upper=torch.tensor([1.]))
    assert visits == [0, 1, 2]
    assert torch.equal(chosen, bank[2:3])
    assert receipt["model_predict_sample_evaluations"] == 4 + 3 * (1 + 14)
    assert receipt["backward_vjp_rows"] == 3 * 8
    assert receipt["proposal_rows"][3]["refined_goal_cost"] is None
