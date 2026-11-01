import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_reconfiguration_reference import future_reference_row, improves_future_reference
from experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration import _declared_future_score_improvement
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as selector


def test_zero_improvement_is_not_a_strong_comparator_witness():
    rows = [{"proposal_index": 0, "immediate_goal_cost": 4.0, "bounded_dls_suffix_goal_cost": 3.0},
            {"proposal_index": 1, "immediate_goal_cost": 1.0, "bounded_dls_suffix_goal_cost": 0.1},
            {"proposal_index": 2, "immediate_goal_cost": 4.1, "bounded_dls_suffix_goal_cost": 2.0}]
    candidate = rows[2]
    assert improves_future_reference(candidate, future_reference_row(rows, "zero_prefix"), score="bounded_dls_suffix_goal_cost", margin=0)
    assert not improves_future_reference(candidate, future_reference_row(rows, "same_bank_greedy"), score="bounded_dls_suffix_goal_cost", margin=0)


def test_guard_allows_a_genuine_predicted_future_gain():
    rows = [{"proposal_index": 0, "immediate_goal_cost": 1.0, "value": 4.0},
            {"proposal_index": 1, "immediate_goal_cost": 2.0, "value": 0.1}]
    assert improves_future_reference(rows[1], future_reference_row(rows, "same_bank_greedy"), score="value", margin=0.2)
    assert not improves_future_reference(rows[0], rows[0], score="value", margin=0)


def test_declared_score_uses_the_explicit_strong_reference():
    receipt = {"future_reference": "same_bank_greedy", "future_reference_score_improvement": -2.0,
               "bounded_dls_suffix_goal_cost_improvement": 3.0}
    assert _declared_future_score_improvement(receipt, "bounded_dls_suffix_goal_cost") == -2.0


@pytest.mark.parametrize("reference", ["invalid", ""])
def test_unknown_references_fail_closed(reference):
    with pytest.raises(ValueError):
        future_reference_row([{"proposal_index": 0, "immediate_goal_cost": 1.0}], reference)


def test_actual_compiler_rejects_the_zero_only_false_witness(monkeypatch):
    # No learned model: all model primitives are synthetic contract fixtures.
    monkeypatch.setattr(selector.p0, "_goal_latent", lambda *args: torch.zeros(1))
    monkeypatch.setattr(selector.p0, "_cost_fn", lambda *args, **kwargs: torch.tensor([4.0, 1.0, 4.1]))
    monkeypatch.setattr(selector, "_state_after_prefix", lambda model, pixels, history, params, **kwargs: (int(params[0, 0, 0]), None))
    def reach(model, goal, index, unused, **kwargs):
        cost = [3.0, 0.1, 2.0][index]
        return {"value": cost, "bounded_dls_suffix_goal_cost": cost,
                "predicted_terminal_goal_cost": cost, "singular_values": [2.0, 1.0],
                "forward_rollout_evaluations": 7, "model_predict_sample_evaluations": 14, "backward_vjp_rows": 3}
    monkeypatch.setattr(selector, "_reachability_value", reach)
    kwargs = dict(reconfiguration_horizon=1, future_horizon=2, action_lower=torch.tensor([-1.0]),
                  action_upper=torch.tensor([1.0]), damping=10, proposals=3, non_greedy_margin=1e-6,
                  reachability_margin=0.0, seed=1, proposal_bank=torch.arange(3.0).reshape(3, 1, 1))
    args = (object(), torch.zeros(1, 3, 1), torch.zeros(1), torch.zeros(1, 3, 1))
    old, _ = selector._compile_reconfiguration(*args, **kwargs)
    assert old is not None and float(old[0, 0, 0]) == 2
    source, source_receipt = selector._compile_reconfiguration(
        *args, **kwargs, proposal_distribution=selector.SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION)
    assert torch.equal(source, old)
    assert source_receipt["proposal_distribution"] == selector.SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION
    strong, receipt = selector._compile_reconfiguration(*args, **kwargs, future_reference="same_bank_greedy")
    assert strong is None
    assert not receipt["non_greedy_witness_found"]
    assert receipt["future_reference_proposal_index"] == 1
