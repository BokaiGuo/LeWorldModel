import pytest
import torch
from copy import deepcopy

from experiments.spectral_reachability.audit_reacher_d2p_model_gap_prerequisites import gap_summary
from experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration import _accept_suffix_update, _reachability_value
from experiments.spectral_reachability.audit_reacher_d2p_suffix_optimizer_repair import compare


class LinearFixture:
    @staticmethod
    def action_encoder(actions):
        return actions

    @staticmethod
    def predict(embeddings, actions):
        return embeddings[:, -1:] + actions[:, -1:]


def test_diagnostic_grid_reuses_the_unmodified_production_dls_score():
    args = (LinearFixture(), torch.ones(1), torch.zeros(1, 3, 1), torch.zeros(1, 3, 1))
    kwargs = dict(future_horizon=1, action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]), damping=0.1)
    base = _reachability_value(*args, **kwargs)
    profiled = _reachability_value(*args, **kwargs, diagnostic_dampings=(0.01, 0.1, 1.0))
    for key in base:
        if key not in ("forward_rollout_evaluations", "model_predict_sample_evaluations"):
            assert profiled[key] == base[key]
    assert base["forward_rollout_evaluations"] == 7
    assert profiled["forward_rollout_evaluations"] == 22
    assert profiled["model_predict_sample_evaluations"] == 22
    same = profiled["diagnostic_damping_scores"][1]
    for key in ("value", "bounded_dls_suffix_goal_cost", "unbounded_dls_parameter_l2"):
        assert same[key] == base[key]
    assert base["first_order_unreachable_residual_fraction"] == 0.0


def test_nonlinear_ascent_reduces_the_actual_returned_update():
    def terminal(params):
        return params.square()
    params, prediction, receipt = _accept_suffix_update(
        terminal, torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([1.0]), step_rule="backtracking")
    assert receipt["full_step_bounded_dls_suffix_goal_cost"] == 9.0
    assert receipt["accepted_step_scale"] == 0.5
    assert torch.equal(params, torch.tensor([1.0]))
    assert torch.equal(terminal(params), prediction)


def test_zero_fallback_when_every_nonzero_step_is_worse():
    params, prediction, receipt = _accept_suffix_update(
        lambda value: -value.square(), torch.tensor([2.0]), torch.tensor([0.0]),
        torch.tensor([1.0]), step_rule="backtracking")
    assert receipt["accepted_step_scale"] == 0.0
    assert torch.equal(params, torch.zeros(1))
    assert torch.equal(prediction, torch.zeros(1))


def test_actual_nonlinear_dls_returns_the_accepted_shrunk_action():
    class CurvedFixture(LinearFixture):
        @staticmethod
        def predict(embeddings, actions):
            value = actions[:, -1:]
            return embeddings[:, -1:] + value + 4 * value.square()
    kwargs = dict(future_horizon=1, action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]),
                  damping=0.01, return_suffix_params=True)
    args = (CurvedFixture(), torch.ones(1), torch.zeros(1, 3, 1), torch.zeros(1, 3, 1))
    receipt = _reachability_value(*args, **kwargs)
    assert receipt["full_step_bounded_dls_suffix_goal_cost"] > receipt["predicted_terminal_goal_cost"]
    assert 0 < receipt["accepted_step_scale"] < 1
    params = receipt["_bounded_dls_suffix_params"]
    value = torch.tanh(params)
    replayed_cost = float((value + 4 * value.square() - 1).square().mean())
    assert replayed_cost == pytest.approx(receipt["bounded_dls_suffix_goal_cost"])
    assert replayed_cost < receipt["predicted_terminal_goal_cost"]


def test_vjp_rows_count_output_dimension_not_singular_values():
    class TallFixture(LinearFixture):
        calls = 0

        def predict(self, embeddings, actions):
            self.calls += int(embeddings.shape[0])
            return embeddings[:, -1:] + actions[:, -1:]
    model = TallFixture()
    receipt = _reachability_value(model, torch.ones(3), torch.zeros(1, 3, 3), torch.zeros(1, 3, 1),
        future_horizon=1, action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]), damping=0.1,
        return_suffix_params=True)
    assert len(receipt["singular_values"]) == 1
    assert receipt["backward_vjp_rows"] == 3
    assert receipt["model_predict_sample_evaluations"] == model.calls == 7
    params = receipt["_bounded_dls_suffix_params"]
    assert float((torch.tanh(params).reshape(-1) - 1).square().mean()) == pytest.approx(receipt["bounded_dls_suffix_goal_cost"])


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_damping_is_an_implementation_error(value):
    with pytest.raises(ValueError, match="finite and positive"):
        _reachability_value(LinearFixture(), torch.ones(1), torch.zeros(1, 3, 1), torch.zeros(1, 3, 1),
                            future_horizon=1, action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]), damping=value)


def test_gap_profile_does_not_accept_a_zero_only_false_witness():
    rows = [{"proposal_index": 0, "immediate_goal_cost": 4.0, "bounded_dls_suffix_goal_cost": 3.0},
            {"proposal_index": 1, "immediate_goal_cost": 1.0, "bounded_dls_suffix_goal_cost": 0.1},
            {"proposal_index": 2, "immediate_goal_cost": 4.1, "bounded_dls_suffix_goal_cost": 2.0}]
    result = gap_summary(rows)
    assert result["strong_future_witness_count"] == 0
    assert result["best_non_greedy_minus_greedy_future_cost"] == pytest.approx(1.9)
    rows[2]["bounded_dls_suffix_goal_cost"] = 0.01
    result = gap_summary(rows)
    assert result["strong_future_witness_count"] == 1
    assert result["selected_index"] == 2


def test_gap_profile_rejects_incomplete_or_nonfinite_banks():
    with pytest.raises(ValueError, match="complete"):
        gap_summary([{"proposal_index": 1}])
    with pytest.raises(ValueError, match="finite"):
        gap_summary([{"proposal_index": 0, "immediate_goal_cost": float("nan"), "bounded_dls_suffix_goal_cost": 0.0}])


def test_optimizer_comparison_pairs_by_root_identity_and_rejects_missing_roots():
    protocol = {"weights_sha256": "weights", "reference_sha256": "reference", "dampings": [0.1],
                "future_horizon": 2, "root_seeds": [1, 2]}
    row = {"proposal_index": 0, "immediate_goal_cost": 1.0, "predicted_terminal_goal_cost": 1.0,
           "singular_values": [1.0], "diagnostic_damping_scores": [{"damping": 0.1, "bounded_dls_suffix_goal_cost": 2.0}]}
    summary = [{"damping": 0.1, "strong_future_witness_count": 0, "selected_index": None}]
    old = {"status": "completed", "protocol": protocol, "roots": [
        {"seed": seed, "root_rgb_sha256": str(seed), "goal_rgb_sha256": "goal", "proposal_rows": [deepcopy(row)],
         "summary": deepcopy(summary)} for seed in (1, 2)]}
    new = deepcopy(old)
    new["protocol"]["suffix_step_rule"] = "backtracking"
    new["roots"].reverse()
    for root in new["roots"]:
        root["proposal_rows"][0]["diagnostic_damping_scores"][0].update({
            "bounded_dls_suffix_goal_cost": 0.5, "full_step_bounded_dls_suffix_goal_cost": 2.0,
            "accepted_step_scale": 0.5})
    result = compare(old, new)
    assert result["status"] == "pass"
    assert result["old_full_step_ascent_count_by_damping"]["0.1"] == 2
    new["roots"].pop()
    with pytest.raises(ValueError, match="root identity"):
        compare(old, new)
