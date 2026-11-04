import pytest
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as selector


@pytest.mark.parametrize("radius,expected_forward,expected_predict,expected_vjp", [
    (0.0, 27, 48, 9),
    (0.02, 43, 78, 15),
])
def test_robust_selector_accounts_for_all_prefix_and_suffix_work(
    monkeypatch, radius, expected_forward, expected_predict, expected_vjp,
):
    monkeypatch.setattr(selector.p0, "_goal_latent", lambda *args: torch.zeros(1))
    monkeypatch.setattr(selector.p0, "_cost_fn", lambda *args, **kwargs: torch.tensor([1.0, 2.0, 0.5]))
    monkeypatch.setattr(
        selector, "_state_after_prefix",
        lambda model, pixels, history, params, **kwargs: (round(float(params[0, 0, 0])), None),
    )

    def reach(model, goal, index, unused, **kwargs):
        cost = [3.0, 2.0, 4.0][index]
        return {
            "value": cost, "bounded_dls_suffix_goal_cost": cost,
            "singular_values": [2.0, 1.0], "forward_rollout_evaluations": 7,
            "model_predict_sample_evaluations": 14, "backward_vjp_rows": 3,
        }

    monkeypatch.setattr(selector, "_reachability_value", reach)
    _, receipt = selector._compile_reconfiguration(
        object(), torch.zeros(1, 3, 1), torch.zeros(1), torch.zeros(1, 3, 1),
        reconfiguration_horizon=1, future_horizon=2, action_lower=torch.tensor([-1.0]),
        action_upper=torch.tensor([1.0]), damping=0.1, proposals=3,
        non_greedy_margin=1e-6, reachability_margin=0.0, seed=1,
        proposal_bank=torch.arange(3.0).reshape(3, 1, 1),
        robustness_perturbation=radius,
    )
    assert receipt["non_greedy_witness_found"]
    assert receipt["forward_rollout_evaluations"] == expected_forward
    assert receipt["model_predict_sample_evaluations"] == expected_predict
    assert receipt["backward_vjp_rows"] == expected_vjp
