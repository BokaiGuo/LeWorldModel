from experiments.theory_spectral_transfer.horizon_sufficient_control import (
    evaluate_horizon_sufficiency,
)


def _kwargs():
    return {
        "horizon": 4,
        "declared_variant": "fourier_plus_phase",
        "observed_variant": "fourier_plus_phase",
        "support_gate": {"passed": True},
        "action_identifiability_gate": {"passed": True},
        "successor_consistency_gate": {"passed": True},
        "risk_bound": 0.2,
        "risk_threshold": 0.5,
        "selector_stability": 0.62,
    }


def test_horizon_sufficient_control_permits_only_all_gates():
    result = evaluate_horizon_sufficiency(**_kwargs())
    assert result["status"] == "permitted"
    assert result["selector_is_not_a_gate"] is True


def test_horizon_sufficient_control_fails_closed_on_action_support():
    kwargs = _kwargs()
    kwargs["action_identifiability_gate"] = {"passed": False}
    result = evaluate_horizon_sufficiency(**kwargs)
    assert result["status"] == "fallback_fail_closed"
    assert result["fallback_required"] is True
