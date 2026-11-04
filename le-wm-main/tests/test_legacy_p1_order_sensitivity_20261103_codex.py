from experiments.empirical_spectral_audit.legacy_p1_order_sensitivity_20261103_codex import changes, snapshot


def test_snapshot_normalizes_clique_order_but_tracks_selector():
    def result(eligible, choice):
        return {"strata": [{"environment": "e", "training_stage": 1, "latent_dimension": 2,
                            "action_distribution": "a", "eligible_checkpoints": eligible,
                            "identifiable": True, "reason_if_gated": None,
                            "selectors": {"mse_only": {"checkpoint": "a"},
                                          "full_mse_action_risk_geometry": {"checkpoint": choice}},
                            "delta_risk_mse_minus_full": 0.0}],
                "summary": {"primary_delta_risk_mse_minus_full": {},
                            "claim_ready_for_positive_delta": False}}
    reference = snapshot(result(["a", "b"], "a"))
    assert changes(reference, snapshot(result(["b", "a"], "a"))) == {}
    assert list(changes(reference, snapshot(result(["b", "a"], "b"))).values()) == [["full"]]


def test_changed_strata_fail_closed():
    import pytest
    with pytest.raises(ValueError):
        changes({"rows": {"a": {}}}, {"rows": {"b": {}}})
