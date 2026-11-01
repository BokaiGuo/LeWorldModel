from copy import deepcopy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_causal_innovation import METHODS, summarize, validate_reference


def test_fixed_branch_summary_separates_past_correction_from_raw_model_fidelity():
    zero = {"raw_predicted_goal_cost": 1., "corrected_predicted_goal_cost": 1., "observed_goal_cost": 1.,
        "raw_latent_l2_error": 2., "corrected_latent_l2_error": 1.}
    candidate = {**zero, "raw_predicted_goal_cost": 1.1, "corrected_predicted_goal_cost": .9, "observed_goal_cost": .8}
    rows = [{"model_eligible": True, "methods": {"dispatch": candidate, "zero": zero}}]
    result = summarize(rows)
    assert result["model_eligible_roots"]["raw_strict_order_agreements_vs_zero"] == 0
    assert result["model_eligible_roots"]["corrected_strict_order_agreements_vs_zero"] == 1
    assert result["all_branch_prediction_error"]["count"] == 2


def test_reference_requires_all_declared_roots_and_all_fixed_branches():
    report = {"schema": "reacher_d2p_program_dispatch_diagnostic.v1", "status": "completed", "gates": {"intact": True},
        "sealed_protocol": {"root_seeds": [1, 2]}, "rows": [{"seed": seed, "methods": {name: {} for name in METHODS}} for seed in (1, 2)]}
    validate_reference(report)
    for broken in ("missing", "duplicate", "branch", "gate"):
        variant = deepcopy(report)
        if broken == "missing":
            variant["rows"].pop()
        elif broken == "duplicate":
            variant["rows"][1]["seed"] = 1
        elif broken == "branch":
            variant["rows"][0]["methods"].pop("zero")
        else:
            variant["gates"]["intact"] = False
        with pytest.raises(ValueError):
            validate_reference(variant)
