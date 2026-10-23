import copy

from experiments.gdd_wm.audit_pointmaze_v6_prospective_routing import _role, audit


def _row(cost_regret: float, endpoint_regret: float, endpoint_scores: list[float], cost_scores: list[float]) -> dict:
    endpoint_choice = str(min(range(len(endpoint_scores)), key=lambda index: endpoint_scores[index]))
    cost_choice = str(min(range(len(cost_scores)), key=lambda index: cost_scores[index]))
    return {
        "candidate_ids": [str(index) for index in range(len(endpoint_scores))],
        "endpoint_predicted_cost": endpoint_scores,
        "cost_predicted_cost": cost_scores,
        "endpoint_candidate_id": endpoint_choice,
        "cost_candidate_id": cost_choice,
        "endpoint_regret": endpoint_regret,
        "cost_regret": cost_regret,
    }


def _diagnostic(high_agreement: dict[str, bool]) -> dict:
    return {
        "available": True,
        "per_root": {
            root: {
                "member_agreement_with_deployed_choice": 1.0 if high else 0.25,
                "distinct_member_winner_count": 1 if high else 4,
                "mean_candidate_score_standard_deviation": 0.1 + index * 0.01,
                "selected_endpoint_xy_spread_mean": 0.1 + index * 0.02,
            }
            for index, (root, high) in enumerate(sorted(high_agreement.items()))
        },
    }


def _report() -> dict:
    by_role = {role: [] for role in range(3)}
    for value in range(10_000):
        root = f"root-{value}"
        role = _role(root)
        if len(by_role[role]) < 24:
            by_role[role].append(root)
        if all(len(roots) == 24 for roots in by_role.values()):
            break
    roots = [root for role in range(3) for root in by_role[role]]
    high_agreement = {root: bool(index % 2) for role in range(3) for index, root in enumerate(by_role[role])}
    rows = {}
    for root in roots:
        # High agreement has an endpoint advantage; low agreement should use
        # the cost arm.  The audit can recover this only from calibration.
        high = high_agreement[root]
        rows[root] = _row(
            cost_regret=0.3,
            endpoint_regret=0.1 if high else 0.5,
            endpoint_scores=[0.1, 0.3, 0.5],
            cost_scores=[0.2, 0.4, 0.7],
        )
    diagnostic = _diagnostic(high_agreement)
    return {
        "schema": "pointmaze_counterfactual_endpoint_control.v1",
        "status": "completed_counterfactual_endpoint_control",
        "control_contract": {"endpoint_ensemble_disagreement_diagnostic_only": True},
        "calibration": {"per_root": rows},
        "outcome": {"per_root": copy.deepcopy(rows)},
        "endpoint_ensemble_diagnostics_pre_outcome": {
            "calibration": diagnostic,
            "outcome": copy.deepcopy(diagnostic),
        },
    }


def test_audit_uses_only_selector_thresholds_and_requires_certification_before_outcome_claim():
    result = audit(_report(), bootstrap_draws=1000, seed=7)
    assert result["status"] == "completed_post_execution_prospective_routing_audit"
    assert result["root_counts"]["outcome"] == 72
    assert sum(result["root_counts"][role] for role in ("selector", "certification", "risk_calibration")) == 72
    assert all(result["root_counts"][role] == 24 for role in ("selector", "certification", "risk_calibration"))
    assert result["selector"]["chosen"]["route"]["feature"] == "ensemble_member_agreement"
    assert result["certification"]["would_authorize_on_calibration"] is True
    assert result["outcome"]["would_improve_over_cost"] is True
    assert result["decision"]["fresh_panel_route_warranted"] is True


def test_audit_rejects_missing_or_outcome_leaking_diagnostic_contract():
    report = _report()
    report["control_contract"]["endpoint_ensemble_disagreement_diagnostic_only"] = False
    try:
        audit(report, bootstrap_draws=1000, seed=7)
    except ValueError as error:
        assert "pre-outcome ensemble diagnostic contract" in str(error)
    else:
        raise AssertionError("missing diagnostic contract must fail closed")
