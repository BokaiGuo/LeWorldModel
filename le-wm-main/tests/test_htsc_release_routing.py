from __future__ import annotations

import copy

import pytest

from experiments.decision_benchmark.audit_htsc_release_routing import audit_report, recovery_route
from experiments.decision_benchmark.run_htsc_matched_transfer import evaluate_records


def _row(root: str, split: str, candidate: int, *, wm: float, fallback: float, truth: float,
         risk: float = 0.1, support: bool = True) -> dict[str, object]:
    return {
        "root_id": root,
        "split": split,
        "candidate_id": candidate,
        "action_sequence": [[float(candidate), 0.0]],
        "wm_score": wm,
        "fallback_score": fallback,
        "true_cost": truth,
        "success": truth == 0.0,
        "risk_score": risk,
        "support_passed": support,
    }


def _panel() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(10):
        rows += [
            _row(f"cal-{index}", "calibration", 0, wm=0.0, fallback=1.0, truth=0.0),
            _row(f"cal-{index}", "calibration", 1, wm=1.0, fallback=0.0, truth=1.0),
        ]
    rows += [
        _row("train", "train", 0, wm=0.0, fallback=1.0, truth=0.0),
        _row("train", "train", 1, wm=1.0, fallback=0.0, truth=1.0),
        # All three failure modes are simultaneously present.  The evaluator
        # must expose this instead of allowing its one status/reason to hide it.
        _row("test", "test", 0, wm=0.0, fallback=1.0, truth=1.0, risk=10.0, support=False),
        _row("test", "test", 1, wm=0.0, fallback=0.0, truth=0.0, risk=10.0, support=False),
    ]
    return rows


def test_failure_vector_preserves_simultaneous_support_uncertainty_and_coverage_causes():
    report = evaluate_records(_panel(), alpha=0.1, risk_quantile=0.9, bootstrap_draws=50, seed=3)
    root = report["test"]["roots"][0]
    assert root["status"] == "abstain"
    assert root["gate_failure_vector"] == {
        "support_failure": True,
        "uncertainty_failure": True,
        "certificate_margin_failure": True,
    }
    assert root["posthoc_utility_ranking_failure"] is True


def test_recovery_route_is_model_only_and_priority_is_fail_closed():
    assert recovery_route({"support_failure": True, "uncertainty_failure": False, "certificate_margin_failure": False}) == "abstain_collect_new_support"
    assert recovery_route({"support_failure": False, "uncertainty_failure": True, "certificate_margin_failure": True}) == "reobserve_uncertainty"
    assert recovery_route({"support_failure": False, "uncertainty_failure": False, "certificate_margin_failure": True}) == "short_observed_effect_probe"
    assert recovery_route({"support_failure": False, "uncertainty_failure": False, "certificate_margin_failure": False}) == "permit_frozen_htsc_choice"


def test_audit_preserves_exact_matched_fallback_mask_and_never_uses_test_cost_for_route():
    initial = audit_report(evaluate_records(_panel(), alpha=0.1, risk_quantile=0.9, bootstrap_draws=50, seed=3))
    altered = copy.deepcopy(_panel())
    for row in altered:
        if row["split"] == "test":
            row["true_cost"] = 1.0 - float(row["true_cost"])
    replay = audit_report(evaluate_records(altered, alpha=0.1, risk_quantile=0.9, bootstrap_draws=50, seed=3))
    assert initial["root_routes"][0]["route"] == replay["root_routes"][0]["route"]
    assert initial["root_routes"][0]["matched_fallback_mask_preserved"] is True
    assert replay["root_routes"][0]["matched_fallback_mask_preserved"] is True


def test_missing_vector_field_fails_loudly():
    with pytest.raises(KeyError):
        recovery_route({"support_failure": False, "uncertainty_failure": False})
