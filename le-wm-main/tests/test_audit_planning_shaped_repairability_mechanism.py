from pathlib import Path

import pytest

from experiments.gdd_wm.audit_planning_shaped_repairability_mechanism import audit, run


def _row(cost_id: str, planning_id: str, cost: float, planning: float, released: bool) -> dict:
    return {
        "cost_candidate_id": cost_id,
        "planning_candidate_id": planning_id,
        "cost_regret": cost,
        "planning_regret": planning,
        "routed_regret": planning if released else cost,
        "planning_beats_cost": planning < cost - 1.0e-12,
        "released_planning": released,
    }


def _report() -> dict:
    rows = {
        "same": _row("a", "a", 0.4, 0.4, False),
        "caught": _row("a", "b", 0.5, 0.1, True),
        "missed": _row("a", "c", 0.6, 0.2, False),
        "harm": _row("a", "d", 0.1, 0.4, True),
    }
    return {"calibration": {"per_root": rows}, "outcome": {"per_root": rows}}


def test_audit_separates_candidate_headroom_effect_and_router_errors() -> None:
    result = audit(_report())
    outcome = result["outcome"]
    assert outcome["candidate_choice"]["disagreement_count"] == 3
    assert outcome["planning_shaped_effect"]["strict_improvement_count_when_disagree"] == 2
    assert outcome["frozen_router"]["precision_for_effective_improvement"] == 0.5
    assert outcome["frozen_router"]["recall_for_effective_improvement"] == 0.5
    assert outcome["frozen_router"]["unreleased_but_improvable_roots"] == ["missed"]
    assert outcome["candidate_bank_headroom"]["cost_oracle_headroom_mean_when_disagree"] == pytest.approx(0.4)


def test_audit_rejects_inconsistent_release_contract() -> None:
    report = _report()
    report["outcome"]["per_root"]["caught"]["routed_regret"] = 0.5
    with pytest.raises(ValueError, match="routed regret"):
        audit(report)


def test_audit_validates_v9_opportunity_fields_against_candidate_identities() -> None:
    report = _report()
    for row in report["outcome"]["per_root"].values():
        disagreement = row["cost_candidate_id"] != row["planning_candidate_id"]
        row["candidate_disagreement"] = disagreement
        row["routing_eligible"] = disagreement
    result = audit(report)
    gate = result["outcome"]["opportunity_gate"]
    assert gate["decision_time_contract_present"] is True
    assert gate["eligible_root_count"] == 3
    assert gate["same_choice_release_count"] == 0
    report["outcome"]["per_root"]["same"]["routing_eligible"] = True
    with pytest.raises(ValueError, match="routing_eligible"):
        audit(report)


def test_audit_rejects_v9_release_of_a_same_choice_root() -> None:
    report = _report()
    row = report["outcome"]["per_root"]["same"]
    row.update({"candidate_disagreement": False, "routing_eligible": False, "released_planning": True})
    with pytest.raises(ValueError, match="opportunity-gated"):
        audit(report)


def test_audit_accepts_v7_all_root_eligibility_when_contract_disables_opportunity_gate() -> None:
    report = _report()
    report["control_contract"] = {"opportunity_gated": False}
    for row in report["outcome"]["per_root"].values():
        row["candidate_disagreement"] = row["cost_candidate_id"] != row["planning_candidate_id"]
        row["routing_eligible"] = True
    result = audit(report)
    assert result["outcome"]["opportunity_gate"]["enabled"] is False
    assert result["outcome"]["opportunity_gate"]["eligible_root_count"] == 4


def test_audit_rejects_non_gated_report_that_marks_a_root_ineligible() -> None:
    report = _report()
    report["control_contract"] = {"opportunity_gated": False}
    for row in report["outcome"]["per_root"].values():
        row["candidate_disagreement"] = row["cost_candidate_id"] != row["planning_candidate_id"]
        row["routing_eligible"] = True
    report["outcome"]["per_root"]["same"]["routing_eligible"] = False
    with pytest.raises(ValueError, match="non-gated"):
        audit(report)


def test_run_refuses_to_overwrite_and_preserves_provenance(tmp_path: Path) -> None:
    import json

    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    output = tmp_path / "audit.json"
    result = run(report_path=report, output=output)
    assert result["provenance"]["report"] == str(report.resolve())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(report_path=report, output=output)
