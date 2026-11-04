from experiments.spectral_reachability import audit_reacher_p1_runner_contract_20261102_exclusive as audit


def test_static_contract_audit_has_explicit_budget_gap():
    report = audit.audit()
    assert report["cuda_started"] is False
    assert report["gate_modified"] is False
    findings = {row["id"]: row for row in report["findings"]}
    assert findings["cli"]["status"] == "verified_static"
    assert findings["p0_evidence"]["status"] == "verified_static"
    assert findings["rgb_action_history_feedback"]["status"] == "verified_static"
    assert findings["physical_execution"]["status"] == "verified_static"
    assert findings["budget_ledger"]["status"] == "gap"
    assert report["claims"]["complete_persistent_budget_ledger"] is False
    assert report["claims"]["equal_total_compute"] is False


def test_p0_is_not_mislabeled_as_candidate_physical_execution():
    report = audit.audit()
    physical = next(row for row in report["findings"] if row["id"] == "physical_execution")
    assert any("P0 preflight does not dispatch" in text for text in physical["evidence"])
