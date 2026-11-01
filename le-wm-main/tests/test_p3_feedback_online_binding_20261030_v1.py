"""CPU-only wiring and failure injection; host.run and consumer are test doubles."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import p3_feedback_online_binding_20261030_v1 as m
from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as saved


@pytest.fixture
def setup_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    monkeypatch.setattr(m, "verify_inputs", lambda: None)
    run = tmp_path / (m.host.NAME + "_RUN_fixture")
    receipt = tmp_path / (m.host.NAME + "_receipt_fixture.json")
    analysis = tmp_path / (m.NAME + "_analysis_fixture.json")
    plan_path = tmp_path / (m.NAME + "_plan_fixture.json")
    plan = m.make_plan(run, receipt, analysis)
    plan_sha = m.budget.save_new(plan_path, plan)
    return plan_path, plan_sha, plan


def test_dispatch_once_with_explicit_guards(setup_plan, monkeypatch):
    path, plan_sha, plan = setup_plan
    calls = []
    monkeypatch.setattr(m.host, "run", lambda *a: calls.append(a) or {"status": "four_complete_candidate"})
    result = m.run(path, plan_sha, "external-ticket-123", m.host.AUTH, m.ACK)
    assert result["status"] == "four_complete_candidate"
    assert len(calls) == 1
    assert calls[0][2] == plan["run_path"]
    assert json.loads(calls[0][4])["plan_sha256"] == plan_sha
    assert json.loads(calls[0][4])["prerun_assertion"] == "operator_asserted_not_verified"


@pytest.mark.parametrize("authorization,ack,reference", [
    ("", m.ACK, "external-ticket-123"),
    (m.host.AUTH, "", "external-ticket-123"),
    (m.host.AUTH, m.ACK, "short"),
    (m.host.AUTH, m.ACK, "ticket\nnewline"),
])
def test_reject_without_dispatch(setup_plan, monkeypatch, authorization, ack, reference):
    path, plan_sha, _ = setup_plan
    calls = []
    monkeypatch.setattr(m.host, "run", lambda *a: calls.append(a))
    with pytest.raises(ValueError):
        m.run(path, plan_sha, reference, authorization, ack)
    assert calls == []


def test_output_collision_and_wrong_plan_hash_reject(setup_plan, monkeypatch):
    path, plan_sha, plan = setup_plan
    calls = []
    monkeypatch.setattr(m.host, "run", lambda *a: calls.append(a))
    with pytest.raises(ValueError):
        m.run(path, "0" * 64, "external-ticket-123", m.host.AUTH, m.ACK)
    Path(plan["run_path"]).mkdir()
    with pytest.raises(ValueError, match="already exists"):
        m.run(path, plan_sha, "external-ticket-123", m.host.AUTH, m.ACK)
    assert calls == []


def test_source_drift_reject(setup_plan, monkeypatch):
    path, plan_sha, _ = setup_plan
    monkeypatch.setattr(m.budget, "sha", lambda p: "changed" if str(p) == m.__file__ else original_sha(p))
    with pytest.raises(ValueError, match="drift"):
        m.verify_plan(path, plan_sha)


original_sha = m.budget.sha


def test_paths_must_be_direct_canonical_and_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    monkeypatch.setattr(m, "verify_inputs", lambda: None)
    with pytest.raises(ValueError):
        m.make_plan(tmp_path / "wrong", tmp_path / (m.host.NAME + "_receipt_x"),
                    tmp_path / (m.NAME + "_analysis_x"))
    run = tmp_path / (m.host.NAME + "_RUN_x")
    run.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        m.make_plan(run, tmp_path / (m.host.NAME + "_receipt_x"), tmp_path / (m.NAME + "_analysis_x"))


def test_partial_consumption_does_not_upgrade_or_retry(setup_plan, monkeypatch):
    path, plan_sha, plan = setup_plan
    run = Path(plan["run_path"])
    run.mkdir()
    reference = "external-ticket-123"
    report_sha = m.host.write_new(run / "report.json", {
        "coordination": m.coordination(plan_sha, reference), "prepare_sha256": m.budget.PREPARE_SHA,
        "scientific_pass": False, "status": "partial_or_failed"})
    completion_sha = m.host.write_new(run / "completion.json", {
        "report_sha256": report_sha, "scientific_pass": False, "complete": False})
    calls = []
    def fake_consume(*args):
        calls.append(args)
        return {"status": "partial_preserved_not_certified", "full_saved_strict": False, "scientific_pass": False,
                "report_sha256": report_sha, "completion_sha256": completion_sha}
    monkeypatch.setattr(saved, "consume", fake_consume)
    with pytest.raises(ValueError, match="run-plan binding"):
        m.consume(path, plan_sha, "different-ticket", report_sha, completion_sha)
    assert not calls
    result = m.consume(path, plan_sha, reference, report_sha, completion_sha)
    assert result["status"] == "partial_preserved_not_certified"
    assert len(calls) == 1
    assert json.loads(Path(plan["receipt_path"]).read_text())["full_saved_strict"] is False
    with pytest.raises(ValueError, match="already exists"):
        m.consume(path, plan_sha, reference, report_sha, completion_sha)
    assert len(calls) == 1


def test_inspect_requires_matching_run_binding(setup_plan, monkeypatch):
    path, plan_sha, plan = setup_plan
    run = Path(plan["run_path"])
    run.mkdir()
    report_sha = m.host.write_new(run / "report.json", {"coordination": "old-run",
        "prepare_sha256": m.budget.PREPARE_SHA, "scientific_pass": False})
    completion_sha = m.host.write_new(run / "completion.json", {"report_sha256": report_sha,
        "scientific_pass": False})
    receipt_sha = m.host.write_new(plan["receipt_path"], {"report_sha256": report_sha,
        "completion_sha256": completion_sha})
    monkeypatch.setattr(m.budget, "inspect", lambda *args: pytest.fail("must not inspect old run"))
    with pytest.raises(ValueError, match="run-plan binding"):
        m.inspect(path, plan_sha, "external-ticket-123", receipt_sha)


def test_complete_wiring_sequence_uses_strict_receipt_and_contract(setup_plan, monkeypatch):
    path, plan_sha, plan = setup_plan
    ref = "external-ticket-123"
    run = Path(plan["run_path"])
    emitted = {}
    def fake_run(prepared, prepare_sha, output, authorization, coordination):
        run.mkdir()
        report_sha = m.host.write_new(run / "report.json", {"coordination": coordination,
            "prepare_sha256": prepare_sha, "scientific_pass": False})
        completion_sha = m.host.write_new(run / "completion.json", {"report_sha256": report_sha,
            "scientific_pass": False})
        emitted.update(report_sha=report_sha, completion_sha=completion_sha)
        return {"status": "four_complete_candidate", "report_sha256": report_sha,
                "completion_sha256": completion_sha}
    monkeypatch.setattr(m.host, "run", fake_run)
    monkeypatch.setattr(saved, "consume", lambda output, report, completion: {
        "status": "four_saved_feedback_engineering_verified_known_roots", "full_saved_strict": True,
        "report_sha256": report, "completion_sha256": completion, "scientific_pass": False})
    inspected = []
    def fake_inspect(*args):
        inspected.append(args)
        return {"status": "complete_saved_measured_work_only", "scientific_pass": False,
                "pre_run_external_timestamp_verified": False}
    monkeypatch.setattr(m.budget, "inspect", fake_inspect)
    monkeypatch.setattr(m, "settle_full_run", lambda *args: {"test_only": True})
    m.run(path, plan_sha, ref, m.host.AUTH, m.ACK)
    receipt = m.consume(path, plan_sha, ref, emitted["report_sha"], emitted["completion_sha"])
    result = m.inspect(path, plan_sha, ref, receipt["receipt_sha256"])
    assert result["status"] == "complete_saved_measured_work_only"
    assert inspected[0][0] == plan["run_path"]
    assert inspected[0][1] == plan["receipt_path"]
    out = json.loads(Path(plan["analysis_path"]).read_text())
    assert out["online_binding"]["plan_sha256"] == plan_sha
    assert out["online_binding"]["external_prerun_timestamp_verified"] is False
    with pytest.raises(ValueError, match="already exists"):
        m.inspect(path, plan_sha, ref, receipt["receipt_sha256"])


def test_inspect_rejects_forged_strict_receipt(setup_plan, monkeypatch):
    path, plan_sha, plan = setup_plan
    run = Path(plan["run_path"])
    run.mkdir()
    ref = "external-ticket-123"
    report_sha = m.host.write_new(run / "report.json", {"coordination": m.coordination(plan_sha, ref),
        "prepare_sha256": m.budget.PREPARE_SHA, "scientific_pass": False})
    completion_sha = m.host.write_new(run / "completion.json", {"report_sha256": report_sha, "scientific_pass": False})
    forged = {"report_sha256": report_sha, "completion_sha256": completion_sha,
              "status": "four_saved_feedback_engineering_verified_known_roots", "full_saved_strict": True}
    receipt_sha = m.host.write_new(plan["receipt_path"], forged)
    monkeypatch.setattr(saved, "consume", lambda *args: dict(forged, full_saved_strict=False))
    monkeypatch.setattr(m.budget, "inspect", lambda *args: pytest.fail("must not use forged receipt"))
    with pytest.raises(ValueError, match="receipt reconstruction mismatch"):
        m.inspect(path, plan_sha, ref, receipt_sha)
    assert not Path(plan["analysis_path"]).exists()


def test_matching_actual_counts_do_not_substitute_for_common_cap(monkeypatch):
    monkeypatch.setattr(m.host.life.seam.core, "audit_ledger", lambda _: None)
    ledger = {"caps": deepcopy(m.budget.EXPECTED), "attempted": deepcopy(m.budget.EXPECTED),
              "completed": deepcopy(m.budget.EXPECTED)}
    m.validate_full_ledger(ledger)
    ledger["caps"]["predict_API"] += 1
    with pytest.raises(ValueError, match="common cap"):
        m.validate_full_ledger(ledger)
    ledger["caps"] = deepcopy(m.budget.EXPECTED)
    ledger["completed"]["predict_API"] -= 1
    with pytest.raises(ValueError, match="actual budget"):
        m.validate_full_ledger(ledger)
