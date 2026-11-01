import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_four_prefix_saved_audit_20261030 as audit


ROOT = Path(__file__).resolve().parents[1] / "results"
RUN = ROOT / "codex_four_arm_prefix_intervention_20261030_run_MAIN01"
PREP = ROOT / "codex_four_arm_prefix_intervention_20261030_prepare_CODEX03.json"


def test_real_saved_records_and_full_denominator():
    rows = audit.audit(RUN, PREP)
    result = audit.analyze(rows)
    assert result["status_counts"] == {
        "completed15": 22, "censored_out_of_domain": 1, "native_stopped": 1}
    assert len(result["rows"]) == 24
    assert len(result["comparisons"]) == result["fixed_comparisons"] == 28
    assert rows[8]["endpoint"] is None and rows[16]["endpoint"] is None
    assert any(c["left_minus_right_object_position_error"] is None for c in result["comparisons"])
    assert result["independent_confirmation"] is False and result["P1_admission"] is False


def test_schedule_rejects_suffix_and_identity_tamper():
    schedule = audit.read(PREP)["schedule"]
    schedule[2]["suffix"][0][0] = 0.125
    schedule[2]["commands"] = schedule[2]["prefix"] + schedule[2]["suffix"]
    with pytest.raises(ValueError, match="shared saved suffix"):
        audit.schedule_check(schedule)


def test_truncated_or_changed_journal_fails(tmp_path):
    e = {"index": 0, "previous_sha256": None, "kind": "run_started"}
    e["event_sha256"] = audit.digest(e)
    p = tmp_path / "journal.jsonl"
    p.write_text(json.dumps(e) + "\n")
    assert len(audit.journal(p, e["event_sha256"])) == 1
    p.write_text(json.dumps(e))
    with pytest.raises(ValueError, match="truncated"):
        audit.journal(p, e["event_sha256"])
    e["kind"] = "changed"
    p.write_text(json.dumps(e) + "\n")
    with pytest.raises(ValueError, match="journal chain"):
        audit.journal(p, e["event_sha256"])


def test_returned_state_tamper_fails(monkeypatch):
    original = audit.journal
    def altered(path, tail):
        events = original(path, tail)
        target = next(e for e in events if e["kind"] == "branch_returned")
        target["state"][0] += 1
        return events
    monkeypatch.setattr(audit, "journal", altered)
    with pytest.raises(ValueError, match="literal action and returned state"):
        audit.audit(RUN, PREP)


def test_censor_evidence_tamper_fails(monkeypatch):
    original = audit.journal
    def altered(path, tail):
        events = original(path, tail)
        target = next(e for e in events if e["kind"] == "censored_out_of_domain")
        target["proof"] = copy.deepcopy(target["proof"])
        target["proof"]["step"] -= 1
        return events
    monkeypatch.setattr(audit, "journal", altered)
    with pytest.raises(ValueError, match="censor evidence"):
        audit.audit(RUN, PREP)


def test_png_tamper_fails(monkeypatch):
    original_sha = audit.sha
    def altered_sha(path):
        if str(path).endswith("branch_00_plan_01.png"):
            return "0" * 64
        return original_sha(path)
    monkeypatch.setattr(audit, "sha", altered_sha)
    with pytest.raises(ValueError, match="frame file binding"):
        audit.audit(RUN, PREP)


def test_operation_completion_tamper_fails(monkeypatch):
    original = audit.journal
    def altered(path, tail):
        events = original(path, tail)
        target = next(e for e in events if e["kind"] == "operation_completed")
        target["resource"] = "close"
        return events
    monkeypatch.setattr(audit, "journal", altered)
    with pytest.raises(ValueError, match="operation completion"):
        audit.audit(RUN, PREP)
