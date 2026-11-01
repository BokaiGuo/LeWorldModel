import copy
import zlib

import pytest
from experiments.spectral_reachability import codex_e1_sealed_event_axis_audit_20261030_v1 as m
from experiments.spectral_reachability import codex_e1_end_to_end_ledger_interface_20261030_v1 as ledger


@pytest.fixture(scope="module")
def saved():
    report = m.parse(m.read_bound(m.REPORT, m.REPORT_SHA, 4 * 1024 * 1024))
    receipt = m.parse(m.read_bound(m.RECEIPT, m.RECEIPT_SHA, 4 * 1024 * 1024))
    records, terminal = m.decode(m.read_bound(m.EVENTS, m.EVENT_SHA, 16 * 1024 * 1024))
    return records, terminal, report, receipt


def test_actual_journal_and_ledger_compatible_values(saved):
    report = m.audit(*saved)
    assert report["journal_events"] == 9382 and len(report["groups"]) == 12
    assert not report["end_to_end_complete"] and not report["equal_budget_certified"]
    for group in report["groups"]:
        n = 144 if group["phase"] == "controller" else 4
        for axis, v in group["measurements"].items():
            assert v["attempted"] == v["completed"] == n
            assert len(v["event_set_sha256"]) == 64
            if axis != "predictor_module_calls":
                assert axis in ledger.AXES and set(v) == {"attempted", "completed", "event_set_sha256"}
    assert report["groups"][0]["measurements"]["predict_api_calls"]["event_set_sha256"] == (
        "9004ef70ea41ef0cda8dc0f6c474194d354bd4ee2eca71bd362fb10dd4ff8f73")


def test_source_pin_and_truncated_gzip(saved):
    with pytest.raises(ValueError):
        m.read_bound(m.EVENTS, "0" * 64, 16 * 1024 * 1024)
    raw = m.read_bound(m.EVENTS, m.EVENT_SHA, 16 * 1024 * 1024)
    with pytest.raises((ValueError, zlib.error)):
        m.decode(raw[:-2])


def test_journal_chain_rejects_changed_content():
    body = {"index": 0, "previous": "0" * 64, "event": {"kind": "test"}}
    record = dict(body, sha256=m.sha(m.canonical(body)))
    record["event"]["kind"] = "altered"
    with pytest.raises(ValueError, match="journal chain"):
        m.decode(zlib.compress(m.canonical(record), wbits=31))


def test_duplicate_attempt_and_missing_return(saved):
    records, terminal, report, receipt = saved
    altered = copy.deepcopy(records)
    altered[5] = copy.deepcopy(altered[4])
    with pytest.raises(ValueError):
        m.audit(altered, terminal, report, receipt)


def test_sample_and_receipt_mismatch(saved):
    records, terminal, report, receipt = saved
    receipt = copy.deepcopy(receipt)
    receipt["rows"][0]["controller_samples"]["attempted"] += 1
    with pytest.raises(ValueError, match="sample count"):
        m.audit(records, terminal, report, receipt)


def test_wrong_terminal_and_slot(saved):
    records, terminal, report, receipt = saved
    with pytest.raises(ValueError, match="report journal"):
        m.audit(records, "0" * 64, report, receipt)
    receipt = copy.deepcopy(receipt)
    receipt["rows"][0]["arm"] = "existing_learned_prior"
    with pytest.raises(ValueError, match="slot binding"):
        m.audit(records, terminal, report, receipt)


def test_failed_dispatch_not_silently_dropped(saved):
    records, terminal, report, receipt = saved
    altered = copy.deepcopy(records)
    altered[4]["event"]["event"]["kind"] = "dispatch_failed"
    with pytest.raises(ValueError, match="unknown dispatch"):
        m.audit(altered, terminal, report, receipt)
