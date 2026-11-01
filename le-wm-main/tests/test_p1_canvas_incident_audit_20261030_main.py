from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_canvas_incident_audit_20261030_main as audit


@pytest.fixture(scope="module")
def saved():
    read = audit.experiment.runner.physical.native.read
    report = read(audit.DIRECTORY / "report.json", cap=64 * 1024**2)
    prepared = read(audit.PREPARED, cap=32 * 1024**2)
    events = audit.prior.events(audit.DIRECTORY / "journal.jsonl", report["journal_chain_tail"])
    return report, events, prepared


def test_actual_incident(saved):
    result = audit.audit(*saved)
    assert result["state"][1] > 512
    assert result["native_terminated"] is False
    assert result["native_truncated"] is False


@pytest.mark.parametrize("field", ["action", "state", "order", "error"])
def test_reject_changed_incident(saved, field):
    report, events, prepared = deepcopy(saved)
    step = next(e for e in events if e.get("branch") == 16 and
                e["kind"] == "returned_transition" and e["phase"] == "probe" and e["step"] == 5)
    if field == "action":
        step["actual_bits"] = "0000000000000000"
    elif field == "state":
        step["state"][1] = 500.0
    elif field == "order":
        step["index"] = 999999
    else:
        report["errors"][0]["message"] = "different error"
    with pytest.raises(Exception):
        audit.audit(report, events, prepared)
