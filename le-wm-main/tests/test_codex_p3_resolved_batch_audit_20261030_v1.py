from pathlib import Path
import pytest

from experiments.spectral_reachability import codex_p3_resolved_batch_audit_20261030_v1 as audit
from experiments.spectral_reachability import codex_p3_resolved_physical_20261030_v1 as physical

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"results/codex_p3_resolved_delivery_20261030_v1_RUN_CODEX01"
SHA = "f94e83cbfde9927da4710e982f9129a9b90f6df23ec03dbcf3328c5176bdbdf8"


def test_actual_partial_with_distinct_executed_candidates():
    result = audit.build(RUN, SHA)
    assert result["status"] == "partial_or_failed"
    assert sum(r["delivered_and_saved"] for r in result["rows"]) == 5
    assert result["signal_reverse_candidate_symmetric_difference"] == {"20": [2, 2], "30": [2, 2]}
    assert all(r["scheduler_calls"] == 1 for r in result["rows"] if r["delivered_and_saved"])
    assert not result["scientific_pass"]


def test_report_tamper_refused():
    with pytest.raises(ValueError, match="batch report hash"):
        audit.build(RUN, "0"*64)


def test_actual_partial_is_not_physical_authorization():
    import json
    report = json.loads((RUN/"report.json").read_bytes())
    events = audit.delivery.host.journal.read(RUN/"events.jsonl", report["journal_tail"], 2*1024**2)
    with pytest.raises(ValueError, match="deadline delivery"):
        physical.check_batch(report, events)
