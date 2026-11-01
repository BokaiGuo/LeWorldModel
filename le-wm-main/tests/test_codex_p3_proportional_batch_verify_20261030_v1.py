import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p3_proportional_batch_verify_20261030_v1 as audit

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"results/codex_p3_proportional_delivery_20261030_v1_RUN_CODEX01"
REPORT_SHA = "c5bf4434a7bcb53e841eb9c76f95572822a1648af7c0b5c586cf432868228463"


def test_real_batch_bindings_and_denominator():
    result = audit.build(RUN, REPORT_SHA)
    assert result["status"] == "batch_bindings_checked"
    assert len(result["rows"]) == 6
    assert result["rows"][1]["allocation"] == [10, 10]
    assert result["rows"][4]["allocation"] == [15, 15]
    assert not result["scientific_pass"]


def test_event_reordering_and_missing_receipt_fail():
    report = json.loads((RUN/"report.json").read_bytes())
    events = audit.delivery.host.journal.read(RUN/"events.jsonl", report["journal_tail"], 2*1024**2)
    events = copy.deepcopy(events)
    events[1], events[3] = events[3], events[1]
    with pytest.raises(ValueError, match="slot"):
        audit.verify_rows(report, events, lambda _: {})
    events[1], events[3] = events[3], events[1]
    with pytest.raises(ValueError, match="receipt status"):
        audit.verify_rows(report, events, lambda _: {})
