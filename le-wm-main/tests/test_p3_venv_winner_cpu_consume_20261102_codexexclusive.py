"""Regression checks against saved CPU records, with deliberately corrupted copies."""
import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p3_venv_winner_cpu_consume_20261102_codexexclusive as consumer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUN = RESULTS / "p3_venv_winner_cpu_physical_20261102_codexexclusive_RUN_01"
PREP = RESULTS / "p3_venv_winner_cpu_physical_20261102_codexexclusive_PREP_01.json"
SHA = "7d5cb06276b9c1fb2d86344a38f3baf4f67b08a6740a60bc321e3e033033df08"


def test_saved_record_and_prepare_sha():
    assert consumer.consume(PREP, SHA, RUN)["native_successes"] == 0
    with pytest.raises(ValueError, match="prepare SHA"):
        consumer.consume(PREP, "0" * 64, RUN)


@pytest.mark.parametrize("corrupt", ["command", "guard", "metric", "closure"])
def test_reject_corrupted_receipts(corrupt):
    prep = json.loads(PREP.read_text())
    report = json.loads((RUN / "report.json").read_text())
    events = [json.loads(line) for line in (RUN / "journal.jsonl").read_text().splitlines()]
    events = copy.deepcopy(events)
    if corrupt == "command":
        next(e for e in events if e["kind"] == "returned_transition" and e["phase"] == "plan")["command"][0] = 9
    elif corrupt == "guard":
        next(e for e in events if e["kind"] == "guard_receipt")["accepted"] = False
    elif corrupt == "metric":
        report["rows"][0]["final_metrics"]["object_position_error"] = -1
        next(e for e in events if e["kind"] == "branch_closed")["receipt"] = report["rows"][0]
    else:
        next(e for e in events if e["kind"] == "branch_closed")["receipt"]["arm"] = "bad"
    with pytest.raises(ValueError):
        consumer.validate(prep, report, events, RUN)
