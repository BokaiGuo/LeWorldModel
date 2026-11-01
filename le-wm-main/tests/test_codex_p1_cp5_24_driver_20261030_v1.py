import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_cp5_24_driver_20261030_v1 as driver


def bundle():
    path = Path(__file__).resolve().parents[1]/"results/codex_p1_three_prefix_blind_inputs_20261030_v1.json"
    return json.loads(path.read_text())


def test_24_fixed_order_and_no_extra_rows():
    events = []
    def run(row):
        return {"index": row["index"], "actions": [], "score": 0, "ledger_receipt": {}}
    result = driver.drive(bundle(), run, events.append)
    assert [x["index"] for x in result] == list(range(24))
    assert [e["kind"] for e in events] == ["row_start", "row_complete"]*24


def test_bad_roster_rejected_before_callbacks():
    b = bundle()
    b["rows"][0], b["rows"][1] = b["rows"][1], b["rows"][0]
    calls = []
    with pytest.raises(ValueError):
        driver.drive(b, calls.append, calls.append)
    assert not calls


def test_failure_stops_and_preserves_event():
    events, calls = [], []
    def run(row):
        calls.append(row["index"])
        if row["index"] == 3:
            raise RuntimeError("synthetic")
        return {"index": row["index"], "actions": [], "score": 0, "ledger_receipt": {}}
    with pytest.raises(RuntimeError, match="synthetic"):
        driver.drive(bundle(), run, events.append)
    assert calls == [0, 1, 2, 3]
    assert events[-1]["kind"] == "row_failure" and events[-1]["index"] == 3
