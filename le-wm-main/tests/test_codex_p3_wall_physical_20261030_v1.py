from copy import deepcopy
import pytest

from experiments.spectral_reachability import codex_p3_wall_physical_20261030_v1 as bridge


def test_batch_rejects_missing_timeout_and_journal_tamper():
    rows = [{"point": p, "arm": a, "delivered_and_saved": True,
             "attempt": {"exit_code": 0, "deadline_exceeded": False, "wall_seconds": 29.0}}
            for p, a in bridge.batch.SLOTS]
    report = {"contract_sha256": bridge.CONTRACT_SHA, "error": None,
              "scientific_pass": False, "rows": rows}
    events = [{"kind": "slot_result", "row": deepcopy(r)} for r in rows]
    events.append({"kind": "batch_terminal", "completed_slots": 6, "failure": None})
    assert bridge.check_batch(report, events) == rows
    for kind in ("timeout", "missing", "journal"):
        r, e = deepcopy(report), deepcopy(events)
        if kind == "timeout":
            r["rows"][0]["attempt"]["wall_seconds"] = 30.1
            e[0]["row"] = r["rows"][0]
        elif kind == "missing":
            r["rows"].pop()
        else:
            e[0]["row"]["arm"] = "reverse"
        with pytest.raises(ValueError):
            bridge.check_batch(r, e)


def test_exclusive_rejects_other_directory(tmp_path):
    with pytest.raises(ValueError):
        bridge.exclusive(tmp_path / (bridge.STEM + "_P20_RUN_01"), "RUN_", 20)
