from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p3_earlystop_physical_20261030_v1 as bridge
from experiments.spectral_reachability import codex_p3_six_physical_contract_20261030_v1 as spec


def test_fixed_six_and_no_distinct_allocation_requirement():
    rows = [dict(point=p, arm=a, delivered_and_saved=True, allocation=[10, 10],
                 attempt=dict(exit_code=0, deadline_exceeded=False, wall_seconds=29.))
            for p, a in bridge.batch.SLOTS]
    report = dict(contract_sha256=bridge.CONTRACT_SHA, error=None,
                  scientific_pass=False, rows=rows)
    events = [dict(kind="slot_result", row=deepcopy(r)) for r in rows]
    events.append(dict(kind="batch_terminal", completed_slots=6, failure=None))
    assert bridge.check_batch(report, events) == rows
    for kind in ("missing", "journal", "timeout"):
        r, e = deepcopy(report), deepcopy(events)
        if kind == "missing":
            r["rows"].pop()
        elif kind == "journal":
            e[0]["row"]["arm"] = "reverse"
        else:
            r["rows"][0]["attempt"]["deadline_exceeded"] = True
            e[0]["row"] = r["rows"][0]
        with pytest.raises(ValueError):
            bridge.check_batch(r, e)


def test_native_ack_censors_and_rejects_wrong_action():
    commands = [[0.25, -0.5]] * 10
    raw = spec.raw(commands)
    events = [dict(step=i, sent_float32_hex=raw[i].tobytes().hex(),
                   actual_float32_hex=raw[i].tobytes().hex(),
                   terminated=i == 2, truncated=False) for i in range(3)]
    result = spec.verify_ack({"commands": commands}, events,
                             {"status": "native_stopped", "returned_steps": 3})
    assert result["missing_steps"] == 7 and not result["padded"]
    bad = deepcopy(events)
    bad[1]["actual_float32_hex"] = "00" * 8
    with pytest.raises(ValueError):
        spec.verify_ack({"commands": commands}, bad,
                        {"status": "native_stopped", "returned_steps": 3})
    with pytest.raises(ValueError):
        spec.verify_ack({"commands": commands}, events + [events[-1]],
                        {"status": "native_stopped", "returned_steps": 4})


def test_exclusive_rejects_foreign_directory(tmp_path):
    with pytest.raises(ValueError):
        bridge.exclusive(tmp_path / (bridge.STEM + "_P20_RUN_01"), "RUN_", 20)
