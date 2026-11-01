from copy import deepcopy
import pytest
from experiments.spectral_reachability import codex_p3_wallmatch_physical_20261030_v1 as b
from experiments.spectral_reachability import codex_p3_six_physical_contract_20261030_v1 as spec


def test_fixed_roster_and_failed_slot_rejected():
    rows = [dict(point=p, arm=a, delivered_and_saved=True,
                 attempt=dict(exit_code=0, deadline_exceeded=False, wall_seconds=29.))
            for p, a in b.batch.SLOTS]
    report = dict(contract_sha256=b.CONTRACT_SHA, error=None, scientific_pass=False, rows=rows)
    events = [dict(kind="slot_result", row=deepcopy(r)) for r in rows]
    events.append(dict(kind="batch_terminal", completed_slots=6, failure=None))
    assert b.check_batch(report, events) == rows
    bad = deepcopy(report)
    bad["rows"][0]["delivered_and_saved"] = False
    with pytest.raises(ValueError):
        b.check_batch(bad, events)


def test_native_ack_stop_and_tamper():
    commands = [[0.25, -0.5]] * 10
    raw = spec.raw(commands)
    events = [dict(step=i, sent_float32_hex=raw[i].tobytes().hex(),
                   actual_float32_hex=raw[i].tobytes().hex(), terminated=i == 2,
                   truncated=False) for i in range(3)]
    result = spec.verify_ack({"commands": commands}, events,
                             {"status": "native_stopped", "returned_steps": 3})
    assert result["missing_steps"] == 7 and not result["padded"]
    bad = deepcopy(events)
    bad[0]["actual_float32_hex"] = "00" * 8
    with pytest.raises(ValueError):
        spec.verify_ack({"commands": commands}, bad,
                        {"status": "native_stopped", "returned_steps": 3})
