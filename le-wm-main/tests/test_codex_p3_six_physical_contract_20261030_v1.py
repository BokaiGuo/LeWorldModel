import hashlib
import numpy as np
import pytest
from experiments.spectral_reachability import codex_p3_six_physical_contract_20261030_v1 as c


def make():
    a = np.arange(20, dtype=np.float32).reshape(10, 2) / np.float32(32)
    plan = {"commands": a.astype(float).tolist(), "raw_float32_sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    events = [{"step": i, "sent_float32_hex": a[i].tobytes().hex(),
               "actual_float32_hex": a[i].tobytes().hex(), "terminated": False, "truncated": False}
              for i in range(10)]
    return plan, events


def test_full_ack_and_stopped_prefix():
    plan, events = make()
    assert c.verify_ack(plan, events, {"status": "horizon_completed", "returned_steps": 10})["missing_steps"] == 0
    events[3]["truncated"] = True
    result = c.verify_ack(plan, events[:4], {"status": "native_stopped", "returned_steps": 4})
    assert result["missing_steps"] == 6 and result["padded"] is False
    with pytest.raises(ValueError, match="ACK sequence"):
        c.verify_ack(plan, events, {"status": "native_stopped", "returned_steps": 10})


def test_bits_and_lengths_fail_closed():
    plan, events = make()
    events[0]["actual_float32_hex"] = "0" * 16
    with pytest.raises(ValueError, match="bit mismatch"):
        c.verify_ack(plan, events, {"status": "horizon_completed", "returned_steps": 10})
    with pytest.raises(ValueError):
        c.verify_ack(plan, [], {"status": "horizon_completed", "returned_steps": 10})


def test_six_roster_and_caps():
    plan, _ = make()
    rows = [dict(plan, root=root, policy=policy, physical_authorized=False) for root, policy in c.ORDER]
    assert c.validate_plan(rows)
    rows[0]["commands"][0][0] = 0.5
    with pytest.raises(ValueError, match="bits"):
        c.validate_plan(rows)
    assert c.contract()["caps"]["public_step"] == 120
