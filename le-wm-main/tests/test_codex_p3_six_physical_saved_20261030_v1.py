import pytest
from experiments.spectral_reachability import codex_p3_six_physical_saved_20261030_v1 as saved
from test_codex_p3_six_physical_contract_20261030_v1 import make


def test_saved_ack_tamper():
    plan, events = make()
    rows = [dict(ordinal=i, sent_dtype="float32", actual_dtype="float32", sent_shape=[2], actual_shape=[2],
                 sent_hex=e["sent_float32_hex"], actual_hex=e["actual_float32_hex"],
                 terminated=False, truncated=False) for i, e in enumerate(events)]
    result = saved.verify_slot(plan, rows, {"status": "horizon_completed", "returned_steps": 10})
    assert result["source_authenticated"] is False
    rows[3]["actual_hex"] = "0" * 16
    with pytest.raises(ValueError, match="bit mismatch"):
        saved.verify_slot(plan, rows, {"status": "horizon_completed", "returned_steps": 10})
