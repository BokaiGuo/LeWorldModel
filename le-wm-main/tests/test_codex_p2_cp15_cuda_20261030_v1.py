import pytest

from experiments.spectral_reachability import codex_p2_cp15_cuda_20261030_v1 as driver
from experiments.spectral_reachability import codex_p2_cp15_cuda_saved_20261030_v1 as saved


def test_fixed_roster_and_meter_failure_is_not_refunded():
    assert driver.ORDER == [(i, arm) for i in range(4) for arm in ("joint", "suffix")]
    events = []
    meter = driver.Meter(events.append, lambda: 1.0)
    with pytest.raises(RuntimeError):
        meter.call_extra("predictor", lambda: (_ for _ in ()).throw(RuntimeError("failure")))
    assert meter.extra_attempted["predictor"] == 1
    assert meter.extra_returned["predictor"] == 0
    assert events[-1]["kind"] == "failure"


def test_consumer_rejects_missing_slot_evidence():
    row = {"base": 0, "arm": "joint", "status": "complete"}
    with pytest.raises(ValueError, match="slot boundaries"):
        saved.validate_slot(row, [], 0, "joint", {"feedback_context_sha256": "x"})
