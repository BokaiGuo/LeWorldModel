import copy

import pytest

from experiments.spectral_reachability import p3_online_pair_contract_20261102_codex_exclusive as c


def slot(arm, weights, point=20):
    return {
        "status": "semantic_saved_checked_not_scientifically_admitted",
        "scientific_pass": False,
        "physical_certified": False,
        "model_arrays_independently_recomputed": False,
        "rows": [{
            "point": point, "arm": arm, "weights": weights,
            "allocation": c.producer.allocation(point, arm, weights),
            "attempted": {axis: 0 for axis in c.producer.AXES},
        }],
    }


def test_signal_changes_routing_and_reverse_is_distinct():
    for weights in ([3.0, 1.0], [1.0, 3.0]):
        result = c.compare_verified_slots(slot("signal", weights), slot("reverse", weights), 20)
        assert result["classification"] == "paired_distinct_allocation"
        assert result["allocations"]["signal"] == result["allocations"]["reverse"][::-1]
        assert result["equal_recorded_axes"]
        assert not result["equal_total_compute"] and not result["P3_admission"]
    assert slot("signal", [3.0, 1.0])["rows"][0]["allocation"] != \
        slot("signal", [1.0, 3.0])["rows"][0]["allocation"]


def test_ties_and_signal_mismatch_are_not_promoted():
    tied = c.compare_verified_slots(slot("signal", [0.0, 0.0]), slot("reverse", [0.0, 0.0]), 20)
    assert tied["classification"] == "degenerate_equal_allocation"
    mismatch = c.compare_verified_slots(slot("signal", [3.0, 1.0]), slot("reverse", [2.0, 1.0]), 20)
    assert mismatch["classification"] == "unpaired_signal_mismatch"


def test_corrupt_slot_or_different_axes():
    a, b = slot("signal", [3.0, 1.0]), slot("reverse", [3.0, 1.0])
    b["rows"][0]["attempted"]["predict"] = 1
    assert not c.compare_verified_slots(a, b, 20)["equal_recorded_axes"]
    corrupt = copy.deepcopy(a)
    corrupt["rows"][0]["allocation"] = [10, 10]
    with pytest.raises(ValueError, match="allocation"):
        c.compare_verified_slots(corrupt, b, 20)
    with pytest.raises(ValueError, match="slot identity"):
        c.compare_verified_slots(b, a, 20)


def test_pair_uses_existing_verifier_for_each_slot(monkeypatch):
    calls = []
    tasks, banks = object(), object()
    def verify(events, got_tasks, got_banks, point, arm):
        calls.append((events, got_tasks, got_banks, point, arm))
        return slot(arm, [3.0, 1.0], point)
    monkeypatch.setattr(c.consumer, "verify_events", verify)
    result = c.verify_pair("signal journal fixture", "reverse journal fixture", tasks, banks, 20)
    assert result["classification"] == "paired_distinct_allocation"
    assert calls == [("signal journal fixture", tasks, banks, 20, "signal"),
                     ("reverse journal fixture", tasks, banks, 20, "reverse")]


def test_verifier_failure_propagates(monkeypatch):
    def reject(*args):
        raise ValueError("invalid event journal")
    monkeypatch.setattr(c.consumer, "verify_events", reject)
    with pytest.raises(ValueError, match="invalid event journal"):
        c.verify_pair([], [], [], [], 20)


def test_real_saved_verifier_rejects_incomplete_journals():
    with pytest.raises(ValueError):
        c.verify_pair([], [], [{}, {}], [list(range(30)), list(range(30))], 20)


def test_preparation_is_not_launch_authority():
    plan = c.preparation_contract()
    assert not plan["host_execution_authorized"]
    assert not plan["physical_release"]
