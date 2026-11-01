import numpy as np
import pytest

from experiments.spectral_reachability import p3_distinct_quota_scheduler_20261102_codex_exclusive as s


class FakeModel:
    def action_encoder(self, value):
        return value

    def predict(self, context, embedded):
        return context


def adapter(point, arm, *, sync=lambda: None, cap=None):
    count = point if cap is None else cap
    return s.boundary.BoundaryAdapter(FakeModel(), point=point, arm=arm, sync=sync,
                                      caps={key: count for key in s.AXES})


def evaluate(meter, root, index):
    meter.predict(np.zeros((1, 2)), np.zeros((1, 2)), phase="search")
    return float(abs(index - (12 if root == 0 else 3)))


@pytest.mark.parametrize("point,allocations", [(20, ([15, 5], [5, 15])),
                                               (30, ([22, 8], [8, 22]))])
def test_distinct_allocations_complete_and_same_recorded_axes(point, allocations):
    signal = s.run_arm(adapter(point, "signal"), evaluate)
    reverse = s.run_arm(adapter(point, "reverse"), evaluate)
    result = s.compare(signal, reverse)
    assert (signal["allocation"], reverse["allocation"]) == allocations
    assert result["both_complete"] and result["equal_measured_model_axes"]
    assert result["policy_distinct"] and not result["equal_total_compute"]
    assert signal["winners"][0]["index"] != reverse["winners"][0]["index"]
    assert not result["P3_admission"] and not result["C3_admission"]


def test_failure_retains_attempt_without_return_and_does_not_pad():
    calls = []
    def sync():
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("sync failed")
    failed = s.run_arm(adapter(20, "signal", sync=sync), evaluate)
    assert failed["status"] == "failed" and failed["winners"] == []
    assert len(failed["candidates"]) == 1
    assert failed["ledger"]["attempted"]["predict"] == 2
    assert failed["ledger"]["returned"]["predict"] == 1
    assert not s.compare(failed, s.run_arm(adapter(20, "reverse"), evaluate))["equal_measured_model_axes"]


def test_bypass_or_nonfinite_score_fails_closed():
    bypass = s.run_arm(adapter(20, "signal"), lambda meter, root, index: 0.0)
    assert bypass["status"] == "failed"
    assert "each measured model axis" in bypass["error"]["message"]
    nonfinite = s.run_arm(adapter(20, "signal"), lambda meter, root, index: (
        evaluate(meter, root, index) * float("nan")))
    assert nonfinite["status"] == "failed" and not nonfinite["winners"]


def test_cap_failure_is_not_silently_equal():
    result = s.run_arm(adapter(20, "signal", cap=2), evaluate)
    assert result["status"] == "failed"
    assert len(result["candidates"]) == 2
    assert "cap exceeded" in result["error"]["message"]


def test_different_caps_are_not_declared_matched():
    signal = s.run_arm(adapter(20, "signal", cap=20), evaluate)
    reverse = s.run_arm(adapter(20, "reverse", cap=21), evaluate)
    assert not s.compare(signal, reverse)["equal_measured_model_axes"]


def test_contract_source_pins_and_gates():
    contract = s.protocol()
    assert contract["complete_ledger_certified"] is False
    assert contract["host_cuda_enforced_here"] is False
    assert contract["gate_modified"] is False
