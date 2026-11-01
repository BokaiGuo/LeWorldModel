from fractions import Fraction

import pytest

from experiments.spectral_reachability import (
    p3_c3_dynamic_budget_20261102_exclusive as subject,
)


def test_exact_allocator_preserves_total_and_reverses_strategy():
    assert subject.allocate(20, (Fraction(3), Fraction(1))) == [15, 5]
    assert subject.allocate(30, (Fraction(3), Fraction(1))) == [23, 7]
    assert subject.allocate(20, (Fraction(1), Fraction(3))) == [5, 15]
    assert subject.allocate(30, (Fraction(1), Fraction(3))) == [8, 22]


def test_protocol_freezes_all_points_before_outcomes():
    value = subject.protocol()
    assert [(row["budget_point"], row["signal"], row["reverse"]) for row in value["points"]] == [
        (20, [15, 5], [5, 15]), (30, [22, 8], [8, 22])
    ]
    assert value["selection_after_outcome"] is False
    assert value["gate_modified"] is False
    assert value["P3_admission"] is value["C3_admission"] is False


def test_call_ledger_classifies_and_separates_attempts_from_returns():
    caps = {name: 1 for name in subject.CALL_CLASSES}
    ledger = subject.CallLedger(caps)
    assert ledger.dispatch("predict", lambda: "ok") == "ok"
    result = ledger.close()
    assert result["attempted"]["predict"] == result["returned"]["predict"] == 1
    assert result["complete"] is True
    assert ledger.events[0]["category"] == "predict"


def test_failed_call_is_charged_and_no_retry():
    caps = {name: 1 for name in subject.CALL_CLASSES}
    ledger = subject.CallLedger(caps)
    with pytest.raises(RuntimeError):
        ledger.dispatch("predict", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert ledger.attempted["predict"] == 1
    assert ledger.returned["predict"] == 0
    with pytest.raises(ValueError, match="ledger closed"):
        ledger.dispatch("predict", lambda: None)


def test_invalid_budget_and_weights_fail_closed():
    with pytest.raises(ValueError):
        subject.allocate(1, (0, 1))
    with pytest.raises(ValueError):
        subject.BudgetPanel((20,), (1,))
    with pytest.raises(ValueError):
        subject.CallLedger({"predict": 1})


def test_equal_axis_comparison_fails_closed_on_missing_and_incomplete():
    vector = {name: 0 for name in subject.CALL_CLASSES}
    signal = {"allocation": [15, 5], "attempted": vector.copy(), "returned": vector.copy()}
    reverse = {"allocation": [5, 15], "attempted": vector.copy(), "returned": vector.copy()}
    assert subject.compare(20, signal, reverse)["equal_recorded_axes"] is True
    signal["attempted"]["predict"] = 1
    result = subject.compare(20, signal, reverse)
    assert result["both_complete"] is result["equal_recorded_axes"] is False
    del reverse["attempted"]["predict"]
    with pytest.raises(ValueError, match="resource vector"):
        subject.compare(20, signal, reverse)


def test_ties_and_collapsed_schedules():
    assert subject.allocate(3, (1, 1)) == [2, 1]
    assert subject.allocate(3, (1, 1), (1, 0)) == [1, 2]
    with pytest.raises(ValueError, match="collapsed"):
        subject.BudgetPanel((20,), (1, 1)).freeze()
