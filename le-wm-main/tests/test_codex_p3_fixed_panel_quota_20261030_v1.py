import pytest

from experiments.spectral_reachability import codex_p3_fixed_panel_quota_20261030_v1 as c


def test_allocation_and_ties():
    assert c.allocation("uniform", [0.7, 0.3]) == (10, 10)
    assert c.allocation("signal", [0.7, 0.3]) == (15, 5)
    assert c.allocation("reverse", [0.7, 0.3]) == (5, 15)
    assert c.allocation("signal", [0.5, 0.5]) == (10, 10)
    for value in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            c.allocation("signal", [value, 0])


def test_equal_work_and_real_eligibility():
    outputs = []
    for policy in c.POLICIES:
        calls, events = [], []
        def evaluate(g, i):
            calls.append((g, i))
            return {"cost": -i if g == 0 else 100 - i, "predict_calls": 2}
        result = c.run(policy, [1, 0], evaluate, events.append)
        assert calls == list(c.PANEL)
        assert result["attempted"] == result["completed"] == c.EXPECTED
        assert len([x for x in result["rows"] if x["eligible"]]) == 20
        assert result["scientific_pass"] is False
        outputs.append(result["winner"]["index"])
    assert outputs == [9, 14, 4]


def test_fail_stop_and_no_retry():
    calls, events = [], []
    def evaluate(g, i):
        calls.append((g, i))
        if i == 3:
            raise RuntimeError("backend failure")
        return {"cost": 0., "predict_calls": 2}
    with pytest.raises(RuntimeError):
        c.run("signal", [1, 0], evaluate, events.append)
    assert len(calls) == 4
    assert events[-1]["completed_candidates"] == 3
    assert events[-1]["inflight_predict_and_metric"] == "unknown"


def test_bad_receipt_fails_closed():
    events = []
    with pytest.raises(ValueError):
        c.run("uniform", [1, 0], lambda g, i: {"cost": 0, "predict_calls": 1}, events.append)
    assert events[-1]["kind"] == "failure"
