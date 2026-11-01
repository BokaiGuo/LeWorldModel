import copy

import pytest

from experiments.spectral_reachability import p1_uphill_feedback_compare_20261031_main as c


def inputs():
    p = c.open_run.paired
    anchors = p.read(p.ANCHORS, p.ANCHORS_SHA)
    opened = p.read(p.CONSUME, p.CONSUME_SHA)
    feedback = p.read(c.FB_SAVED, c.FB_SAVED_SHA)
    old_report = p.read(c.open_run.RUN / "run_report.json", c.open_run.REPORT_SHA)
    new_report = p.read(c.FB_DIR / "run_report.json", c.FB_REPORT)
    return anchors, opened, feedback, old_report, new_report


def transitions(i, a, b):
    return (c.traces.transitions(c.open_run.RUN / f"row_{i:02d}", a["report_sha256"]),
            c.traces.transitions(c.FB_DIR / f"row_{i:02d}", b["report_sha256"]))


def test_full_same_first_five_and_censored_endpoint():
    result = c.compare(*inputs(), transitions)
    assert len(result["rows"]) == 24
    assert all(row["first5_exact"] for row in result["rows"])
    assert result["rows"][1]["open_last_success"]
    assert result["rows"][1]["open_endpoint"] is None
    assert result["rows"][1]["feedback_minus_open"] is None


def test_rejects_first_packet_state_drift():
    def altered(i, a, b):
        x, y = transitions(i, a, b)
        if i == 0:
            y = copy.deepcopy(y)
            y[0]["actual_bits"] = "wrong"
        return x, y

    with pytest.raises(Exception, match="first five differ"):
        c.compare(*inputs(), altered)
