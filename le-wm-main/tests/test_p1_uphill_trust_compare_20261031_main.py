import copy

import pytest

from experiments.spectral_reachability import p1_uphill_trust_compare_20261031_main as c

REPORT = "2b839f8cd7663aef0a4ab08058ed12644cda950ff1ee0fbf051310d4d7895413"
SAVED = "34254403a3845965b81ecb19edfa7ad750220dd896937348debdbfa10aced495"


def inputs():
    p = c.prior
    read = p.open_run.paired.read
    return (read(p.open_run.paired.ANCHORS, p.open_run.paired.ANCHORS_SHA),
            read(p.FB_SAVED, p.FB_SAVED_SHA), read(c.SAVED, SAVED),
            read(p.FB_DIR / "run_report.json", p.FB_REPORT),
            read(c.RUN / "run_report.json", REPORT), read(c.trust.PLAN, c.trust.PLAN_SHA))


def test_complete_comparison_and_reject_action_mismatch():
    anchors, reference, actual, rr, ar, plans = inputs()

    def action(i):
        return c.prior.traces.transitions(c.RUN / f"row_{i:02d}", ar["shards"][i]["report_sha256"])

    def transitions(i, a, b):
        return (c.prior.traces.transitions(c.prior.FB_DIR / f"row_{i:02d}", a["report_sha256"]), action(i))

    result = c.compare(anchors, reference, actual, rr, ar, transitions, plans, action)
    assert len(result["rows"]) == 24
    assert not result["P1_admission"]
    assert all(row["feedback_minus_open"] is None for row in result["rows"]
               if row["open_endpoint"] is None or row["feedback_endpoint"] is None)

    def tampered(i):
        events = copy.deepcopy(action(i))
        if i == 0:
            events[5]["actual_bits"] = "bad"
        return events

    with pytest.raises(Exception, match="executed action mismatch"):
        c.compare(anchors, reference, actual, rr, ar, transitions, plans, tampered)
