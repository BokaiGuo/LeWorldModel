import copy

import pytest

from experiments.spectral_reachability import p1_uphill_midpoint_source_compare_20261031_main as c


def inputs():
    p = c.prior
    read = p.open_run.paired.read
    return (read(p.open_run.paired.ANCHORS, p.open_run.paired.ANCHORS_SHA),
            read(p.FB_SAVED, p.FB_SAVED_SHA), read(c.IM_SAVED, c.IM_SAVED_SHA),
            read(p.FB_DIR / "run_report.json", p.FB_REPORT),
            read(c.IM_DIR / "run_report.json", c.IM_REPORT))


def transitions(i, a, b):
    return (c.prior.traces.transitions(c.prior.FB_DIR / f"row_{i:02d}", a["report_sha256"]),
            c.prior.traces.transitions(c.IM_DIR / f"row_{i:02d}", b["report_sha256"]))


def test_all_pairs_and_censoring():
    result = c.compare(*inputs(), transitions)
    assert len(result["rows"]) == 24
    assert all(row["first5_exact"] for row in result["rows"])
    assert all(row["imagined_minus_real"] is None for row in result["rows"]
               if row["real_endpoint"] is None or row["imagined_endpoint"] is None)
    assert not result["P1_admission"]


def test_rejects_mutated_first_step():
    def altered(i, a, b):
        x, y = transitions(i, a, b)
        if i == 0:
            y = copy.deepcopy(y)
            y[0]["state"] = "bad"
        return x, y

    with pytest.raises(Exception, match="first five differ"):
        c.compare(*inputs(), altered)
