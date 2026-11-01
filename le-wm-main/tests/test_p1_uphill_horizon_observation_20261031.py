import copy
import json

import pytest

from experiments.spectral_reachability import p1_uphill_horizon_observation_20261031 as audit


def inputs():
    pairs = audit.paired.read(audit.rgb.PAIR, audit.rgb.PAIR_SHA)
    top = audit.paired.read(audit.rgb.RUN / "run_report.json", audit.rgb.REPORT_SHA)
    anchors = audit.paired.read(audit.paired.ANCHORS, audit.paired.ANCHORS_SHA)
    reports = [audit.paired.read(audit.rgb.RUN / f"row_{s['index']:02d}" / "report.json",
                                 s["report_sha256"]) for s in top["shards"]]
    return pairs["rows"], top["shards"], anchors["anchors"], reports


def test_terminal_at_horizon_is_observed_without_imputing_earlier_stops():
    rows = audit.extract(*inputs())
    assert len(rows) == 24
    assert sum(r["h10_observed"] for r in rows) == 22
    assert [r["index"] for r in rows if not r["h10_observed"]] == [1, 4]
    assert [r["index"] for r in rows if r["status"] == "native_stopped"
            and r["h10_observed"]] == [7, 19, 22]
    assert all(rows[i]["h10_metrics"]["native_success"] for i in (7, 19, 22))


def test_rejects_mismatched_terminal_frame_and_early_terminal():
    rows, shards, anchors, reports = inputs()
    reports = copy.deepcopy(reports)
    frame = next(f for f in reports[7]["row"]["trajectory"]
                 if f["phase"] == "probe" and f["step"] == 10 and f["overlay_index"] == 1)
    frame["state"][0] += 1
    with pytest.raises(ValueError):
        audit.extract(rows, shards, anchors, reports)
    rows, shards, anchors, reports = inputs()
    reports = copy.deepcopy(reports)
    reports[7]["row"]["responses"][0]["terminated"] = True
    with pytest.raises(ValueError):
        audit.extract(rows, shards, anchors, reports)
