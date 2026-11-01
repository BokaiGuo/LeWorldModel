import copy
import json
import pytest

from experiments.spectral_reachability import codex_p1_cp5_twenty_physical_20261030_v1 as run
from experiments.spectral_reachability import codex_p1_cp5_twenty_physical_saved_20261030_v1 as saved


def test_actual_twenty_planner_ledger_and_roster():
    receipt, manifest, events = run.planner_sources()
    anchors = json.loads(run.h10.INPUT.read_bytes())["anchors"]
    blind = json.loads(run.old.plans.run.BUNDLE.read_bytes())["rows"]
    roster = run.old.make_roster(anchors, blind, events, receipt)
    assert len(roster) == 24 and manifest["optimizer"]["steps"] == 20
    assert all(len(bytes.fromhex(item["raw10_float32_hex"])) == 80 for item in roster)
    assert all(e["result"]["ledger_receipt"]["planning"]["charged_upper_bounds"]["predict_calls"] == 42
               for e in events)
    bad = copy.deepcopy(events)
    bad[0]["index"] = 1
    with pytest.raises(ValueError, match="identity"):
        run.old.make_roster(anchors, blind, bad, receipt)


def test_terminal_at_horizon_is_not_early_stop():
    def response(step, stop):
        return {"step": step, "state": [0.]*7, "terminated": stop, "truncated": False}
    report = {"row": {"status": "native_stopped",
                      "responses": [response(i, i == 10) for i in range(1, 11)]}}
    result = saved.endpoint(report, [0.]*7)
    assert result["native_stop_at_horizon"] and not result["early_native_stop"]
    assert result["horizon_endpoint_state"] == [0.]*7
    report["row"]["responses"] = [response(i, i == 9) for i in range(1, 10)]
    result = saved.endpoint(report, [0.]*7)
    assert result["early_native_stop"] and result["horizon_endpoint_state"] is None
    report["row"]["status"] = "censored_out_of_domain"
    assert saved.endpoint(report, [0.]*7)["horizon_endpoint_state"] is None


def test_no_overwrite_or_missing_source(tmp_path):
    with pytest.raises(ValueError):
        run.prepare(tmp_path/"wrong")
    with pytest.raises(ValueError, match="source SHA"):
        run.read(tmp_path/"missing", "0"*64)
