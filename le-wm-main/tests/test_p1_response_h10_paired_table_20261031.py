import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_response_h10_paired_table_20261031 as table


def fixture():
    path = Path(__file__).resolve().parents[1] / "results/p1_response_prefix_h10_20261031_prepare_RUN01.json"
    prepared = json.loads(path.read_text())
    rows = []
    for anchor in prepared["anchors"]:
        state = anchor["checkpoint"]["state"]
        rows.append({"anchor": anchor["index"], "statuses": ["probe_completed"] * 42,
                     "endpoints": [state] * 42})
    return prepared, {"status": "saved_checks_passed", "fixed_branches": 588, "rows": rows}


def test_table_retains_censoring_and_prefix_metrics():
    prepared, consumed = fixture()
    consumed["rows"][0]["statuses"][1] = "native_stopped"
    consumed["rows"][0]["endpoints"][1] = None
    result = table.build(prepared, consumed)
    assert len(result["rows"]) == 588
    assert result["censored_before_cp5"] == [1, 5]
    assert result["rows"][1]["endpoint_metrics"] is None
    assert result["rows"][1]["endpoint_minus_own_zero_suffix"] is None
    assert result["rows"][0]["endpoint_minus_own_zero_suffix"]["object_position_error"] == 0
    assert "pusher_position_error" in result["rows"][0]["prefix_minus_root"]
    assert len(result["candidate_comparisons"]) == 6 * 42
    assert result["candidate_comparisons"][0]["model_selected_minus_incumbent_response_vs_own_zero"][
        "object_position_error"] == 0
    assert result["P1_admission"] is False


def test_rejects_fabricated_stopped_endpoint():
    prepared, consumed = fixture()
    consumed["rows"][0]["statuses"][1] = "native_stopped"
    with pytest.raises(Exception):
        table.build(prepared, consumed)


def test_rejects_mismatched_root_in_candidate_pair():
    prepared, consumed = fixture()
    prepared["anchors"][2]["root_state"][0] += 1
    with pytest.raises(Exception):
        table.build(prepared, consumed)


def test_attaches_separate_controls_and_planned_outcomes():
    prepared, consumed = fixture()
    result = table.build(prepared, consumed)
    old = json.loads(table.control.OLD.read_text())
    pairing = json.loads(table.control.PAIR.read_text())
    planned = json.loads(table.PLANNED.read_text())
    table.attach_controls(result, prepared, old, pairing, planned)
    first = result["rows"][0]
    assert set(first["control_comparisons"]) == {"zero", "greedy"}
    assert first["independently_optimized_suffix_status"] == planned["rows"][0]["status"]
    assert "independently_optimized_suffix" in first["control_comparisons"]["zero"]
    assert first["independently_optimized_suffix_endpoint_metrics"] == table.physical_metrics(
        planned["rows"][0]["completed_endpoint"], prepared["anchors"][0]["goal"])
    assert first["control_comparisons"]["greedy"]["independently_optimized_endpoint_minus_control"] is not None
    assert result["equal_budget_certified"] is False


def test_censored_independent_plan_has_no_comparable_endpoint():
    prepared, consumed = fixture()
    result = table.build(prepared, consumed)
    old = json.loads(table.control.OLD.read_text())
    pairing = json.loads(table.control.PAIR.read_text())
    planned = json.loads(table.PLANNED.read_text())
    planned["rows"][0]["status"] = "censored_out_of_domain"
    planned["rows"][0]["completed_endpoint"] = None
    table.attach_controls(result, prepared, old, pairing, planned)
    row = result["rows"][0]
    assert row["independently_optimized_suffix_endpoint_metrics"] is None
    assert row["control_comparisons"]["greedy"]["independently_optimized_endpoint_minus_control"] is None


def test_real_planning_contract_and_mismatched_initialization():
    prepared, _ = fixture()
    assert table.sha(table.NEW_PLAN) == table.NEW_PLAN_SHA
    assert table.sha(table.OLD_PLAN) == table.OLD_PLAN_SHA
    new_plan = json.loads(table.NEW_PLAN.read_text())
    old_plan = json.loads(table.OLD_PLAN.read_text())
    table.verify_planning_protocol(prepared, new_plan, old_plan)
    old_plan["rows"][1]["initial_f32_sha256"] = "wrong"
    with pytest.raises(Exception, match="matched initialization"):
        table.verify_planning_protocol(prepared, new_plan, old_plan)


def test_stops_preserve_last_accepted_without_fabricating_endpoint():
    prepared, consumed = fixture()
    consumed["rows"][0]["statuses"][1:3] = ["native_stopped", "censored_out_of_domain"]
    consumed["rows"][0]["endpoints"][1:3] = [None, None]
    value = table.build(prepared, consumed)
    anchor = prepared["anchors"][0]
    state = anchor["checkpoint"]["state"]
    response = lambda step, terminated=False: {
        "step": step, "state": state, "terminated": terminated, "truncated": False}
    branches = [{"index": i, "status": "probe_completed",
                 "responses": [response(n) for n in range(1, 11)]} for i in range(42)]
    branches[1] = {"index": 1, "status": "native_stopped",
                   "responses": [response(1), response(2, True)]}
    branches[2] = {"index": 2, "status": "censored_out_of_domain", "responses": []}
    table.attach_verified_stops(value, anchor, {"rows": branches})
    assert value["rows"][1]["endpoint_metrics"] is None
    assert value["rows"][1]["accepted_suffix_steps"] == 2
    assert value["rows"][1]["last_accepted_terminated"] is True
    assert value["rows"][1]["last_accepted_metrics"] == table.physical_metrics(state, anchor["goal"])
    assert value["rows"][1]["last_accepted_is_fixed_horizon_endpoint"] is False
    assert value["rows"][2]["last_accepted_metrics"] is None
    branches[1]["responses"][-1]["terminated"] = False
    with pytest.raises(Exception, match="native stop flags"):
        table.attach_verified_stops(value, anchor, {"rows": branches})
