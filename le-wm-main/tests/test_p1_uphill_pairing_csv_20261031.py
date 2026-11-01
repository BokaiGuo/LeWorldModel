import copy

import pytest

from experiments.spectral_reachability import p1_uphill_complete_pairing_20261030 as pairing
from experiments.spectral_reachability import p1_uphill_pairing_csv_20261031 as table
from test_p1_uphill_complete_pairing_20261030 import inputs


def test_export_keeps_censoring_and_planning_separate():
    result = pairing.join(*inputs())
    rows = table.flatten(result)
    assert len(rows) == 1008
    assert rows[0]["common_endpoint_object_position_error"] is None
    assert rows[0]["common_native_success"] is None
    assert rows[0]["common_last_accepted_native_success"] is not None
    assert rows[0]["common_last_accepted_terminated"] is True
    assert rows[0]["zero_initialization_status"] in ("probe_completed", "native_stopped")
    anchor = result["anchors"][0]
    assert rows[0]["prefix_cp5_pixel_sha256"] == anchor["prefix_cp5_pixel_sha256"]
    assert rows[0]["root_native_success"] == anchor["root_metrics"]["native_success"]
    assert rows[0]["prefix_native_success"] == anchor["prefix_metrics"]["native_success"]
    for metric in pairing.FIELDS:
        assert rows[0][f"root_{metric}"] == anchor["root_metrics"][metric]
        assert rows[0][f"prefix_{metric}"] == anchor["prefix_metrics"][metric]
        assert rows[0][f"prefix_delta_{metric}"] == pytest.approx(
            rows[0][f"prefix_{metric}"] - rows[0][f"root_{metric}"]
        )
    assert rows[0]["prefix_delta_pusher_position_error"] is not None
    assert rows[0]["multistart_status"] == "probe_completed"
    assert rows[0]["multistart_endpoint_object_position_error"] is not None
    assert rows[0]["midpoint_real_status"] in ("probe_completed", "native_stopped")
    assert rows[0]["midpoint_trust_success"] is not None
    stopped = [row for row in rows if row["warm_initialization_status"] == "native_stopped"]
    assert stopped
    assert all(row["warm_initialization_native_success"] is None for row in stopped)
    assert all(row["warm_initialization_last_accepted_native_success"] is not None for row in stopped)


def test_planned_deltas_use_paired_zero_and_preserve_missing_endpoints():
    result = pairing.join(*inputs())
    rows = table.flatten(result)
    anchors = {(a["identity"]["logical_pair"], a["identity"]["role"]): a
               for a in result["anchors"]}
    for row in rows:
        anchor = anchors[(row["logical_pair"], row["role"])]
        zero = anchors[(row["logical_pair"], "zero")]
        for name in ("zero_initialization", "warm_initialization"):
            endpoint = anchor[name]["endpoint"]
            zero_endpoint = zero[name]["endpoint"]
            for metric in pairing.FIELDS:
                actual = row[f"{name}_endpoint_delta_prefix_{metric}"]
                paired = row[f"{name}_endpoint_delta_zero_prefix_{metric}"]
                if endpoint is None:
                    assert actual is None and paired is None
                else:
                    assert actual == pytest.approx(endpoint[metric] - anchor["prefix_metrics"][metric])
                    if zero_endpoint is None:
                        assert paired is None
                    else:
                        assert paired == pytest.approx(endpoint[metric] - zero_endpoint[metric])


def test_rejects_binding_and_admission_corruption():
    result = pairing.join(*inputs())
    bad = copy.deepcopy(result)
    bad["common_suffix_pairs"][0]["roles"]["selected"]["anchor_index"] = 1
    with pytest.raises(Exception, match="anchor binding"):
        table.flatten(bad)
    result["equal_budget_certified"] = True
    with pytest.raises(Exception, match="pairing contract"):
        table.flatten(result)

def test_rejects_duplicate_probes_and_duplicate_anchors():
    result = pairing.join(*inputs())
    bad = copy.deepcopy(result)
    bad["common_suffix_pairs"][1] = copy.deepcopy(bad["common_suffix_pairs"][0])
    with pytest.raises(ValueError, match="unique probe roster"):
        table.flatten(bad)
    bad = copy.deepcopy(result)
    bad["anchors"].append(copy.deepcopy(bad["anchors"][0]))
    with pytest.raises(ValueError, match="complete roster"):
        table.flatten(bad)


def test_multistart_midpoint_binding_and_censoring():
    result = pairing.join(*inputs())
    comparison = pairing.read_checked(pairing.MIDPOINT_MULTI, pairing.MIDPOINT_MULTI_SHA)
    result = pairing.bind_midpoint_multistart(result, comparison)
    rows = table.flatten(result)
    assert rows[0]["midpoint_multistart_status"] == comparison["rows"][0]["feedback_status"]
    stopped = [row for row in rows if row["midpoint_multistart_status"] == "native_stopped"]
    assert stopped and all(row["midpoint_multistart_endpoint_object_position_error"] is None for row in stopped)
    bad = copy.deepcopy(comparison)
    bad["rows"][0]["first5_exact"] = False
    with pytest.raises(Exception, match="midpoint multistart identity"):
        pairing.bind_midpoint_multistart(result, bad)
    bad = copy.deepcopy(comparison)
    bad["rows"][0]["open_last_success"] = not bad["rows"][0]["open_last_success"]
    with pytest.raises(Exception, match="midpoint multistart open binding"):
        pairing.bind_midpoint_multistart(result, bad)


def test_feedback_deltas_are_paired_by_strategy_and_preserve_censoring():
    result = pairing.join(*inputs())
    comparison = pairing.read_checked(pairing.MIDPOINT_MULTI, pairing.MIDPOINT_MULTI_SHA)
    result = pairing.bind_midpoint_multistart(result, comparison)
    rows = table.flatten(result)
    anchors = {(a["identity"]["logical_pair"], a["identity"]["role"]): a
               for a in result["anchors"]}
    for row in rows:
        anchor = anchors[(row["logical_pair"], row["role"])]
        zero = anchors[(row["logical_pair"], "zero")]
        for strategy, outcome in anchor["midpoint_feedback"].items():
            reference = zero["midpoint_feedback"][strategy]["endpoint"]
            for metric in pairing.FIELDS:
                own = row[f"midpoint_{strategy}_endpoint_delta_prefix_{metric}"]
                paired = row[f"midpoint_{strategy}_endpoint_delta_zero_prefix_{metric}"]
                endpoint = outcome["endpoint"]
                if endpoint is None:
                    assert own is None and paired is None
                else:
                    assert own == pytest.approx(endpoint[metric] - anchor["prefix_metrics"][metric])
                    if reference is None:
                        assert paired is None
                    else:
                        assert paired == pytest.approx(endpoint[metric] - reference[metric])
