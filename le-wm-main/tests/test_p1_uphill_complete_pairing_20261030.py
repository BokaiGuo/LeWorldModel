import copy

import pytest

from experiments.spectral_reachability import p1_uphill_complete_pairing_20261030 as pairing
from experiments.spectral_reachability import codex_p1_h10_paired_table_20261030_main as metric


def inputs():
    prepared = pairing.read_checked(pairing.PREP, pairing.PREP_SHA)
    zero = pairing.read_checked(pairing.ZERO, pairing.ZERO_SHA)
    warm = pairing.read_checked(pairing.WARM, pairing.WARM_SHA)
    rows = []
    for source in prepared["anchors"]:
        root = metric.physical_metrics(source["root_state"], source["goal"])
        prefix = metric.physical_metrics(source["checkpoint"]["state"], source["goal"])
        for probe in prepared["probe_banks"][source["logical_pair"]]:
            row = {key: source[key] for key in pairing.IDENTITY}
            row.update(probe=probe["name"], probe_sha256=probe["raw10_sha256"],
                       root=root, prefix=prefix, status="native_stopped",
                       returned_suffix_steps=5, endpoint=None, endpoint_minus_own_zero=None,
                       last_accepted_metrics=prefix, last_accepted_terminated=True,
                       last_accepted_truncated=False)
            rows.append(row)
    panel = {"schema": "p1_uphill_h10_consume_20261030", "prepare_sha256": pairing.PREP_SHA,
             "P1_admission": False, "rows": rows}
    multistart = pairing.read_checked(pairing.MULTISTART, pairing.MULTISTART_SHA)
    return prepared, zero, warm, panel, multistart, pairing.read_checked(pairing.TRIPLET, pairing.TRIPLET_SHA)


def test_roster_and_censoring():
    prepared, zero, warm, panel, multi, triplet = inputs()
    result = pairing.join(prepared, zero, warm, panel, multi, triplet)
    assert len(result["anchors"]) == 24
    assert len(result["common_suffix_pairs"]) == 336
    assert result["common_suffix_pairs"][0]["roles"]["selected"]["endpoint_minus_zero_prefix"] is None
    assert not result["P1_admission"]


def test_rejects_identity_and_censoring_mismatch():
    prepared, zero, warm, panel, multi, triplet = inputs()
    bad = copy.deepcopy(panel)
    bad["rows"][0]["probe_sha256"] = "bad"
    with pytest.raises(Exception):
        pairing.join(prepared, zero, warm, bad, multi, triplet)


def test_endpoint_and_own_zero_response_are_distinct():
    prepared, zero, warm, panel, multi, triplet = inputs()
    for row in panel["rows"]:
        if row["logical_pair"] != 0 or row["probe"] != prepared["probe_banks"][0][1]["name"]:
            continue
        offset = {"selected": 3.0, "zero": 1.0, "greedy": 2.0}[row["role"]]
        row["status"] = "probe_completed"
        row["returned_suffix_steps"] = 10
        row["endpoint"] = {key: offset + i for i, key in enumerate(pairing.FIELDS)}
        row["endpoint_minus_own_zero"] = {key: offset * 2 + i for i, key in enumerate(pairing.FIELDS)}
    result = pairing.join(prepared, zero, warm, panel, multi, triplet)
    roles = result["common_suffix_pairs"][1]["roles"]
    assert roles["selected"]["endpoint_minus_zero_prefix"]["object_position_error"] == 2.0
    assert roles["selected"]["response_minus_zero_prefix"]["object_position_error"] == 4.0
    assert roles["greedy"]["endpoint_minus_zero_prefix"]["object_position_error"] == 1.0
    assert roles["selected"]["returned_suffix_steps"] == 10


def test_rejects_warm_action_identity_drift():
    prepared, zero, warm, panel, multi, triplet = inputs()
    warm = copy.deepcopy(warm)
    warm["rows"][1]["prefix_action_sha256"] = "wrong"
    with pytest.raises(Exception):
        pairing.join(prepared, zero, warm, panel, multi, triplet)
    bad = copy.deepcopy(panel)
    bad["rows"][0]["status"] = "probe_completed"
    with pytest.raises(Exception):
        pairing.join(prepared, zero, warm, bad, multi, triplet)


def test_rejects_corrupted_prefix_difference_in_either_plan():
    prepared, zero, warm, panel, multi, triplet = inputs()
    for source in (zero, warm):
        corrupted = copy.deepcopy(source)
        corrupted["rows"][0]["prefix_minus_root"]["pusher_position_error"] += 1.0
        with pytest.raises(Exception, match="prefix metric difference"):
            pairing.join(prepared, corrupted if source is zero else zero,
                         corrupted if source is warm else warm, panel, multi, triplet)


def test_rejects_multistart_identity_or_censor_mismatch():
    prepared, zero, warm, panel, multi, triplet = inputs()
    bad = copy.deepcopy(multi)
    bad["rows"][0]["task_id"] = "different"
    with pytest.raises(Exception, match="anchor identity"):
        pairing.join(prepared, zero, warm, panel, bad, triplet)
    bad = copy.deepcopy(multi)
    bad["rows"][0]["status"] = "native_stopped"
    with pytest.raises(Exception, match="multistart endpoint censoring"):
        pairing.join(prepared, zero, warm, panel, bad, triplet)


def test_rejects_midpoint_misbind_and_false_endpoint():
    prepared, zero, warm, panel, multi, triplet = inputs()
    bad = copy.deepcopy(triplet)
    bad["groups"][0]["task_id"] = "different"
    with pytest.raises(Exception, match="triplet identity"):
        pairing.join(prepared, zero, warm, panel, multi, bad)
    bad = copy.deepcopy(triplet)
    item = bad["groups"][0]["prefixes"][0]["real"]
    item["endpoint"] = None if item["status"] == "probe_completed" else {"invalid": True}
    with pytest.raises(Exception, match="triplet feedback censoring"):
        pairing.join(prepared, zero, warm, panel, multi, bad)
