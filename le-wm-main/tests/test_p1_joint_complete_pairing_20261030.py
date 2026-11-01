import copy

import pytest

from experiments.spectral_reachability import p1_joint_complete_pairing_20261030 as pairing


def fixed_panel():
    prepared = pairing.read_checked(pairing.PREP_PATH, pairing.PREP_SHA)
    anchors = pairing.read_checked(pairing.ANCHOR_PATH, pairing.ANCHOR_SHA)
    rows = []
    for logical, bank in enumerate(prepared["probe_banks"]):
        for probe in bank:
            for anchor in anchors["rows"][logical * 3:logical * 3 + 3]:
                rows.append({
                    "logical_pair": logical, "probe": probe["name"], "role": anchor["role"],
                    "root_slot": anchor["root_slot"], "condition": anchor["condition"],
                    "arm": anchor["arm"], "probe_sha256": probe["raw10_sha256"],
                    "root": anchor["root_metrics"], "prefix": anchor["prefix_metrics"],
                    "endpoint": anchor["prefix_metrics"], "status": "probe_completed",
                    "endpoint_minus_own_zero": {k: 0.0 for k in pairing.NUMERIC},
                })
    panel = {"schema": "p1_joint_h10_consume_20261030", "prepare_sha256": pairing.PREP_SHA,
             "P1_admission": False, "rows": rows}
    return prepared, anchors, panel


def test_complete_pairing_and_identity_failure():
    prepared, anchors, panel = fixed_panel()
    result = pairing.join(prepared, anchors, panel)
    assert len(result["anchors"]) == 24
    assert len(result["common_suffix_pairs"]) == 336
    assert result["independent_suffix_observations"] == 24
    damaged = copy.deepcopy(panel)
    damaged["rows"][0]["probe_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="panel identity"):
        pairing.join(prepared, anchors, damaged)


def test_censoring_and_missing_panel():
    prepared, anchors, panel = fixed_panel()
    panel["rows"][0].update(endpoint=None, status="early_stop", endpoint_minus_own_zero=None)
    result = pairing.join(prepared, anchors, panel)
    selected = result["common_suffix_pairs"][0]["roles"]["selected"]
    assert selected["same_suffix_endpoint_minus_zero_prefix"] is None
    assert selected["response_minus_zero_prefix_response"] is None
    panel["rows"].pop()
    with pytest.raises(ValueError, match="complete roster"):
        pairing.join(prepared, anchors, panel)
