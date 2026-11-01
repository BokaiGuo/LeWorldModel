from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_common_cp10_pair_audit_20261101_leibniz as audit


@pytest.fixture(scope="module")
def saved():
    return audit.load()[:2]


@pytest.mark.parametrize("field", ["goal", "model", "budget", "normalizer"])
def test_reject_pair_identity_mismatch(saved, field):
    reports, prepares = deepcopy(saved)
    if field == "goal":
        reports["cem"]["cp10_goal_binding"]["goal_pixel_sha256"] = "0"*64
    elif field == "model":
        reports["cem"]["model_after"]["all_buffers"] = "0"*64
    elif field == "budget":
        reports["cem"]["protocol"]["common_node_caps"]["predict_calls"] += 1
    else:
        prepares["cem"]["source"]["tasks"][0]["action_mean"][0] += 1
    with pytest.raises(ValueError):
        audit.verify_pair(reports, prepares)


def test_tampered_executed_action_rejected(saved):
    reports, prepares = deepcopy(saved)
    binding = audit.verify_pair(reports, prepares)
    reports["structured"]["rows"][0]["executed_raw"][0][0] += .1
    with pytest.raises(ValueError, match="actual action/order"):
        audit.audit_arm("structured", reports["structured"], binding)


def test_tampered_ledger_charge_rejected(saved):
    reports, prepares = deepcopy(saved)
    binding = audit.verify_pair(reports, prepares)
    node = reports["structured"]["nodes"]["s0_structured"][0]
    node["receipt"]["account"]["charged_upper_bounds"]["predict_calls"] += 1
    with pytest.raises(ValueError, match="ledger settlement"):
        audit.audit_arm("structured", reports["structured"], binding)


def test_pair_audit_from_saved_receipts(saved):
    reports, prepares = saved
    binding = audit.verify_pair(reports, prepares)
    s = audit.audit_arm("structured", reports["structured"], binding)
    c = audit.audit_arm("cem", reports["cem"], binding)
    assert (s["steps"], c["steps"]) == (13, 17)
    assert (s["totals"]["predict_calls"], c["totals"]["predict_calls"]) == (252, 775)
    assert (s["totals"]["backward_calls"], c["totals"]["backward_calls"]) == (624, 0)
    assert s["initial_pixel_sha256"] == c["initial_pixel_sha256"]
