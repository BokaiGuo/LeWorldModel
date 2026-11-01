import copy
import json

import pytest

from experiments.spectral_reachability import codex_p1_source_authority_gap_20261030 as m


def test_empty_receipt_has_all_gaps():
    result = m.validate_receipt(m.empty_receipt())
    assert len(result["training_missing"]) == 7
    assert sum(map(len, result["calls_missing"].values())) == 10
    assert len(result["calls_missing"]["B"]) == 3
    assert len(result["scope_missing"]) == 5
    assert not result["source_run_authorized"]


def test_claimed_complete_does_not_grant_authority():
    r = m.empty_receipt()
    r["training"] = {k: "a" * 64 for k in m.TRAIN_FIELDS}
    r["scope"] = {k: "b" * 64 for k in m.SCOPE_FIELDS}
    r["calls"] = {c: "c" * 64 for calls in m.CALLS.values() for c in calls}
    out = m.validate_receipt(r)
    assert not out["training_missing"] and not out["scope_missing"]
    assert not any(out["calls_missing"].values())
    assert out["submitted_hashes_are_unverified_claims"]
    assert not out["history_complete"] and not out["freshness_certified"]


@pytest.mark.parametrize("mutation", ["extra", "outcome", "wrong_model", "invalid_hash",
                                      "unknown_call", "boolean"])
def test_reject_unexpected_or_leaking_input(mutation):
    r = copy.deepcopy(m.empty_receipt())
    if mutation == "extra":
        r["seed"] = 1
    elif mutation == "outcome":
        r["training"]["success"] = "a" * 64
    elif mutation == "wrong_model":
        r["checkpoint_sha256"] = "0" * 64
    elif mutation == "invalid_hash":
        r["training"][m.TRAIN_FIELDS[0]] = "bad"
    elif mutation == "unknown_call":
        r["calls"]["unknown"] = "a" * 64
    else:
        r["scope"][m.SCOPE_FIELDS[0]] = True
    with pytest.raises(ValueError):
        m.validate_receipt(r)


def test_strict_json():
    with pytest.raises(ValueError):
        m.parse_receipt(b'{"schema":1,"schema":2}')
    with pytest.raises(ValueError):
        m.parse_receipt(b'{"value":NaN}')
    assert m.parse_receipt(json.dumps(m.empty_receipt()).encode()) == m.empty_receipt()


def test_fixed_local_metadata():
    actual = m.snapshot()
    assert actual["checkpoint_weights_sha256"] == m.WEIGHTS
    assert actual["known_dispatch_count"] == 10
    assert actual["inventory_assessment"]["training_lineage_verified"] is False
