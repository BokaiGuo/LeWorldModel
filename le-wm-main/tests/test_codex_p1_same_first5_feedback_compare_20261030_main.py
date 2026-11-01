import copy

import pytest

from experiments.spectral_reachability import codex_p1_same_first5_feedback_compare_20261030_main as cmp


def fixtures():
    metric = dict.fromkeys(cmp.FIELDS, 1.0)
    metric["native_success"] = False
    rows_a, rows_b, shards = [], [], []
    for i in range(24):
        identity = {"index": i, "logical_pair": i // 3, "root_slot": 0,
                    "condition": "on", "arm": "test", "role": "selected"}
        rows_a.append(dict(identity, endpoint_metrics=metric, status="probe_completed", returned_suffix=10))
        rows_b.append(dict(identity, endpoint_metrics=metric, status="probe_completed", accepted_suffix_steps=10))
        shards.append({"index": i})
    transition = [dict(step=i+1, actual_bits="00", state=[0], snapshot={"s": 0},
                       terminated=False, truncated=False) for i in range(5)]
    return ({"rows": rows_a}, {"rows": rows_b}, {"shards": shards}, {"shards": shards}, transition)


def test_first_five_and_null_endpoint():
    a, b, ar, br, transitions = fixtures()
    a["rows"][0]["endpoint_metrics"] = None
    result = cmp.compare(a, b, ar, br, lambda *_: (transitions, transitions))
    assert result["summary"]["common_first5_verified"] == 24
    assert result["summary"]["comparable"] == 23
    assert result["rows"][0]["feedback_minus_open"] is None


def test_reject_state_or_action_mismatch():
    a, b, ar, br, transitions = fixtures()
    changed = copy.deepcopy(transitions)
    changed[2]["actual_bits"] = "01"
    with pytest.raises(ValueError, match="first five"):
        cmp.compare(a, b, ar, br, lambda *_: (transitions, changed))
    b["rows"][0]["condition"] = "off"
    with pytest.raises(ValueError, match="identity"):
        cmp.compare(a, b, ar, br, lambda *_: (transitions, transitions))
