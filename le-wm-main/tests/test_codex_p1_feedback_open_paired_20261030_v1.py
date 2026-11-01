import copy
import json

import pytest

from experiments.spectral_reachability import codex_p1_feedback_open_paired_20261030_v1 as paired


def items():
    anchors = paired.open_saved.run.h10.native.read(paired.open_saved.run.h10.INPUT, cap=64*1024**2)["anchors"]
    op = json.loads(paired.OPEN_SAVED.read_bytes())["rows"][0]
    fb = json.loads(paired.FB_SAVED.read_bytes())["rows"][0]
    roster = json.loads(paired.FB_PREP.read_bytes())["roster"][0]
    return copy.deepcopy(anchors[0]), copy.deepcopy(op), copy.deepcopy(fb), copy.deepcopy(roster)


def test_identity_and_differences():
    a, op, fb, roster = items()
    result = paired.pair_row(a, op, fb, roster, {"open": {}, "feedback": []})
    for key in paired.METRICS:
        assert result["feedback_minus_open"][key] == (
            fb["endpoint_metrics"][key] - op["twenty_step"]["horizon_metrics"][key])
    assert result["identity"]["prefix5_float32_hex"] == op["prefix5_float32_hex"]


@pytest.mark.parametrize("field", ["index", "logical_pair", "root_slot", "condition", "arm", "role",
                                  "task_id", "physical_root_sha256", "prefix5_float32_hex", "cp5_file_sha256"])
def test_identity_tamper_rejected(field):
    a, op, fb, roster = items()
    op[field] = "tampered"
    with pytest.raises(ValueError):
        paired.pair_row(a, op, fb, roster, {"open": {}, "feedback": []})


def test_early_stop_is_missing_not_zero_and_stop_at_horizon_is_full():
    a, op, fb, roster = items()
    fb.update(status="native_stopped", accepted_suffix_steps=5, early_stop=True,
              native_stop_at_horizon=False, endpoint_metrics=None)
    result = paired.pair_row(a, op, fb, roster, {"open": {}, "feedback": []})
    assert result["feedback_minus_open"] is None
    fb.update(status="native_stopped", accepted_suffix_steps=10, early_stop=False,
              native_stop_at_horizon=True, endpoint_metrics=fb["last_observed_metrics"])
    result = paired.pair_row(a, op, fb, roster, {"open": {}, "feedback": []})
    assert result["feedback_minus_open"] is not None
    fb["early_stop"] = True
    with pytest.raises(ValueError):
        paired.pair_row(a, op, fb, roster, {"open": {}, "feedback": []})


def test_censor_is_missing_not_completed():
    a, op, fb, roster = items()
    old = op["twenty_step"]
    old.update(status="censored_out_of_domain", accepted_suffix_steps=5, censored=True,
               horizon_metrics=None)
    assert paired.pair_row(a, op, fb, roster, {"open": {}, "feedback": []})["feedback_minus_open"] is None


def test_action_source_hash_mismatch_rejected():
    _, op, fb, _ = items()
    op["twenty_step"]["raw10_sha256"] = "0"*64
    with pytest.raises(ValueError, match="open action SHA"):
        paired.action_sources(0, op, fb, {"shards": [{"index": 0}]})


def test_feedback_cp5_snapshot_mismatch_rejected():
    a, op, fb, roster = items()
    roster["checkpoint"]["state"][0] += 1
    with pytest.raises(ValueError, match="cp5 source"):
        paired.pair_row(a, op, fb, roster, {"open": {}, "feedback": []})
