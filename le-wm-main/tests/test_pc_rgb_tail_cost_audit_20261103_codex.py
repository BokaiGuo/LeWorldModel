import pytest

from experiments.pc_rgb_tail_cost_audit_20261103_codex import audit


def records():
    key = {"root": 0, "budget": 900, "method": "spectral"}
    first = [[0., 0.]] * 10
    tail = [[.1, 0.]] * 5
    counts = {"prediction_token_samples": 3, "token_cap": 3}
    root = {"rows": [{**key, "chosen": {"chain": first + tail},
                      "counts": counts, "generation_counts": {},
                      "encoding_calls": 3, "encoding_seconds": 1,
                      "svd": {}, "planning_wall_seconds": 2}]}
    feedback = {"root_plan_sha256": "root", "rows": [{**key, "first10": first,
                "old_tail": tail, "decision": {"tail": tail}, "counts": counts,
                "encoding_calls": 3, "encoding_seconds": 1, "planning_seconds": 2}]}
    execution = {"cp10_sha256": "feedback", "images_sha256": "images",
                 "rows": [{**key, "first10": first, "suffix": tail}]}
    return root, feedback, execution


def test_audit_counts_only_known_and_does_not_certify():
    result = audit(*records(), "root", "feedback", "images")
    assert result["rows"][0]["known_prediction_token_samples_total"] == 6
    assert result["rows"][0]["selected_execution_actions"] == 15
    assert result["full_cost_certified"] is False
    assert result["unknown"]


def test_rejects_binding_and_hash_mismatch():
    root, feedback, execution = records()
    with pytest.raises(ValueError):
        audit(root, feedback, execution, "wrong", "feedback", "images")
    execution["rows"][0]["suffix"] = [[0., 0.]] * 5
    with pytest.raises(ValueError):
        audit(root, feedback, execution, "root", "feedback", "images")
