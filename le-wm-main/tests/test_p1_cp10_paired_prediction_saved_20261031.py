import copy

import pytest

from experiments.spectral_reachability import p1_cp10_paired_prediction_saved_20261031 as audit


def result_fixture():
    def arm(value):
        return {"prediction": [value] * 192, "actual": [value] * 192,
                "predicted_goal_mse": value ** 2, "real_goal_mse": value ** 2,
                "prediction_real_mse": 0.0}
    return {"schema": audit.run.NAME, "status": "complete", "host": {"device": "cuda:0"},
            "model_before": "fixed", "model_after": "fixed", "sources": {},
            "rows": [{"root_slot": i, "goal": [0.0] * 192,
                      "arms": {"open": arm(1.0), "feedback": arm(0.5)}} for i in range(2)]}


def test_saved_vectors_reject_mismatched_mse_or_root(monkeypatch):
    result = result_fixture()
    monkeypatch.setattr(audit.run, "sha", lambda _: "digest")
    monkeypatch.setattr(audit.run.feedback_plan.source.physical.h, "read", lambda _: result)
    checked = audit.verify("input", "digest")
    assert checked["rows"][0]["predicted_feedback_minus_open"] == -0.75
    bad = copy.deepcopy(result)
    bad["rows"][0]["arms"]["feedback"]["real_goal_mse"] = 99.0
    monkeypatch.setattr(audit.run.feedback_plan.source.physical.h, "read", lambda _: bad)
    with pytest.raises(ValueError, match="MSE arithmetic"):
        audit.verify("input", "digest")
    bad = copy.deepcopy(result)
    bad["rows"][1]["root_slot"] = 0
    monkeypatch.setattr(audit.run.feedback_plan.source.physical.h, "read", lambda _: bad)
    with pytest.raises(ValueError, match="fixed roster"):
        audit.verify("input", "digest")
