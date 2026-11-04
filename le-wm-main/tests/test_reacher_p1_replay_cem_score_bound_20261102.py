import numpy as np
import pytest

from experiments.spectral_reachability import reacher_p1_replay_cem_score_bound_20261102 as check


def test_winner_bound_for_l2_and_q_and_violation(monkeypatch):
    monkeypatch.setattr(check.qanalysis.pairwise, "analyze", lambda receipt: None)
    rows = [{"group": "L2", "seed": 1, "arm": "a"}, {"group": "Q", "seed": 1, "arm": "a"}]
    event = {"kind": "arm_returned", "seed": 1, "arm": "a",
             "result": {"suffix_receipt": {"trace": [3.0, 1.0], "final_selection": "best_sample"}}}
    pred = [{"kind": "model_returned", "phase": "prediction", "seed": 1, "arm": "a",
             "group": group, "predicted": [1.0, 1.0], "goal": [0.0, 0.0]}
            for group in ("L2", "Q")]
    result = check.score_bounds({"rows": rows}, {"L2": [event], "Q": [event]},
                                pred, {1: np.eye(2) * 2})
    assert result["rows"][0]["within_bound"]
    assert result["rows"][1]["replayed_score"] == 4.0
    assert result["violations"] == 1
    with pytest.raises(ValueError, match="roster"):
        check.score_bounds({"rows": rows}, {"L2": [event, event], "Q": [event]},
                           pred, {1: np.eye(2)})
