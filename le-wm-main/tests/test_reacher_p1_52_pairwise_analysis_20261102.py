import copy

import pytest

from experiments.spectral_reachability import reacher_p1_52_pairwise_analysis_20261102 as analysis


def fixture():
    rows = []
    for group in ("L2", "Q"):
        for seed in (20263160, 20263161):
            for i in range(13):
                rows.append({"group": group, "seed": seed, "prefix_index": i, "status": "complete",
                             "predicted_goal_distance": float(i),
                             "actual_rgb_goal_distance": float(-i),
                             "physical_distance": float(i), "prediction_error": 1.0})
    return {"status": "complete", "completed": 52, "rows": rows}


def test_reversal_and_ties():
    report = analysis.analyze(fixture())
    counts = report["panels"][0]["pairwise"]
    assert counts["predicted_goal_distance_vs_actual_rgb_goal_distance"] == {
        "reversed": 78, "non_tied": 78, "total": 78}
    assert report["same_prefix_q_vs_l2"][0]["q_minus_l2"][
        "predicted_goal_distance_vs_actual_rgb_goal_distance"]["non_tied"] == 0


def test_rejects_missing_reordered_and_nonfinite():
    receipt = fixture()
    for mutation in ("missing", "reordered", "nan"):
        changed = copy.deepcopy(receipt)
        if mutation == "missing":
            changed["rows"].pop()
        elif mutation == "reordered":
            changed["rows"][0], changed["rows"][1] = changed["rows"][1], changed["rows"][0]
        else:
            changed["rows"][0]["physical_distance"] = float("nan")
        with pytest.raises(ValueError):
            analysis.analyze(changed)
