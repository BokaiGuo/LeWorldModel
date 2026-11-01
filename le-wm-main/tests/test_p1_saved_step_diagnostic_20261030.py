from copy import deepcopy

import pytest

from experiments.spectral_reachability.p1_saved_step_diagnostic_20261030 import analyze


def fixture():
    evaluations = []
    for i in range(9):
        evaluations.append(dict(iteration=i, logits=[[float(-i)] * 10, [0.] * 10],
                                raw=[[[0., 0.]] * 5, [[0., 0.]] * 5],
                                raw_gradient=[[[1., 1.]] * 5, [[0., 0.]] * 5],
                                regularized=float(i)))
    return dict(rows=[dict(slot=0, arm="fixed", start=0, iteration=0,
                           evaluations=deepcopy(evaluations)) for _ in range(160)])


def test_increase_despite_descent():
    out = analyze(fixture())
    assert out["steps"] == 1280
    assert out["objective_increases"] == out["first_order_descent"] == 1280
    assert out["increases_despite_descent"] == 1280
    assert not out["gradient_recomputed"]


def test_rejects_missing_iterate_and_nonfinite():
    report = fixture()
    report["rows"][0]["evaluations"].pop()
    with pytest.raises(ValueError):
        analyze(report)
    report = fixture()
    report["rows"][0]["evaluations"][0]["regularized"] = float("nan")
    with pytest.raises(ValueError):
        analyze(report)
