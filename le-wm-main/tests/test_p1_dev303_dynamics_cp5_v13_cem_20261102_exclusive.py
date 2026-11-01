from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_dynamics_cp5_v13_cem_20261102_exclusive as subject


@pytest.fixture(scope="module")
def sealed():
    return (subject.fixed(subject.MODEL_REPORT, subject.MODEL_SHA),
            subject.fixed(subject.PHYS_DIR / "report.json", subject.PHYS_SHA))


def test_real_saved_prefix_and_cp5_bindings(sealed):
    roles = subject.prefix_rows(*sealed, subject.PHYS_DIR)
    assert [r["index"] for r in roles] == [1, 8, 9, 10]
    assert all(len(r["actual_prefix"]) == 5 and len(r["cp5_pixel_sha256"]) == 64 for r in roles)


def test_outcomes_not_used(sealed):
    changed = deepcopy(sealed[1])
    for row in changed["rows"]:
        for transition in row["transitions"]:
            transition["state"] = "irrelevant"
            transition["reward"] = "irrelevant"
        for observation in row["observations"]:
            observation["state"] = "irrelevant"
            observation["snapshot"] = "irrelevant"
    assert subject.prefix_rows(sealed[0], changed, subject.PHYS_DIR) == subject.prefix_rows(
        *sealed, subject.PHYS_DIR)


@pytest.mark.parametrize("kind", ["action", "order", "pixel", "warmup", "selected"])
def test_corruption_rejected(sealed, kind):
    model, physical = deepcopy(sealed)
    row = next(x for x in physical["rows"] if x["index"] == 1 and x["metric"] == "goal_tangent")
    if kind == "action":
        next(t for t in row["transitions"] if t["phase"] == "prefix")["action"][0] = 0
    elif kind == "order":
        next(t for t in row["transitions"] if t["phase"] == "prefix")["step"] = 2
    elif kind == "warmup":
        next(t for t in row["transitions"] if t["phase"] == "warmup")["action"][0] = 1
    elif kind == "pixel":
        next(o for o in row["observations"] if o["phase"] == "prefix" and o["step"] == 5)["pixel_sha256"] = "0" * 64
    else:
        model["selections"]["goal_tangent"] = 8
    with pytest.raises(ValueError):
        subject.prefix_rows(model, physical, subject.PHYS_DIR)


def test_v13_equal_initialization_and_budget():
    calls = []
    def score(bank):
        calls.append(bank.shape)
        return np.square(bank).sum(axis=(2, 3, 4))
    one = subject.planner.plan_one([0.5, 0.5], score)
    two = subject.planner.plan_one([0.5, 0.5], score)
    assert len(calls) == 10 and set(calls) == {(64, 5, 2, 5, 2)}
    assert one == two and len(one["iterations"]) == 5


def test_preparation_source_and_boundaries():
    value = subject.protocol()
    assert value["source_report_sha256"] == {"model": subject.MODEL_SHA, "physical": subject.PHYS_SHA,
                                               "analysis": subject.ANALYSIS_SHA}
    assert value["helper_sha256"] == subject.METRIC_SHA
    assert value["shared_preparation"] == {"goal_encode_calls": 11, "cp5_encode_calls": 4}
    assert value["physical_actions_authorized"] is value["P1_admission"] is False
    assert value["per_plan"]["predict_samples"] == 3200
