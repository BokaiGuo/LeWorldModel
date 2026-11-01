import copy
import pytest

from experiments.spectral_reachability import codex_e1_strict_saved_consumer_20261030_v1 as m


def actual():
    return m.load_pinned(m.PANEL, m.PANEL_SHA), m.load_pinned(m.CONTROLLER, m.CONTROLLER_SHA)


def identity():
    return {"root": 0, "step": 5, "task_sha256": "a" * 64, "target_sha256": "b" * 64,
            "metric": "object_xy", "budget": {x: {"attempted": 2, "completed": 2} for x in m.AXES}}


def test_actual_panel_remains_blocked():
    report = m.analyze(*actual())
    assert len(report["rows"]) == 18
    assert report["rows"][0]["saved_value"] is not None
    assert report["rankings"] is None and report["equal_total_budget"] == "unknown"
    assert report["scientific_admission"] is False and report["fresh_roots"] == 0


@pytest.mark.parametrize("field,value", [("root", 1), ("step", 10), ("task_sha256", "c" * 64),
                                         ("target_sha256", "d" * 64), ("metric", "native7")])
def test_distinct_goal_or_scope_rejected(field, value):
    a, b = identity(), identity()
    b[field] = value
    with pytest.raises(ValueError):
        m.check_comparable(a, b)


@pytest.mark.parametrize("axis", m.AXES)
def test_each_budget_axis_must_match(axis):
    a, b = identity(), identity()
    b["budget"][axis]["attempted"] += 1
    with pytest.raises(ValueError):
        m.check_comparable(a, b)


def test_identical_declarations_do_not_grant_authority():
    a = identity()
    assert m.check_comparable(a, copy.deepcopy(a)) is None


def test_missing_budget_and_boolean_rejected():
    a, b = identity(), identity()
    del b["budget"]["proposal"]
    with pytest.raises(ValueError):
        m.check_comparable(a, b)
    b = identity()
    b["budget"]["native"]["attempted"] = True
    with pytest.raises(ValueError):
        m.check_comparable(a, b)


@pytest.mark.parametrize("mutation", [
    lambda p: p["result"]["cells"][0]["value"]["goal"].__setitem__(0, -1),
    lambda p: p["result"]["cells"].__setitem__(0, copy.deepcopy(p["result"]["cells"][1])),
    lambda p: p["result"].update(equal_total_compute=True),
    lambda p: p["result"].update(checkpoint_denominator=17),
    lambda p: p.update(receipt_sha256="0" * 64),
])
def test_tampered_panel_rejected(mutation):
    p, c = actual()
    mutation(p)
    with pytest.raises(ValueError):
        m.analyze(p, c)


def test_hash_mismatch_rejected():
    with pytest.raises(ValueError):
        m.load_pinned(m.PANEL, "0" * 64)
