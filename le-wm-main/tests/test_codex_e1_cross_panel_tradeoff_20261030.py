import copy
import hashlib
import json

import pytest

from experiments.spectral_reachability import codex_e1_cross_panel_tradeoff_20261030 as m


def saved():
    raw = m.SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == m.SHA
    return json.loads(raw)


@pytest.mark.parametrize("a,b,expected", [
    ([1, 2], [1, 2], "tie"),
    ([1, 2], [0, 2], "forced_dominates"),
    ([1, 2], [2, 3], "subgoal_dominates"),
    ([1, 2], [0, 3], "tradeoff"),
    (None, [0, 3], "missing"),
])
def test_classification(a, b, expected):
    assert m.classify(a, b) == expected


def test_invalid_metric():
    with pytest.raises(ValueError):
        m.classify([float("nan"), 1], [1, 1])


def test_real_fixed_grid():
    result = m.analyze(saved())
    assert len(result["rows"]) == 54
    assert all(sum(x.values()) == 9 for x in result["counts"].values())
    assert result["fresh_roots"] == 0


def test_duplicate_and_goal_mismatch_rejected():
    data = saved()
    data["result"]["pairs"][1] = copy.deepcopy(data["result"]["pairs"][0])
    with pytest.raises(ValueError):
        m.analyze(data)
    data = saved()
    data["result"]["pairs"][1]["forced"]["goal"][0] += 1
    with pytest.raises(ValueError):
        m.analyze(data)


def test_missing_not_coerced_to_tie():
    data = saved()
    pair = data["result"]["pairs"][1]
    pair.update(status="missing", missing=["forced_checkpoint"], forced=None)
    data["result"]["observed"] -= 1
    assert m.analyze(data)["rows"][1]["category"] == "missing"
