import copy
import math

import pytest

from experiments.spectral_reachability import p1_two_horizon_rank_20261031 as selector


def test_rank_is_tie_stable_and_input_order_independent():
    values = [(0, 0.1, 0.4), (1, 0.2, 0.1), (2, 0.3, 0.2), (3, 0.4, 0.3)]
    assert selector.choose(values) == selector.choose(list(reversed(values)))
    assert selector.choose(values) == {"mid": 0, "end": 1, "combined": 1}
    assert selector.choose([(i, 1., 1.) for i in range(4)])["combined"] == 0
    with pytest.raises(Exception):
        selector.choose([(0, math.nan, 0), *values[1:]])


def test_outcomes_do_not_change_choice():
    data = selector.pair.pairing.read(selector.SOURCE, selector.SOURCE_SHA)
    before = selector.analyze(data)
    changed = copy.deepcopy(data)
    for row in changed["rows"]:
        row["success"] = not row["success"]
    after = selector.analyze(changed)
    assert [r["choices"] for r in before["rows"]] == [r["choices"] for r in after["rows"]]
