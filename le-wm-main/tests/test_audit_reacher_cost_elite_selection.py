from __future__ import annotations

import numpy as np
import pytest

from experiments.audit_reacher_cost_elite_selection import _elite_selection_scores


def test_elite_selection_uses_model_cost_only_inside_safe_set() -> None:
    no_dynamics = np.asarray([0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3])
    model_cost = np.asarray([2.0, 0.0, -10.0, -20.0, 0.0, 1.0, -10.0, -20.0])
    groups = ["a"] * 4 + ["b"] * 4
    scores = _elite_selection_scores(no_dynamics, model_cost, groups, 2)
    assert int(np.argmin(scores[:4])) == 1
    assert int(np.argmin(scores[4:])) == 0
    assert np.isinf(scores[2]) and np.isinf(scores[3])


def test_elite_selection_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        _elite_selection_scores(np.ones(2), np.ones(2), ["a", "a"], 0)
