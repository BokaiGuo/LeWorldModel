from __future__ import annotations

import numpy as np
import pytest

from experiments.decision_benchmark.build_pointmaze_support_expanded_effect_policy import (
    FORBIDDEN_RUNTIME_ROOTS,
    _prefix_actions,
    parse_roots,
    validate_collection_partition,
)
from experiments.run_live_pointmaze_planning import GOAL_STATE


def test_support_expansion_partition_rejects_gate_or_test_roots() -> None:
    validate_collection_partition([1800, 1801], [1854, 1855])
    with pytest.raises(ValueError, match="disjoint"):
        validate_collection_partition([1800, 1854], [1854, 1855])
    with pytest.raises(ValueError, match="may not read"):
        validate_collection_partition([1800], [min(FORBIDDEN_RUNTIME_ROOTS)])


def test_support_expansion_prefixes_are_deterministic_runtime_pulses() -> None:
    state = np.asarray([0.2, 0.4, 0.0, 0.0])
    left = _prefix_actions(state, 1800, 3)
    right = _prefix_actions(state, 1800, 3)
    assert len(left) == 3
    assert all(value.shape == (2,) for value in left)
    assert np.allclose(left, right)
    assert np.all(np.linalg.norm(np.asarray(left), axis=1) <= 1.0 + 1.0e-6)
    assert np.linalg.norm(GOAL_STATE - state[:2]) > 0.0


def test_support_expansion_root_parser_rejects_duplicates() -> None:
    assert parse_roots("1800, 1801", "roots") == [1800, 1801]
    with pytest.raises(ValueError, match="duplicate"):
        parse_roots("1800,1800", "roots")
