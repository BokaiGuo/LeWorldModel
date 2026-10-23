from __future__ import annotations

import numpy as np

from experiments.run_pointmaze_v15_reachable_rgb_geometry import _select


def test_rgb_greedy_uses_only_position_from_four_dimensional_state() -> None:
    current = np.asarray([0.0, 0.0, 4.0, -3.0])
    target = np.asarray([1.0, 0.0])
    bank = np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    index, endpoints = _select("rgb_greedy", current, current, target, bank, None, None)
    assert index == 1
    assert endpoints is None
