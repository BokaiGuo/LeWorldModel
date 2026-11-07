import numpy as np
import pytest

from experiments.pc_exact_blocked_detour_20261107_codex import generate
from experiments.pc_exact_prefix_successor_composition_20261107_codex import rollout_n


def test_produces_bounded_replayable_actions():
    start = np.array([-.5, -.45, 0.])
    actions, trace, triggers, counts = generate(start, np.array([.5, -.35]), [-.5, .5, 1.])
    assert actions.shape == (3, 15, 2)
    assert trace.shape == (3, 16, 3)
    assert np.abs(actions).max() <= 1
    assert np.array_equal(triggers.sum(axis=1), counts)
    replay, _ = rollout_n(start, actions)
    np.testing.assert_allclose(trace, replay, atol=1e-12, rtol=0)


@pytest.mark.parametrize("offsets", [[], [float("nan")]])
def test_invalid_input(offsets):
    with pytest.raises(ValueError):
        generate([-.5, 0, 0], [.5, 0], offsets)
