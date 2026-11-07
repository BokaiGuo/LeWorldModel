import numpy as np
import pytest

from experiments.pc_tworoom_goal_chart_actual_successor_20261107_codex import make_chart


def test_chart_recovers_known_xy_displacements():
    rng = np.random.default_rng(16)
    jacobian = rng.normal(size=(32, 2))
    baseline = rng.normal(size=32)
    probes = np.stack((baseline + 4 * jacobian[:, 0], baseline - 4 * jacobian[:, 0],
                       baseline + 4 * jacobian[:, 1], baseline - 4 * jacobian[:, 1]))
    chart = make_chart(probes)
    np.testing.assert_allclose(chart @ (jacobian @ np.array([3.5, -2.0])), [3.5, -2.0], atol=1e-12)


def test_chart_rejects_invalid_or_degenerate():
    assert make_chart(np.zeros((4, 10))) is None
    with pytest.raises(ValueError):
        make_chart(np.zeros((3, 10)))
