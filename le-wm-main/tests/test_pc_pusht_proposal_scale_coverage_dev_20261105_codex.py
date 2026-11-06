import numpy as np

from experiments.pc_pusht_proposal_scale_coverage_dev_20261105_codex import banks, metrics


def test_banks_are_shared_bounded_and_repeatable():
    a, b = banks(), banks()
    for name in a:
        assert a[name].shape == (64, 15, 2)
        assert a[name].dtype == np.float32
        assert np.max(np.abs(a[name])) <= 1
        np.testing.assert_array_equal(a[name], b[name])
        for t in (0, 5, 10):
            np.testing.assert_array_equal(a[name][:, t:t+5], np.repeat(a[name][:, t:t+1], 5, axis=1))
    np.testing.assert_array_equal(a["mixed"][:32], a["correlated"][:32])
    np.testing.assert_array_equal(a["mixed"][32:], a["wide"][32:])


def test_periodic_success_and_boundary_count():
    target = np.array([100, 100, 200, 200, 0.0])
    states = np.array([[100, 100, 200, 200, 2*np.pi], [0, 100, 200, 200, 0]])
    result = metrics(states, target, target, np.array([False, True]))
    assert result["success_count"] == 1
    assert result["terminal_center_outside_count"] == 1
    assert result["trajectory_center_outside_count"] == 1
