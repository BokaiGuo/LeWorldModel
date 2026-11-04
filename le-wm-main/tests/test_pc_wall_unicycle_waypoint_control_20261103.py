import json

import numpy as np

from experiments.spectral_reachability import pc_wall_unicycle_waypoint_control_20261103 as control


def test_programs_are_bounded_for_extreme_waypoints():
    for point in ([0., 1.1], [-.5, -1.5], [.5, 1.5], [-.5, 0.]):
        actions = control.build(point)
        assert actions.shape == (16, 2)
        assert np.all(actions >= control.wall.LOW) and np.all(actions <= control.wall.HIGH)
    witness_like = control.wall.replay(control.build([0., 1.1]))
    assert witness_like["terminal_error"] < 1e-12
    assert not any(witness_like["blocked"])


def test_saved_control_replay_budget_and_selection():
    path = control.ROOT / "results/pc_wall_unicycle_waypoint_control_20261103_DEV01.json"
    report = json.loads(path.read_text())
    for row in report["rows"]:
        assert row["backend"]["transitions"] == 207 * 16
        assert row["backend"]["transitions"] <= 3320
        scores = np.asarray(row["scores"])
        assert len(scores) == 207
        assert row["selected_index"] == int(np.argmin(scores))
        rebuilt = control.build(row["waypoints"][row["selected_index"]])
        np.testing.assert_array_equal(rebuilt, row["selected"]["actions"])
        outcome = control.wall.replay(rebuilt)
        np.testing.assert_allclose(outcome["trajectory"], row["selected"]["trajectory"], rtol=0, atol=1e-12)
        assert abs(outcome["terminal_error"] - scores[row["selected_index"]]) < 1e-12
