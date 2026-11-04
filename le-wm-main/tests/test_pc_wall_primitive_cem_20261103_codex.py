import numpy as np
import pytest

from experiments.spectral_reachability.pc_wall_primitive_cem_20261103_codex import (
    ELITES, POP, ROUNDS, primitive, run,
)
from experiments.spectral_reachability.pc_wall_unicycle_atlas_cem_20261103 import HIGH, LOW, replay


def test_primitive_action_binding_and_domain():
    values = np.array([.7, .1, -.3, -.2])
    actions = primitive(values)
    assert actions.shape == (16, 2)
    np.testing.assert_array_equal(actions[0], [0., .7])
    np.testing.assert_array_equal(actions[1:6], np.tile([.1, 0.], (5, 1)))
    np.testing.assert_array_equal(actions[6:8], np.tile([0., -.3], (2, 1)))
    np.testing.assert_array_equal(actions[8:13], np.tile([-.2, 0.], (5, 1)))
    assert np.all(actions >= LOW) and np.all(actions <= HIGH)
    with pytest.raises(ValueError):
        primitive([0., HIGH[0] + .01, 0., 0.])


@pytest.mark.parametrize("arm", ["flat", "primitive"])
def test_all_saved_candidates_and_selection_replay(arm):
    row = run(20261103, arm)
    assert row["backend"]["rollouts"] == ROUNDS * POP
    assert row["backend"]["transitions"] == ROUNDS * POP * 16
    best = float("inf")
    for round_record in row["rounds"]:
        assert len(round_record["samples"]) == POP
        scores = []
        for sample in round_record["samples"]:
            actions = sample if arm == "flat" else primitive(sample)
            scores.append(replay(actions)["terminal_error"])
        np.testing.assert_allclose(scores, round_record["scores"], atol=1e-12, rtol=0)
        assert round_record["elite_indices"] == np.argsort(scores, kind="stable")[:ELITES].tolist()
        best = min(best, min(scores))
        assert round_record["best_so_far"] == pytest.approx(best, abs=1e-12)
    assert row["selected"]["terminal_error"] == pytest.approx(best, abs=1e-12)
    assert replay(row["selected"]["actions"]) == row["selected"]
