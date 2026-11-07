import numpy as np
import pytest

from experiments import pc_waypoint_geometry_20261108_codex as study


def test_zero_longitudinal_exactly_matches_old_compiler():
    task = {"start": [-.5, -.375, -.3], "goal": [.5, .25]}
    for lateral in np.linspace(-2, 2, 41):
        actual = study.generate(task, lateral, 0)
        expected = study.base.previous.geom.make_chain(task["start"], task["goal"], lateral)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[2] == expected[2]


def test_untruncated_program_reaches_goal_without_wall():
    task = {"start": [-.5, .1, .4], "goal": [.5, -.2]}
    found = 0
    for lateral in (-1., 0., 1.):
        for longitudinal in (-.5, 0., .5):
            chain, _, steps, _ = study.generate(task, lateral, longitudinal)
            assert chain.shape == (15, 2)
            assert np.abs(chain).max() <= 1
            if steps <= 15:
                found += 1
                path = study.free_trajectory(task["start"], chain[None])
                np.testing.assert_allclose(path[0, -1, :2], task["goal"], atol=1e-12)
    assert found == 9


def test_two_dimensional_waypoint_is_not_forced_onto_perpendicular_bisector():
    task = {"start": [-.5, -.2, 0.], "goal": [.5, .2]}
    _, waypoint, _, _ = study.generate(task, .9, .5)
    delta = np.array(task["goal"]) - task["start"][:2]
    midpoint = (np.array(task["goal"]) + task["start"][:2]) / 2
    assert np.dot(waypoint - midpoint, delta) > .5


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        study.compile_waypoint([0, 0, 0], [1, 1], [np.nan, 0])


def test_privileged_witness_is_legal_but_not_a_generator_input():
    chain, steps, _ = study.compile_waypoint([-.5, -.375, -.3], [.5, .25], [0., 1.05])
    path, _ = study.base.trajectory([-.5, -.375, -.3], chain[None])
    assert steps <= 15
    np.testing.assert_allclose(path[0, -1, :2], [.5, .25], atol=1e-12)
