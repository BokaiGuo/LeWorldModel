import numpy as np

from experiments.pc_jepa_multiroot_decision_regret_posthoc_20261105_codex import metrics


def test_joint_conditions_include_agent_and_angle():
    goal = np.zeros(5)
    states = np.array([[0, 0, 0, 0, 0], [21, 0, 0, 0, 0], [0, 0, 0, 0, np.pi]])
    _, _, _, native, wrapped, violation = metrics(states, goal)
    assert native.tolist() == wrapped.tolist() == [True, False, False]
    assert violation[0] == 0 and np.all(violation[1:] > 1)


def test_native_angle_bug_exposed_not_silently_corrected():
    goal = np.zeros(5)
    states = np.array([[0, 0, 0, 0, 8 * np.pi + 1]])
    _, native_angle, wrapped_angle, native, wrapped, _ = metrics(states, goal)
    assert native_angle[0] < 0
    assert np.isclose(wrapped_angle[0], 1)
    assert native[0] and not wrapped[0]
