import numpy as np

from experiments import run_pusht_verified_contact_ranking_ladder as ladder


def test_paired_torque_actions_hold_the_normal_component_and_reverse_tangent(monkeypatch):
    monkeypatch.setattr(
        ladder, "boundary_inward_normal", lambda env, state: np.asarray([1.0, 0.0])
    )
    plus, minus, zero = ladder.paired_torque_actions(
        object(), np.zeros(7), inward_magnitude=0.1, tangent_magnitude=0.2
    )

    np.testing.assert_allclose(plus, [0.1, 0.2])
    np.testing.assert_allclose(minus, [0.1, -0.2])
    np.testing.assert_allclose(zero, [0.0, 0.0])


def test_same_state_paired_audit_reports_signed_contrast_and_no_runtime_permission():
    source = np.zeros(7, dtype=np.float64)
    goal = source.copy()
    goal[4] = 1.0

    def successor(state, action):
        after = state.copy()
        after[4] += float(action[1])
        return after

    audit = ladder.audit_same_state_pair(
        source,
        goal,
        plus=np.asarray([0.0, 0.2]),
        minus=np.asarray([0.0, -0.2]),
        zero=np.zeros(2),
        successor_fn=successor,
        angle_error_fn=lambda angle, target: abs(angle - target),
    )

    assert audit["same_source_state"] is True
    assert audit["plus_minus_contrast"] > 0.0
    assert audit["best_member"] == "plus"
    assert audit["runtime_successor_access"] is False
    assert audit["runtime_privileged_state_access"] is False


def test_two_pulse_oracle_selects_the_pair_with_lowest_real_terminal_cost():
    state = np.asarray([0.0])
    actions = np.asarray([[1.0], [2.0]])

    def successor(current, action):
        return current + action

    result = ladder.select_two_pulse_oracle(
        state,
        actions,
        successor_fn=successor,
        cost_fn=lambda final: float((final[0] - 3.0) ** 2),
    )

    assert result["first_index"] == 0
    assert result["second_index"] == 1
    assert result["terminal_cost"] == 0.0
    np.testing.assert_allclose(result["after_first"], [1.0])
    np.testing.assert_allclose(result["after_second"], [3.0])


def test_runtime_boundary_explicitly_rejects_privileged_successor_use():
    boundary = ladder.non_privileged_rgb_runtime_boundary()
    assert boundary == {
        "runtime_successor_access": False,
        "runtime_privileged_state_access": False,
        "runtime_observation": "RGB history and declared actions only",
        "status": "future_interface_only_not_evaluated_here",
    }


def test_extended_oracle_bank_requires_the_declared_113_candidates():
    ladder.validate_declared_oracle_bank(np.zeros((113, 2)), expected_count=113)
    try:
        ladder.validate_declared_oracle_bank(np.zeros((61, 2)), expected_count=113)
    except ValueError as error:
        assert "113" in str(error)
    else:
        raise AssertionError("a reduced candidate bank must be rejected")
