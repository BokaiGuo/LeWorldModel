import numpy as np
import pytest

from experiments.spectral_reachability.pc_exact_unicycle_development import evaluate, run_panel, step, structured


def test_turn_then_progress_changes_future_velocity_response():
    goal = np.array([0.0, 1.0])
    actions, ledger = structured(goal)
    final, trajectory = evaluate(actions, goal)
    assert np.linalg.norm(trajectory[1, :2] - goal) == pytest.approx(1.0)
    assert final == pytest.approx(0.0, abs=1e-12)
    assert ledger["zero_prefix_final_error"] == pytest.approx(1.0)
    assert ledger["transitions"] == 10


def test_panel_is_reproducible_and_has_budget_contract():
    first, second = run_panel(4), run_panel(4)
    assert first == second
    for row in first:
        assert row["methods"]["cem"]["ledger"]["transitions"] == 64
        assert row["methods"]["structured"]["ledger"]["transitions"] <= 64
    with pytest.raises(ValueError, match="bounds"):
        step(np.zeros(3), [2, 0])


def test_velocity_response_matches_centered_finite_difference():
    state = np.array([0.2, -0.4, 0.71])
    epsilon = 1e-6
    finite_difference = (step(state, [epsilon, 0]) - step(state, [-epsilon, 0]))[:2] / (2 * epsilon)
    assert np.allclose(finite_difference, [np.cos(state[2]), np.sin(state[2])], atol=1e-10)
