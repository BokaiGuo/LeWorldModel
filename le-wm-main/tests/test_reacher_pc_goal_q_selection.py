import pytest

from experiments.spectral_reachability.reacher_pc_goal_q_selection import select


def test_reference_eligibility_and_stable_tie():
    rows = [{"proposal_index": i, "immediate_goal_cost": a, "cem_future_goal_cost": b}
            for i, (a, b) in enumerate([(1., 1.), (1.1, 0.8), (1.2, 0.8), (0.9, 0.1)])]
    assert select(rows) == (1, 2)
    assert select(rows[:1] + [{**rows[1], "immediate_goal_cost": 0.9}]) == (None, 0)


def test_invalid_bank_and_nonfinite_fail_closed():
    with pytest.raises(ValueError):
        select([{"proposal_index": 1, "immediate_goal_cost": 1., "cem_future_goal_cost": 1.}] * 2)
    with pytest.raises(ValueError):
        select([{"proposal_index": i, "immediate_goal_cost": float("nan"), "cem_future_goal_cost": 1.}
                for i in range(2)])
