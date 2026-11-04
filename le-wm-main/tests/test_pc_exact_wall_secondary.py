import numpy as np

from experiments.spectral_reachability import pc_exact_wall_kill as wall
from experiments.spectral_reachability.pc_exact_wall_secondary import bridge_program, random_bridge, gradient


def test_bridge_program_respects_horizon_and_action_limits():
    program = bridge_program([0., 1.1])
    assert program.shape == (wall.HORIZON, 2)
    assert np.all(np.linalg.norm(program, axis=1) <= wall.RADIUS + 1e-12)


def test_candidate_ledger_and_selected_cost():
    result = random_bridge()
    assert len(result["candidates"]) == 64
    assert result["ledger"] == {"scores": 64, "transitions": 768}
    assert result["best_cost"] == min(row["cost"] for row in result["candidates"])


def test_gradient_records_actual_scoring_calls():
    result = gradient()
    count = sum(len(row["objective_scores"]) for row in result["starts"])
    assert result["ledger"]["scores"] == count
    assert result["ledger"]["transitions"] == count * wall.HORIZON
    assert np.isfinite(result["best_cost"])
    for row in result["starts"]:
        replay, _ = wall.rollout_batch(np.asarray(row["selected_program"])[None])
        assert np.isclose(row["cost"], replay[0], rtol=0, atol=1e-12)
        assert row["cost"] == row["objective_scores"][-1]
