import numpy as np
import pytest

from experiments.spectral_reachability import pc_wall_unicycle_atlas_cem_20261103 as exp


def test_witness_bounds_blocking_and_nongreedy_detour():
    actions = exp.witness()
    outcome = exp.replay(actions)
    assert actions.shape == (16, 2)
    assert np.all(actions >= exp.LOW) and np.all(actions <= exp.HIGH)
    assert outcome["terminal_error"] < 1e-12
    assert not any(outcome["blocked"])
    assert outcome["max_endpoint_distance"] > 1.


def test_wall_block_and_turn_are_separate():
    backend = exp.Backend()
    state, blocked = backend.advance(np.array([-.1, 0., 0.]), np.array([.2, .4]))
    assert blocked
    np.testing.assert_allclose(state, [-.1, 0., .4])
    with pytest.raises(ValueError, match="domain"):
        backend.advance(state, np.array([.31, 0.]))


def test_fd_jacobian_matches_finer_difference_away_from_wall():
    actions = np.array([[.04, .07], [.03, -.02], [-.01, .04], [.02, 0.]])
    backend = exp.Backend()
    jac = backend.jacobian(exp.ROOT, actions)
    assert backend.jacobians == 1 and backend.fd_rollouts == 16 and backend.transitions == 64
    reference = np.empty_like(jac)
    for i in range(actions.size):
        a, b = actions.copy(), actions.copy()
        a.flat[i] += 1e-6
        b.flat[i] -= 1e-6
        left = np.asarray(backend.rollout(exp.ROOT, a)["latent_history"][-1])
        right = np.asarray(backend.rollout(exp.ROOT, b)["latent_history"][-1])
        reference[:, i] = (left - right) / 2e-6
    np.testing.assert_allclose(jac, reference, atol=2e-9, rtol=1e-7)


def test_saved_results_replay_and_cem_budget():
    import json
    path = exp.Path(exp.__file__).resolve().parents[2] / "results/pc_wall_unicycle_atlas_cem_20261103_DEV01.json"
    result = json.loads(path.read_text())
    for row in result["rows"]:
        assert row["cem"]["backend"]["transitions"] <= row["atlas"]["backend"]["transitions"]
        for arm in ("atlas", "cem"):
            saved = row[arm]["selected"]
            current = exp.replay(np.asarray(saved["actions"]))
            np.testing.assert_allclose(current["trajectory"], saved["trajectory"], rtol=0, atol=1e-12)
            assert current["blocked"] == saved["blocked"]
