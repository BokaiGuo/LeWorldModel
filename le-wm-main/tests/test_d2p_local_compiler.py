import numpy as np
import pytest

from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
from experiments.spectral_reachability.d2p_local_compiler import (
    CompilerConfig,
    LocalCompiler,
    local_geometry,
)


def context(latent=(0.0, 0.0), action_dim=1):
    return {
        "layout": "states_with_between_state_actions_v1",
        "context_length": 2,
        "latent_dim": len(latent),
        "action_dim": action_dim,
        "latent_history": [list(latent), list(latent)],
        "action_history": [[0.0] * action_dim],
    }


def test_local_geometry_truncates_unreachable_residual():
    result = local_geometry(
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([1.0, 2.0]),
        np.eye(2),
        np.eye(2),
        0.01,
    )
    assert result["rank"] == 1
    assert result["unreachable_residual"][1] == pytest.approx(2.0)
    assert result["unreachable_fraction"] == pytest.approx(4.0 / 5.0)


def test_local_geometry_rejects_malformed_metrics():
    with pytest.raises(ValueError):
        local_geometry(np.eye(2), [1.0, 2.0], np.eye(3), np.eye(2), 0.01)
    with pytest.raises(ValueError):
        local_geometry(np.eye(2), [1.0, 2.0], np.eye(2), np.eye(2), 0.0)


def test_weighted_ridge_agrees_with_normal_equations():
    g = np.array([[1., 2.], [-1., .5], [2., 0.]])
    q = np.diag([2., .3, 1.])
    ri = np.diag([.4, 3.])
    r = np.array([3., -2., 1.])
    k = q @ g @ ri
    step = np.linalg.solve(k.T @ k + .1 * np.eye(2), k.T @ q @ r)
    got = local_geometry(g, r, q, ri, .1)
    np.testing.assert_allclose(got["raw_delta"], ri @ step)


def test_zero_rank_retains_entire_residual():
    got = local_geometry(np.zeros((2, 3)), [1., 2.], np.eye(2), np.eye(3), .1)
    assert got["rank"] == 0
    assert got["raw_spectral_directions"] == []
    np.testing.assert_array_equal(got["unreachable_residual"], [1., 2.])


class LinearBackend:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)

    def rollout(self, current, actions):
        result = dict(current)
        result["latent_history"] = list(current["latent_history"])
        result["action_history"] = list(current["action_history"])
        state = np.asarray(current["latent_history"][-1])
        for action in actions:
            state = state + self.matrix @ action
            result["latent_history"].append(state.tolist())
            result["action_history"].append(action.tolist())
            result["latent_history"] = result["latent_history"][-2:]
            result["action_history"] = result["action_history"][-1:]
        return result

    def jacobian(self, current, actions):
        return np.kron(np.ones((1, len(actions))), self.matrix)


def compiler():
    config = CompilerConfig(
        packet_steps=1,
        suffix_packets=1,
        spectral_modes=1,
        prefix_radius=0.5,
        line_scales=(0.0, 0.5, 1.0),
    )
    return LocalCompiler(
        LinearBackend([[1.0], [0.0]]),
        goal=np.array([1.0, 0.0]),
        sqrt_q=np.eye(2),
        action_scale=np.ones(1),
        low=np.array([-1.0]),
        high=np.array([1.0]),
        config=config,
    )


def test_progress_keeps_zero_incumbent_and_bounds_actions():
    compiler_instance = compiler()
    result = compiler_instance.progress(context())
    actions = np.asarray(result["actions"])
    assert actions.shape == (1, 1)
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)
    assert result["trials"][0]["scale"] == 0.0
    assert result["accepted_scale"] >= 0.0


def test_compile_replays_full_action_witness_and_relinearizes_suffix():
    compiler_instance = compiler()
    result = compiler_instance.compile(context())
    assert result["schema"] == "d2p_local_compiler.v1"
    assert result["programs"]
    assert all(program["witness_replayed"] for program in result["programs"])
    for program in result["programs"]:
        assert digest(program["output_context"]) == digest(
            compiler_instance.backend.rollout(
                context(), np.asarray(program["actions"], dtype=float)
            )
        )
        assert program["suffix"]["geometry"]["rank"] == 1
    assert result["operation_counts"]["rollout_attempted"] == result["operation_counts"]["rollout_returned"]


def test_compiler_does_not_mutate_input_context_or_action_arrays():
    compiler_instance = compiler()
    root = context()
    snapshot = repr(root)
    compiler_instance.compile(root)
    assert repr(root) == snapshot


def test_budget_failure_is_counted_and_fail_closed():
    config = CompilerConfig(packet_steps=1, max_rollout_operations=1)
    instance = LocalCompiler(
        LinearBackend([[1.0], [0.0]]),
        goal=np.array([1.0, 0.0]),
        sqrt_q=np.eye(2),
        action_scale=np.ones(1),
        low=np.array([-1.0]),
        high=np.array([1.0]),
        config=config,
    )
    with pytest.raises(RuntimeError):
        instance.progress(context())
    assert instance.counts["rollout_attempted"] == 1
    assert instance.counts["rollout_returned"] == 1
