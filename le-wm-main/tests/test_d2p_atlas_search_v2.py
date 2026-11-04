import numpy as np
import pytest

from experiments.spectral_reachability.d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from experiments.spectral_reachability.d2p_local_compiler import CompilerConfig, LocalCompiler
from experiments.spectral_reachability.pc_atlas_unicycle_smoke_20261103 import UnicycleBackend


def context():
    return {"layout": "states_with_between_state_actions_v1", "context_length": 2,
            "latent_dim": 3, "action_dim": 2, "latent_history": [[0., 0., 0.]] * 2,
            "action_history": [[0., 0.]]}


class ExactUnicycle:
    def rollout(self, context, actions):
        result = {**context, "latent_history": list(context["latent_history"]),
                  "action_history": list(context["action_history"])}
        x, y, angle = result["latent_history"][-1]
        for velocity, turn in actions:
            x, y, angle = x + velocity * np.cos(angle), y + velocity * np.sin(angle), angle + turn
            result["latent_history"] = (result["latent_history"] + [[float(x), float(y), float(angle)]])[-2:]
            result["action_history"] = (result["action_history"] + [[float(velocity), float(turn)]])[-1:]
        return result

    def jacobian(self, context, actions):
        result = np.zeros((3, actions.size))
        angle = context["latent_history"][-1][2]
        for index, (velocity, turn) in enumerate(actions):
            transition = np.array([[1., 0., -velocity * np.sin(angle)],
                                   [0., 1., velocity * np.cos(angle)], [0., 0., 1.]])
            result = transition @ result
            result[:, 2 * index:2 * index + 2] += [[np.cos(angle), 0.],
                                                    [np.sin(angle), 0.], [0., 1.]]
            angle += turn
        return result


def make_search(**kwargs):
    compiler = LocalCompiler(ExactUnicycle(), goal=[0., 1., 0.], sqrt_q=np.diag([1., 1., 0.]),
                             action_scale=[1., 1.], low=[-1., -1.], high=[1., 1.],
                             config=CompilerConfig(packet_steps=1, spectral_modes=2,
                                                   max_rollout_operations=1000,
                                                   max_jacobian_operations=100))
    return AtlasSearch(compiler, AtlasConfig(**kwargs))


def test_search_expands_and_replays_bounded_full_chains():
    result = make_search(seed=9).search(context())
    assert len(result["programs"]) >= 4
    assert all(len(p["edges"]) == 2 and p["witness_replayed"] for p in result["programs"])
    assert all(np.max(np.abs(p["actions"])) <= 1 for p in result["programs"])
    selected = result["programs"][result["selected_index"]]
    assert selected["terminal_cost"] == min(p["terminal_cost"] for p in result["programs"])
    assert result["operation_counts"]["jacobian_returned"] > len(result["programs"])
    assert result["P1_admission"] is False


def test_deterministic_and_abstains_without_strict_margin():
    left = make_search(seed=9, improvement_margin=100.).search(context())
    right = make_search(seed=9, improvement_margin=100.).search(context())
    assert left == right
    assert left["qualified_reconfiguration_index"] is None


def test_qualified_candidate_must_strictly_beat_both_controls():
    result = make_search(seed=13).search(context())
    index = result["qualified_reconfiguration_index"]
    if index is not None:
        candidate = result["programs"][index]
        assert candidate["prefix_cost"] >= result["root_cost"]
        assert candidate["terminal_cost"] < result["programs"][result["zero_index"]]["terminal_cost"]
        assert candidate["terminal_cost"] < result["programs"][result["same_bank_greedy_index"]]["terminal_cost"]


def test_budget_exhaustion_does_not_return_partial_search():
    search = make_search()
    search.compiler.config = CompilerConfig(packet_steps=1, max_rollout_operations=1)
    with pytest.raises(RuntimeError, match="cap exhausted"):
        search.search(context())


def test_configuration_rejects_invalid_composition():
    with pytest.raises(ValueError):
        AtlasConfig(depth=1)
    with pytest.raises(ValueError):
        AtlasConfig(improvement_margin=float("nan"))


def test_smoke_backend_multistep_jacobian_matches_finite_difference():
    backend = UnicycleBackend()
    actions = np.array([[.3, .4], [-.2, -.5]])
    actual = backend.jacobian(context(), actions)
    numeric = np.empty_like(actual)
    epsilon = 1e-6
    for index in range(actions.size):
        delta = np.zeros_like(actions)
        delta.flat[index] = epsilon
        plus = backend.rollout(context(), actions + delta)["latent_history"][-1]
        minus = backend.rollout(context(), actions - delta)["latent_history"][-1]
        numeric[:, index] = (np.array(plus) - np.array(minus)) / (2 * epsilon)
    np.testing.assert_allclose(actual, numeric, atol=1e-9)
