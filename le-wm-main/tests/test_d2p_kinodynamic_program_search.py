"""Finite non-neural graph fixtures, not physical efficacy evidence."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudgetExceeded
from experiments.spectral_reachability.d2p_kinodynamic_program_search import search_programs


def context(x=0):
    return dict(layout="states_with_between_state_actions_v1", context_length=2,
                latent_dim=1, action_dim=2, latent_history=[[0], [x]],
                action_history=[[0, 0]])


def packet(x):
    return np.full((5, 2), x)


def replay(parent, actions):
    child = deepcopy(parent)
    child["latent_history"] = [parent["latent_history"][-1],
                                [float(parent["latent_history"][-1][0] + actions[0, 0])]]
    child["action_history"] = [actions[-1].tolist()]
    return child


def search(**overrides):
    args = dict(primitive_bank=[packet(-1.), packet(1.)], root_context=context(),
                replay=replay, target_metric=lambda c: abs(2 - c["latent_history"][-1][0]),
                max_depth=2, beam_width=4, max_attempted_expansions=6)
    return search_programs(**(args | overrides))


def test_exhaustive_finite_tree_exact_witness():
    result = search()
    plan = result["best_plan"]
    assert result["status"] == "completed"
    assert not result["partial_coverage"]
    assert result["attempted_expansions"] == 6
    assert plan["primitive_indices"] == (1, 1)
    assert plan["cost"] == 0
    assert plan["actions"] == np.ones((10, 2)).tolist()
    assert plan["context"] == replay(replay(context(), packet(1.)), packet(1.))
    assert not result["optimality_claim"] and not result["p1_admitted"]


def test_deterministic_ties_shorter_then_lexicographic():
    for _ in range(2):
        result = search(target_metric=lambda c: 3.)
        assert result["best_plan"]["primitive_indices"] == (0,)
        assert [e["primitive_indices"] for e in result["attempts"]] == [
            [0], [1], [0, 0], [0, 1], [1, 0], [1, 1]]


def test_beam_pruning_reports_partial_and_no_optimum():
    result = search(beam_width=1)
    assert result["status"] == "completed"
    assert result["partial_coverage"] and result["beam_pruned"] > 0
    assert result["attempted_expansions"] == 4
    assert result["best_plan"]["cost"] == 0


def test_tree_search_finds_non_greedy_two_edge_route():
    transitions = {(0, -1): 1, (0, 1): 3, (1, -1): 2, (1, 1): 4,
                   (3, -1): 4, (3, 1): 4}
    costs = {1: 5., 3: 1., 2: 0., 4: 9.}
    def graph(c, a):
        return context(transitions[c["latent_history"][-1][0], int(a[0, 0])])
    metric = lambda c: costs[c["latent_history"][-1][0]]
    result = search(replay=graph, target_metric=metric, beam_width=2)
    assert result["best_plan"]["primitive_indices"] == (0, 0)
    assert result["best_plan"]["cost"] == 0
    greedy = search(replay=graph, target_metric=metric, beam_width=1)
    assert greedy["best_plan"]["primitive_indices"] == (1,)
    assert greedy["partial_coverage"]


@pytest.mark.parametrize("cap", [1, 2, 3, 5, 6])
def test_hard_attempt_cap_and_exact_boundary(cap):
    result = search(max_attempted_expansions=cap)
    assert result["attempted_expansions"] == cap
    assert result["status"] == ("completed" if cap == 6 else "budget_exhausted")
    assert result["best_plan"] is not None


def test_count_replay_failures_continue_other_edges():
    def broken(c, a):
        if a[0, 0] < 0:
            raise RuntimeError("edge failed")
        return replay(c, a)
    result = search(replay=broken)
    assert result["status"] == "failed"
    assert result["attempted_expansions"] == 4
    assert result["failed_attempts"] == result["replay_failed_attempts"] == 2
    assert result["best_plan"]["cost"] == 0
    assert result["partial_coverage"]


def test_prediction_budget_exception_stops_without_equating_samples():
    calls = []
    def guarded(c, a):
        calls.append(1)
        if len(calls) == 3:
            raise PredictionBudgetExceeded("external predict cap")
        return replay(c, a)
    result = search(replay=guarded)
    assert result["status"] == "budget_exhausted"
    assert result["attempted_expansions"] == len(calls) == 3
    assert result["failed_attempts"] == 1
    assert result["best_plan"]["depth"] == 1
    assert "predict_samples" not in result


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, 1j, "1", [1]])
def test_bad_metrics_never_select_fallback(bad):
    result = search(target_metric=lambda c: bad)
    assert result["status"] == "failed"
    assert result["best_plan"] is None
    assert result["metric_failed_attempts"] == 2


def test_failed_replay_returns_no_root_or_zero_fallback():
    def broken(c, a):
        raise ValueError("all fail")
    result = search(replay=broken)
    assert result["best_plan"] is None
    assert result["attempted_expansions"] == 2
    assert result["failed_attempts"] == 2


def test_nonmutation_even_with_mutating_callbacks():
    root = context()
    bank = [packet(-1.), packet(1.)]
    before_root, before_bank = deepcopy(root), deepcopy(bank)
    def mutating_replay(c, a):
        child = replay(c, a)
        c["latent_history"][-1][0] = 999
        a[:] = 0
        return child
    def mutating_metric(c):
        cost = abs(2 - c["latent_history"][-1][0])
        c["latent_history"][-1][0] = 999
        return cost
    result = search(root_context=root, primitive_bank=bank, replay=mutating_replay,
                    target_metric=mutating_metric)
    assert root == before_root
    for a, b in zip(bank, before_bank):
        np.testing.assert_array_equal(a, b)
    assert result["best_plan"]["context"]["latent_history"][-1] == [2.]
    assert result["best_plan"]["actions"] == np.ones((10, 2)).tolist()


@pytest.mark.parametrize("bad", [np.zeros((4, 2)), np.zeros((6, 2)),
                                  np.zeros((5, 3)), np.full((5, 2), 1.01),
                                  np.full((5, 2), np.nan), np.ones((5, 2), bool),
                                  np.ones((5, 2), complex)])
def test_no_padding_clipping_lossy_cast_or_partial_packets(bad):
    with pytest.raises(ValueError):
        search(primitive_bank=[bad])


def test_multi_packet_primitive_keeps_all_actions_and_precision():
    bank = [np.full((10, 2), 0.123456789012345)]
    result = search(primitive_bank=bank, max_depth=1)
    assert result["best_plan"]["actions"] == bank[0].tolist()
    assert result["best_plan"]["depth"] == 1


@pytest.mark.parametrize("name", ["max_depth", "beam_width", "max_attempted_expansions"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_strict_limits(name, bad):
    with pytest.raises(ValueError):
        search(**{name: bad})


def test_no_extra_outcome_context_fields_and_no_layout_drift():
    with pytest.raises(ValueError):
        search(root_context=context() | {"physical_success": True})
    def drift(c, a):
        return context() | {"outcome": 1}
    result = search(replay=drift)
    assert result["best_plan"] is None
    assert result["failed_attempts"] == 2
