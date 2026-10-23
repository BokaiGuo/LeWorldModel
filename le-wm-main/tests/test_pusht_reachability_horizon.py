import numpy as np
from experiments import audit_pusht_reachability_horizon as audit

from experiments.audit_pusht_reachability_horizon import (
    action_bank_fingerprint,
    route_root,
    select_horizon_oracle,
)


def test_one_pulse_oracle_selects_best_true_successor():
    result = select_horizon_oracle(
        np.asarray([0.0]),
        horizon=1,
        action_bank_fn=lambda _state: np.asarray([[1.0], [3.0]]),
        successor_fn=lambda state, action: state + action,
        cost_fn=lambda state: float((state[0] - 3.0) ** 2),
        beam_width=2,
    )
    assert result.first_index == 1
    assert result.sequence_indices == (1,)
    assert result.terminal_cost == 0.0
    assert result.expanded_nodes == 2
    assert result.bounded is False


def test_two_pulse_oracle_finds_sequence_unavailable_to_one_pulse():
    bank = lambda _state: np.asarray([[1.0], [2.0]])
    successor = lambda state, action: state + action
    cost = lambda state: float((state[0] - 3.0) ** 2)
    one = select_horizon_oracle(np.asarray([0.0]), horizon=1, action_bank_fn=bank, successor_fn=successor, cost_fn=cost, beam_width=2)
    two = select_horizon_oracle(np.asarray([0.0]), horizon=2, action_bank_fn=bank, successor_fn=successor, cost_fn=cost, beam_width=2)
    assert one.terminal_cost == 1.0
    assert two.terminal_cost == 0.0
    assert two.sequence_indices == (0, 1)
    assert two.expanded_nodes == 6
    assert two.bounded is False


def test_three_pulse_oracle_is_explicitly_beam_bounded():
    bank = lambda _state: np.asarray([[1.0], [2.0]])
    result = select_horizon_oracle(
        np.asarray([0.0]), horizon=3, action_bank_fn=bank,
        successor_fn=lambda state, action: state + action,
        cost_fn=lambda state: float((state[0] - 5.0) ** 2), beam_width=1,
    )
    assert result.bounded is True
    assert result.searched_horizon == 3
    # depth 1 has 2 expansions, then only the best single beam node expands
    # at depths 2 and 3 (2 + 2 + 2), unlike an exhaustive 2**3 search.
    assert result.expanded_nodes == 6


def test_routing_requires_three_matched_horizon_arms():
    common = {"pose_success": False}
    routes = {
        "one_pulse": common,
        "two_pulse": common,
        "three_pulse_bounded": {"pose_success": True},
    }
    positive = route_root(routes)
    assert positive["route"] == "ranking_limited"
    assert positive["first_successful_arm"] == "three_pulse_bounded"
    negative = route_root({name: {"pose_success": False} for name in routes})
    assert negative["route"] == "candidate_or_horizon_limited"
    assert negative["oracle_reachable"] is False


def test_action_bank_fingerprint_is_order_sensitive_and_stable():
    bank = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    assert action_bank_fingerprint(bank) == action_bank_fingerprint(bank.copy())
    assert action_bank_fingerprint(bank) != action_bank_fingerprint(bank[::-1])


def test_nonterminal_root_collection_filters_terminal_and_duplicate_roots(monkeypatch):
    terminal = {"state": np.zeros(7), "goal": np.zeros(7)}
    active_a = {"state": np.ones(7), "goal": np.zeros(7)}
    active_b = {"state": np.full(7, 2.0), "goal": np.zeros(7)}
    monkeypatch.setattr(
        audit,
        "make_boundary_contact_roots",
        lambda _env, _count, _seed: [terminal, active_a, active_a, active_b],
    )
    monkeypatch.setattr(
        audit,
        "is_pose_success",
        lambda state, goal: bool(np.allclose(state, goal)),
    )
    roots = audit.collect_nonterminal_boundary_contact_roots(object(), count=2, seed=7)
    assert len(roots) == 2
    assert len({audit.root_fingerprint(root) for root in roots}) == 2
    assert all(not np.allclose(root["state"], root["goal"]) for root in roots)
