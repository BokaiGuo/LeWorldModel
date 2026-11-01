import pytest
from copy import deepcopy

from experiments.spectral_reachability import codex_p1_root1_candidate_binding_20261101_exclusive as m


def test_role_binding_requires_correct_bank_and_exact_prefix():
    bank = [[[float(i) / 10, 0.0]] * 5 for i in range(3)]
    event = {"role": "greedy", "bank": [deepcopy(bank), deepcopy(bank)], "stage_scores": [0.2, 0.1],
             "original_selected_index": 0, "full_selected": bank}
    assert m.check_prefix(event, "greedy", bank[0]) == m.arr(bank, (3, 5, 2)).tolist()
    with pytest.raises(ValueError):
        m.check_prefix(event, "selected", bank[0])
    with pytest.raises(ValueError):
        m.check_prefix(event, "greedy", [[1, 0]] * 5)
    event["bank"][1][1][0][0] = 2
    with pytest.raises(ValueError):
        m.check_prefix(event, "greedy", bank[0])


def test_online_adapter_only_accepts_bound_root_goal_and_cp5():
    entry = {"required_cp5": {"pixel_sha256": "p", "goal_rgb_sha256": "g", "state": [1]},
             "prefix": [[0, 0]] * 5, "hold": [[[0, 0]] * 5] * 2,
             "optimized": [[[0.1, 0]] * 5] * 2}
    assert len(m.bind_inputs(entry, entry["required_cp5"], entry["prefix"])["optimized"]) == 2
    with pytest.raises(ValueError):
        m.bind_inputs(entry, dict(entry["required_cp5"], goal_rgb_sha256="other"), entry["prefix"])
    with pytest.raises(ValueError):
        m.bind_inputs(entry, entry["required_cp5"], [[1, 0]] * 5)


def test_trace_terminal_requires_actual_complete_node_status():
    m.check_terminal([{"kind": "terminal", "payload": {"status": "complete_node"}}])
    with pytest.raises(ValueError):
        m.check_terminal([{"kind": "terminal", "payload": {"status": "failed"}}])
