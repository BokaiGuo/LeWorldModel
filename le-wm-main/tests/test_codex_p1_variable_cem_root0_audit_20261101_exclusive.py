from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p1_variable_cem_root0_audit_20261101_exclusive as audit


def test_saved_root0():
    result = audit.audit()
    assert result["steps"] == 17
    assert result["node_predict"] == [195, 194, 193, 193]
    assert result["issued_steps"] == [5, 5, 5, 2]
    assert result["native_success"]


def test_saved_root1():
    result = audit.audit(1)
    assert result["steps"] == 6
    assert result["node_predict"] == [195, 194]
    assert result["unstarted_boundaries"] == [10, 15]
    assert result["issued_steps"] == [5, 1]
    assert result["native_success"]


def test_semantic_tamper_of_winner_or_score_rejected():
    _, _, events, nodes = audit.load_sources()
    changed = deepcopy(events)
    chosen = next(x for x in changed if x.get("type") == "selection" and x["boundary"] == 15)
    chosen["index"] = 999999
    with pytest.raises(ValueError):
        audit.verify_cem_node(changed, nodes[-1], 15)
    changed = deepcopy(events)
    update = next(x for x in changed if x.get("type") == "update" and x["boundary"] == 5)
    update["elite_indices"][0] = 99
    with pytest.raises(ValueError):
        audit.verify_cem_node(changed, nodes[1], 5)


def test_native_success_requires_all_xy_and_angle():
    assert audit.native([10, 0, 10, 0, 0.1, 100, -100], [0]*7)[3]
    assert not audit.native([19, 0, 19, 0, 0, 0, 0], [0]*7)[3]


def test_root1_missing_score_and_changed_packet_rejected():
    _, _, events, nodes = audit.load_sources(1)
    changed = deepcopy(events)
    changed.remove(next(x for x in changed if x.get("type") == "score" and x["boundary"] == 5))
    with pytest.raises(ValueError):
        audit.verify_cem_node(changed, nodes[1], 5)
    changed_nodes = deepcopy(nodes)
    changed_nodes[1]["receipt"]["commands"][0][0] += 0.1
    with pytest.raises(ValueError):
        audit.verify_cem_node(events, changed_nodes[1], 5)
