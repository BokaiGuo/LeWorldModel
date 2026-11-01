"""Pure score selector fixtures; no CPU neural experiment."""
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_native_global_word import select_global_word


def scores():
    return [{"proposal_index": i, "immediate_goal_cost": 1. if i == 0 else .5, "refined_goal_cost": .1 if i == 12 else 1.} for i in range(17)]


def test_minimum_is_not_forced_to_be_non_greedy():
    receipt = select_global_word(scores())
    assert receipt["selected_program_index"] == 12
    assert receipt["immediate_model_non_greedy"] is False
    assert receipt["physical_labels_used"] is False


def test_tie_uses_word_index():
    rows = scores()
    rows[0]["refined_goal_cost"] = .1
    assert select_global_word(rows)["selected_program_index"] == 0


def test_physical_labels_do_not_affect_model_decision():
    rows = scores()
    before = select_global_word(rows)
    for r in rows:
        r["qpos_distance"] = 0 if r["proposal_index"] == 1 else 100
    assert select_global_word(rows) == before


@pytest.mark.parametrize("kind", ["partial", "order", "nan", "negative", "margin"])
def test_broken_score_contracts_fail_closed(kind):
    rows = scores()
    if kind == "partial":
        rows.pop()
    elif kind == "order":
        rows.reverse()
    elif kind == "nan":
        rows[0]["refined_goal_cost"] = float("nan")
    elif kind == "negative":
        rows[0]["immediate_goal_cost"] = -1
    with pytest.raises(ValueError):
        select_global_word(rows, non_greedy_margin=-1 if kind == "margin" else 1e-6)
