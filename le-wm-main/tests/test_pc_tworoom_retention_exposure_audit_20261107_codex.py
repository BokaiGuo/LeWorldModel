from experiments.pc_tworoom_retention_exposure_audit_20261107_codex import compare


def decision(iteration, actions, dispatched, retained):
    return {"iteration": iteration, "actions": actions, "dispatched": dispatched,
            "retained": retained, "candidate_seed": 10 + iteration}


def test_different_old_tail_is_not_counted_as_same_search_state():
    first_a = decision(0, [1, 2], [1], [0, 1])
    first_b = decision(0, [1, 3], [1], [0, 2])
    second_a = decision(1, [4], [4], [0, 1])
    second_b = decision(1, [5], [5], [0, 2])
    rows = [{"episode": 4, "arms": {"lookahead": {"decisions": [first_a, second_a]},
                                    "effect": {"decisions": [first_b, second_b]}}}]
    counts, events = compare(rows)
    assert counts["same_dispatched_history"] == 2
    assert counts["same_history_and_old_tail"] == 1
    assert counts["different_retained"] == 1
    assert counts.get("different_dispatch", 0) == 0
    assert events == []


def test_stops_comparison_after_dispatch_diverges():
    a = [decision(0, [1], [1], [0, 1]), decision(1, [2], [2], [0, 1])]
    b = [decision(0, [3], [3], [0, 2]), decision(1, [4], [4], [0, 2])]
    counts, events = compare([{"episode": 7, "arms": {"lookahead": {"decisions": a},
                                                       "effect": {"decisions": b}}}])
    assert counts["same_dispatched_history"] == 1
    assert counts["different_dispatch"] == 1
    assert events == [[7, 0]]
