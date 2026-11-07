import numpy as np

from experiments.pc_wall_crossing_tail_intervention_20261107_codex import choose


def test_choose_earliest_crossing_not_lowest_cost_non_crossing():
    task = {"start": [-.1, 1.1, 0.], "goal": [.5, 1.1]}
    chains = np.zeros((3, 15, 2))
    chains[0, 1, 0] = 1
    chains[1, 0, 0] = 1
    record = {"actions": chains.tolist(), "costs": [.2, .4, .0]}
    selected = choose(task, [("bank", [record])])
    assert selected[0] == 1
    assert selected[4] == 1


def test_no_crossing_returns_none():
    task = {"start": [-.1, 0., 0.], "goal": [.5, 0.]}
    assert choose(task, [("bank", [{"actions": np.zeros((1, 15, 2)).tolist(), "costs": [.6]}])]) is None
