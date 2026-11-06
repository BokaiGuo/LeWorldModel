from experiments.spectral_reachability.pc_recursive_reachability_v19 import preserve_reachability_winner


def test_best_reachability_already_retained_does_not_reserve_runner_up():
    chosen = [1, 2]
    values = [{"candidate": 1, "value": 0.}, {"candidate": 3, "value": 1.}]
    assert preserve_reachability_winner(chosen, values, 4) == [1, 2]


def test_best_reachability_is_added_if_not_already_retained():
    chosen = [1, 2]
    values = [{"candidate": 3, "value": 0.}, {"candidate": 1, "value": 1.}]
    assert preserve_reachability_winner(chosen, values, 4) == [1, 2, 3]
