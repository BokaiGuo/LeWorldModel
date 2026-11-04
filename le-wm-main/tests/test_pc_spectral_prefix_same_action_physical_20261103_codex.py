from experiments.pc_spectral_prefix_same_action_physical_20261103_codex import reversals


def test_reversals_count_exact_ties_separately():
    assert reversals([0, 1, 2], [2, 1, 0]) == {"reversed": 3, "comparable": 3, "ties": 0}
    assert reversals([0, 0, 1], [0, 1, 1]) == {"reversed": 0, "comparable": 1, "ties": 2}
