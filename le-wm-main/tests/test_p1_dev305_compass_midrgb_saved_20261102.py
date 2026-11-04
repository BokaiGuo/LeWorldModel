from experiments.spectral_reachability.p1_dev305_compass_midrgb_saved_20261102 import pair_counts


def test_pair_counts_keep_ties_and_reverse():
    rows = [
        {"model": 0.0, "actual_goal_mse": 0.0},
        {"model": 1.0, "actual_goal_mse": 2.0},
        {"model": 2.0, "actual_goal_mse": 1.0},
        {"model": 2.0, "actual_goal_mse": 1.0},
    ]
    assert pair_counts(rows, "model") == {"same": 3, "reversed": 2, "ties": 1}


def test_pair_counts_apply_tolerance_to_both_metrics():
    rows = [
        {"model": 0.0, "actual_goal_mse": 0.0},
        {"model": 0.5e-6, "actual_goal_mse": 2.0},
        {"model": 2.0, "actual_goal_mse": 0.5e-6},
    ]
    assert pair_counts(rows, "model", 1e-6) == {"same": 0, "reversed": 1, "ties": 2}
