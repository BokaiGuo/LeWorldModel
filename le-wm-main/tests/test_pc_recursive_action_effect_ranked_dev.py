import numpy as np

from experiments.spectral_reachability.pc_recursive_action_effect_ranked_dev import ranked_unique_shortlist


def test_ranked_shortlist_deduplicates_actions_and_keeps_stable_score_order():
    candidates = [
        {"chain": np.array([[0.0, 0.0]])},
        {"chain": np.array([[0.0, 0.0]])},
        {"chain": np.array([[0.2, 0.0]])},
        {"chain": np.array([[0.3, 0.0]])},
    ]
    assert ranked_unique_shortlist([0.1, 0.1, 0.1, 0.2], candidates, 3) == [0, 2, 3]
