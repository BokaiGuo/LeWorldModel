import numpy as np

from experiments.spectral_reachability import p1_dev305_two_packet_metric_decompose_20261102 as source
from experiments.spectral_reachability import p1_dev305_two_packet_reward_verify_20261102 as reward


def test_pair_counts_include_ties_and_reversals():
    result = source.pair_counts(np.array([0., 1., 2., 2.]), np.array([0., 2., 1., 1.]), 1e-6)
    assert result == {"same": 3, "reversed": 2, "ties": 1}


def test_native_reward_against_all_saved_transitions():
    result = reward.analyze()
    assert result["physical_sequences"] == 867
    assert result["transitions"] == 8670
    assert result["max_reward_absolute_error"] <= 1e-9
    assert result["native_success_transitions"] == 0
