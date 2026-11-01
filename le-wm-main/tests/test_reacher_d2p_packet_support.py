import numpy as np
import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_packet_support import (
    SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION,
    proposal_parameters,
    sample_history_packets,
    sample_packets,
)


def test_reacher_support_sampling_is_seeded_and_keeps_time_major_packets():
    pool = np.arange(8 * 5 * 2, dtype=np.float32).reshape(8, 5, 2)
    first, first_indices = sample_packets(pool, count=3, seed=12, stream=7)
    second, second_indices = sample_packets(pool, count=3, seed=12, stream=7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first, pool[first_indices])


def test_reacher_support_sampling_rejects_invalid_count():
    with pytest.raises(ValueError):
        sample_packets(np.zeros((2, 5, 2), dtype=np.float32), count=3, seed=1, stream=1)


def test_reacher_source_proposals_keep_raw_zero_and_preserve_sampled_packets():
    pool = np.linspace(-0.5, 0.5, 8 * 5 * 2, dtype=np.float32).reshape(8, 5, 2)
    params, receipt = proposal_parameters(pool, proposals=4, horizon=1, seed=4, mean=np.zeros(10, dtype=np.float32), std=np.ones(10, dtype=np.float32), lower=torch.full((10,), -1.0), upper=torch.full((10,), 1.0))
    assert torch.count_nonzero(params[0]) == 0
    assert receipt["proposal_distribution"] == SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION + "_with_raw_zero_at_index_0"
    recovered = torch.tanh(params[1:]).cpu().numpy().reshape(3, 5, 2)
    np.testing.assert_allclose(recovered, pool[np.asarray(receipt["proposal_source_indices"]).reshape(-1)], atol=1.0e-6)


def test_reacher_contiguous_history_sampling_never_mixes_source_sequences():
    pool = np.arange(4 * 3 * 5 * 2, dtype=np.float32).reshape(4, 3, 5, 2)
    sampled, index = sample_history_packets(pool, seed=23)
    np.testing.assert_array_equal(sampled, pool[index])
