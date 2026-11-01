import numpy as np
import pytest

from experiments.spectral_reachability.p1_cp10_normalizer_rank_saved_20261031 import check_tokens


def test_all_history_and_candidate_tokens_are_bound():
    raw = np.arange(30, dtype=np.float32).reshape(3, 5, 2) / 10
    mean = np.array([0.2, -0.1], dtype=np.float32)
    std = np.array([0.3, 0.4], dtype=np.float32)
    tokens = ((raw - mean) / std).reshape(3, 10)
    check_tokens(tokens, raw, mean, std)
    for packet in range(3):
        wrong = tokens.copy()
        wrong[packet, 0] += 0.1
        with pytest.raises(ValueError, match="token binding"):
            check_tokens(wrong, raw, mean, std)
