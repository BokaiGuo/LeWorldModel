import numpy as np

from experiments.run_pusht_candidate_rank_diagnostic import correlation


def test_rank_correlation_detects_monotone_order():
    assert correlation(np.arange(8.0), np.arange(8.0)) > 0.99
