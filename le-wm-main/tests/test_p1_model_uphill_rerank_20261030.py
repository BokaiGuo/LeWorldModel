import pytest

from experiments.spectral_reachability import p1_model_uphill_rerank_20261030 as rerank


def test_feasibility_ties_and_no_fallback():
    records = [
        {"candidate_index": 0, "iteration": 0, "prefix_mse": 0.9, "terminal_mse": 0.01},
        {"candidate_index": 2, "iteration": 0, "prefix_mse": 1.0, "terminal_mse": 0.3},
        {"candidate_index": 1, "iteration": 2, "prefix_mse": 1.1, "terminal_mse": 0.3},
    ]
    assert rerank.choose(records, 1.0) == records[2]
    assert rerank.choose(records, 2.0) is None


def test_score_validation():
    assert rerank.mse([1.0] * 192, [0.0] * 192) == 1.0
    with pytest.raises(ValueError, match="finite"):
        rerank.mse([float("nan")] * 192, [0.0] * 192)
    with pytest.raises(ValueError, match="shape"):
        rerank.mse([0.0], [0.0])
