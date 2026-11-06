import numpy as np

from experiments.spectral_reachability.pc_response_retention_e0 import ridge_residual, select


def test_ridge_value_and_stable_selector():
    matrix = np.asarray([[1., 0.], [0., 0.]])
    assert ridge_residual(matrix, [1., 0.]) < ridge_residual(matrix, [0., 1.])
    scores = np.arange(5.)
    state = np.arange(5.)[:, None]
    responses = np.arange(5.)[:, None, None]
    chosen = select(scores, state, responses, [4., 3., 2., 1., 0.], [4., 3., 2., 1., 0.], seed=12)
    assert chosen["greedy"] == [0, 1, 2, 3]
    assert chosen["response"][:2] == [0, 4]
    assert chosen["lookahead"] == [4, 3, 2, 1]
    assert chosen == select(scores, state, responses, [4., 3., 2., 1., 0.], [4., 3., 2., 1., 0.], seed=12)
    assert all(len(set(indices)) == 4 for indices in chosen.values())
