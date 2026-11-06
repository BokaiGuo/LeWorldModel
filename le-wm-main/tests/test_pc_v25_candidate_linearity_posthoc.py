import numpy as np

from experiments.pc_v25_candidate_linearity_posthoc_20261104_codex import candidate_error, pairwise


def test_linear_error_and_zero_response_are_handled():
    base = np.zeros((1, 2))
    matrix = np.array([[1, 2], [0, 1], [1, 0]], dtype=float)
    action = np.array([[.5, -.25]])
    feature = matrix @ action.reshape(-1)
    row = candidate_error(base, np.zeros(3), matrix, action, feature)
    assert row["position_absolute_error"] == 0
    assert row["score_absolute_error"] == 0
    zero = candidate_error(base, np.zeros(3), np.zeros((3, 2)), action, np.zeros(3))
    assert zero["position_relative_error"] is None


def test_pairwise_ties_and_discordance():
    records = [{"predicted": [0, 0, a], "observed": [0, 0, b]}
               for a, b in [(1, 1), (2, 0), (2, 3)]]
    assert pairwise(records) == {"concordant": 1, "discordant": 1, "tied": 1}
