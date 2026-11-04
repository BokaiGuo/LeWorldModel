import numpy as np
import pytest

from experiments.spectral_reachability import reacher_p1_52_q_metric_analysis_20261102 as analysis


def test_goal_cometric_matches_closed_form():
    root = analysis.metric_sqrt([[0.0, 0.0], [0.0, 0.99]])
    np.testing.assert_allclose(root, np.diag([1.0, 0.1]), atol=1e-12)


@pytest.mark.parametrize("gram", [[[1, 2], [0, 1]], [[-1, 0], [0, 1]], [[float("nan")]]])
def test_bad_gram_rejected(gram):
    with pytest.raises(ValueError):
        analysis.metric_sqrt(gram)
