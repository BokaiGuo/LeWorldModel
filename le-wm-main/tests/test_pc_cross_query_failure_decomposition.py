import numpy as np

from experiments.pc_cross_query import core
from experiments import pc_cross_query_failure_decomposition_20261107_codex as audit


def test_alternative_lstsq_reconstructs_rank_two_without_truth():
    rng = np.random.default_rng(564)
    truth = rng.normal(size=(8, 1)) + rng.normal(size=(1, 32)) + rng.normal(size=(8, 2)) @ rng.normal(size=(2, 32))
    anchors = core.anchor_rows(truth[:, :4])
    observed = np.where(core.cross_mask(anchors), truth, np.nan)
    np.testing.assert_allclose(audit.alternative_prediction(observed, anchors), truth, atol=1e-10)
