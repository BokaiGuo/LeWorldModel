import numpy as np

from experiments.pc_v25_same_chain_search_loss_posthoc_20261104_codex import add_batch


def test_stage_cost_and_full_horizon_best():
    stages = {}
    add_batch(stages, "probe", np.zeros((3, 5, 2)), np.array([4., 2., 3.]))
    add_batch(stages, "probe", np.zeros((2, 15, 2)), np.array([5., 1.]))
    add_batch(stages, "probe", np.zeros((1, 15, 2)), np.array([2.]))
    assert stages["probe"] == {"calls": 3, "samples": 6, "tokens": 12, "best_full_score": 1.}
