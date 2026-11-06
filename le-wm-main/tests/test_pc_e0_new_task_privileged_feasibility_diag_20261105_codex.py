import numpy as np
from experiments.pc_e0_new_task_privileged_feasibility_diag_20261105_codex import witness


def test_geometry_witness_is_bounded_and_not_truncated():
    n, chain = witness({"start": [-.5, .14, .8], "goal": [.5, .07]}, 1)
    assert n <= 15 and chain.shape == (15, 2)
    assert np.isfinite(chain).all() and np.abs(chain).max() <= 1
    n, chain = witness({"start": [-.5, 50., .8], "goal": [.5, .07]}, -1)
    assert n > 15 and chain is None
