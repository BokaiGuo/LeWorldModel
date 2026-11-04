import numpy as np
import pytest

from experiments.spectral_reachability import pc_pointmaze_velocity_score_crosstab_cuda_20261103_codexexclusive as c


def test_pairwise_ranks_and_ties():
    x = np.arange(48, dtype=float)
    assert c.paired_rank(x, x)["reversed"] == 0
    assert c.paired_rank(x, -x)["reversed"] == 1128
    ties = x.copy()
    ties[1] = ties[0]
    assert c.paired_rank(ties, x)["score_ties"] == 1
    with pytest.raises(ValueError):
        c.paired_rank(x[:-1], x)
    x[0] = np.nan
    with pytest.raises(ValueError):
        c.paired_rank(x, x)


def test_allclose_rejects_mismatch_and_nonfinite():
    assert c.require_close([1., 2.], [1., 2.], "test") == 0
    with pytest.raises(ValueError):
        c.require_close([1., 2.], [1., 3.], "test")
    with pytest.raises(ValueError):
        c.require_close([np.nan], [1.], "test")


def test_locked_source_roster_and_output_boundary():
    plan = c.source_plan()
    assert len(plan["groups"]) == 8 and plan["candidates_per_group"] == 48
    assert plan["modes"] == ["zero", "causal", "privileged"]
    assert plan["P1_admission"] is False
