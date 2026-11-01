import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p3_raw_chart_nonlinear_20261030_main as exp


def test_fixed_roster_and_saved_cost():
    roster = exp.load_roster()
    assert len(roster) == 72
    assert tuple(dict.fromkeys(row[0] for row in roster)) == exp.ROSTER
    for _, _, task, _, row in roster:
        assert exp.projected_cost(task, row["suffix_path"][-1]["latent_history"][-1]) == pytest.approx(
            row["bounded_suffix_cost"], abs=1e-12)


def test_projected_cost_rejects_nonfinite():
    task = {"projection": [[1.0]], "goal": [0.0]}
    assert exp.projected_cost(task, [2.0]) == 4.0
    with pytest.raises(ValueError, match="nonfinite"):
        exp.projected_cost(task, [np.nan])


def test_context_digest_detects_edit():
    roster = exp.load_roster()
    row = copy.deepcopy(roster[0][4])
    assert exp.digest(row["postcontext"]) == row["signal"]["parent_context_sha256"]
    row["postcontext"]["latent_history"][0][0] += 1
    assert exp.digest(row["postcontext"]) != row["signal"]["parent_context_sha256"]
