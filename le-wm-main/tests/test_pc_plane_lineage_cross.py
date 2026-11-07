import copy

import numpy as np
import pytest

from experiments.pc_plane_lineage_cross import run
from experiments.pc_plane_lineage_cross.audit import verify_record


TASK = {"start": [-.45, .02, .11], "goal": [.47, -.13]}


def test_nominals_are_exact_prefix_of_complete_generator():
    first, params = run.source_chains(TASK, 42)
    all_chains, all_params = run.source_chains(TASK, 42, 128)
    np.testing.assert_array_equal(first, all_chains[:16])
    np.testing.assert_array_equal(params, all_params[:16])
    assert np.any(params[:, 1] != 0)
    for i in range(16):
        expected = run.geometry.generate(TASK, *params[i])[0]
        np.testing.assert_array_equal(first[i], expected)


@pytest.mark.parametrize("count", [0, 129, True, 16.0])
def test_invalid_generator_count_rejected(count):
    with pytest.raises(ValueError, match="integer count"):
        run.source_chains(TASK, 42, count)


def test_fixture_plan_preserves_every_paid_nominal_and_budget():
    row = run.plan_one(TASK, 100, 0)
    for arm in run.ARMS:
        batches = run.batches(row, arm)
        np.testing.assert_array_equal(batches[0]["actions"], row["nominals"])
        assert sum(len(batch["costs"]) for batch in batches) == 128
        assert row["outcomes"][arm]["choice"]["score"] <= min(batches[0]["costs"])
    audited = verify_record(row)
    assert audited["scalar_replays"] == 128 * 11
    assert audited["max_path_error"] < 1e-12
    assert audited["max_formula_error"] < 1e-10


def test_audit_rejects_corrupt_selection_and_unqueried_truth():
    row = run.plan_one(TASK, 100, 1)
    changed = copy.deepcopy(row)
    changed["outcomes"]["cross_interaction"]["choice"]["query_index"] = -1
    with pytest.raises(ValueError, match="incumbent"):
        verify_record(changed)
    changed = copy.deepcopy(row)
    mask = run.core.support((16, 17), row["anchors"])
    i, j = np.argwhere(~mask)[0]
    changed["observed"][i][j] = .01
    with pytest.raises(ValueError, match="unknown cells"):
        verify_record(changed)
