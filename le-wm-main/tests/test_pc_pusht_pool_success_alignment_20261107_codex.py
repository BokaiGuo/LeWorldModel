"""Integrity checks for the completed, locally generated diagnostic artifact."""
import json

import numpy as np
import pytest

from experiments.pc_input_permissions import file_hash
from experiments.pc_pusht_pool_success_alignment_20261107_codex import OUT


@pytest.mark.skipif(not OUT.with_suffix(".json").exists(), reason="diagnostic result not generated")
def test_saved_representations_and_task_grouping():
    result = json.loads(OUT.with_suffix(".json").read_text())
    assert file_hash(OUT.with_suffix(".npz")) == result["arrays_sha256"]
    data = np.load(OUT.with_suffix(".npz"), allow_pickle=False)
    predicted, actual, goal = data["predicted"], data["actual"], data["goal"]
    assert predicted.shape == actual.shape == (99, 5, 192)
    assert goal.shape == (99, 192)
    for i, row in enumerate(result["rows"]):
        np.testing.assert_allclose(((predicted[i] - goal[i]) ** 2).sum(-1), row["predicted_costs"])
        np.testing.assert_allclose(((actual[i] - goal[i]) ** 2).sum(-1), row["actual_encoded_costs"])
        assert len(row["successes"]) == 25
    for task, expected_count in enumerate((81, 14, 2, 2)):
        rows = [x for x in result["rows"] if x["task"] == task]
        assert len(rows) == expected_count
        assert sum(x["selected"] for x in rows) == 1
        assert all(x["selected"] or any(x["successes"]) for x in rows)
