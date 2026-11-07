import numpy as np
import pytest

from experiments.pc_tworoom_temporal_retention_fixed_pool_20261107_codex import coverage, keep


def test_retention_stable_and_rejects_bad_scores():
    assert keep([[4, 2, 3], [2, 5, 6], [1, 9, 9]]) == [2, 0]
    with pytest.raises(ValueError):
        keep([[np.nan] * 3] * 3)


def test_strict_tail_success_excludes_prefix_and_initial_success():
    def point(success):
        return {"success": success, "distance": 1 if success else 20}
    entry = {"prefix": 1, "remaining": 2, "initial_success": False,
             "physical": [[point(True)], [point(False), point(True)],
                          [point(False), point(False)], [point(False), point(False)],
                          [point(False), point(False)], [point(False), point(False)]]}
    assert coverage(entry, [0, 1]) == {"ever": True, "tail": True}
    assert coverage(entry, [1, 2]) == {"ever": False, "tail": False}
    entry["initial_success"] = True
    assert coverage(entry, [0, 1]) == {"ever": True, "tail": False}
