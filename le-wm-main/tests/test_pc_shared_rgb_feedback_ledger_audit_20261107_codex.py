import numpy as np
import pytest

from experiments.pc_shared_rgb_feedback_ledger_audit_20261107_codex import reconstruct
from experiments.pc_shared_rgb_feedback_smoke_20261107_codex import action_hash, build_candidates


def test_reconstruct_parent_and_detect_changed_score():
    seed = 123
    parents, _, _ = build_candidates(seed, None)
    item = {"candidate_seed": seed, "parent_cost": [1, 2, 3], "probe_cost": [[5, 6], [7, 8], [9, 10]],
            "tail_cost": [11, 12], "retained": [0], "queries": 11, "forward_calls": 3,
            "choice": {"source": "parent", "index": 0, "cost": 1},
            "actions": parents[0].tolist(), "actions_sha256": action_hash(parents[0])}
    np.testing.assert_array_equal(reconstruct(item, None), parents[0])
    item["parent_cost"][1] = 0
    with pytest.raises(ValueError, match="identity"):
        reconstruct(item, None)
