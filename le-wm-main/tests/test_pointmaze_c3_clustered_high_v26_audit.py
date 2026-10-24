from __future__ import annotations

import pytest

from experiments.decision_benchmark import audit_pointmaze_c3_clustered_high_v26 as audit


def test_v26_pair_ledger_requires_two_roots_per_source_pair() -> None:
    assert audit._paired_clusters({"new_root_id_range": [10, 13], "pairs": [{"pair_index": 0}, {"pair_index": 1}]}) == {
        "pointmaze_v6_episode_10": "pointmaze_v6_episode_10",
        "pointmaze_v6_episode_11": "pointmaze_v6_episode_10",
        "pointmaze_v6_episode_12": "pointmaze_v6_episode_12",
        "pointmaze_v6_episode_13": "pointmaze_v6_episode_12",
    }


def test_v26_pair_ledger_rejects_bad_range() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        audit._paired_clusters({"new_root_id_range": [10, 12], "pairs": [{"pair_index": 0}, {"pair_index": 1}]})
