from __future__ import annotations

import pytest

from experiments.decision_benchmark import audit_pointmaze_c3_independent_direct_v25 as audit


def test_pair_ledger_maps_every_opposed_velocity_pair_to_one_cluster() -> None:
    manifest = {
        "new_root_id_range": [34248, 34251],
        "pairs": [{"pair_index": 0}, {"pair_index": 1}],
    }
    assert audit._pairs(manifest) == {
        "pointmaze_v6_episode_34248": "pointmaze_v6_episode_34248",
        "pointmaze_v6_episode_34249": "pointmaze_v6_episode_34248",
        "pointmaze_v6_episode_34250": "pointmaze_v6_episode_34250",
        "pointmaze_v6_episode_34251": "pointmaze_v6_episode_34250",
    }


def test_cluster_conformal_uses_one_maximum_error_per_paired_source() -> None:
    mapping = {
        "pointmaze_v6_episode_0": "pair_0", "pointmaze_v6_episode_1": "pair_0",
        "pointmaze_v6_episode_2": "pair_1", "pointmaze_v6_episode_3": "pair_1",
    }
    clusters = audit._cluster_values(
        {
            "pointmaze_v6_episode_0": .01, "pointmaze_v6_episode_1": .06,
            "pointmaze_v6_episode_2": .02, "pointmaze_v6_episode_3": .03,
        },
        mapping,
    )
    assert clusters == {"pair_0": .06, "pair_1": .03}
    # ceil((2 + 1) * .9) is 3, which exceeds the two source pairs and must
    # fail closed rather than silently treat four correlated roots as enough.
    assert audit._conformal_upper_bound(clusters, alpha=.1)["bound"] is None


def test_cluster_values_rejects_a_half_observed_pair() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        audit._cluster_values({"pointmaze_v6_episode_0": .01}, {"pointmaze_v6_episode_0": "pair_0"})
