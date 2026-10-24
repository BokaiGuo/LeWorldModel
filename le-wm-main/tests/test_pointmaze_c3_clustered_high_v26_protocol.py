from __future__ import annotations

from experiments.gdd_wm import preflight_pointmaze_c3_clustered_high_v26 as preflight


def test_v26_high_motion_pairs_are_explicit_conformal_clusters() -> None:
    assert preflight.POOL_SCHEMA == "pointmaze_mixed_velocity_repairability_pool.v3_clustered_high"
    groups = preflight._high_groups()
    assert groups[0] == (35080, 35081)
    assert groups[-1] == (35526, 35527)


def test_v26_split_keeps_pairs_intact_but_reports_64_independent_high_units() -> None:
    split, design, counts = preflight._assign(202611923, {"train": 256, "calibration": 256, "outcome": 256})
    assert len(split) == 768 and len(design) == 64
    for left, right in preflight._high_groups():
        if left in split:
            assert split[left] == split[right]
    assert all(value["high_motion_independent_source_pairs"] == 64 for value in counts.values())
