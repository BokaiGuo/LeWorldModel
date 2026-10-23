from __future__ import annotations

from experiments.preflight_pointmaze_candidate_anchored_probe import _summarize


def test_preflight_summary_requires_exact_bank_membership_and_rank_three_geometry() -> None:
    report = _summarize([{
        "configurations": [
            {
                "probe_direction_cap": 4,
                "probe_magnitude_cap": 0.25,
                "selected_probe_count": 3,
                "affine_design_rank": 3,
                "candidate_membership_exact": True,
                "repairability_geometry_available": True,
            },
            {
                "probe_direction_cap": 4,
                "probe_magnitude_cap": 0.5,
                "selected_probe_count": 0,
                "affine_design_rank": 0,
                "candidate_membership_exact": True,
                "repairability_geometry_available": False,
            },
        ],
    }])
    small, large = report["configurations"]
    assert small["geometry_available_rate"] == 1.0
    assert small["exact_bank_membership_rate"] == 1.0
    assert large["geometry_available_root_count"] == 0
