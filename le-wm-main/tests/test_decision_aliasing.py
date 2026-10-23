from __future__ import annotations

import numpy as np
import pytest

from experiments.decision_benchmark.decision_aliasing import (
    calibrate_representation_radius,
    empirical_decision_aliasing_audit,
)


def test_near_aliasing_is_root_disjoint_empirical_evidence_not_a_theorem() -> None:
    report = empirical_decision_aliasing_audit(
        np.asarray([[0.0, 0.0], [3.0, 3.0]]), np.asarray([[0.0, 1.0], [0.0, 0.0]]), np.asarray(["fit-a", "fit-b"]),
        np.asarray([[0.01, 0.0]]), np.asarray([[2.0, 1.0]]), np.asarray(["outcome-a"]),
        representation_radius=0.1, contrast_gap_threshold=1.0,
    )
    assert report["reference_query_root_disjoint"] is True
    assert report["near_decision_alias_count"] == 1
    assert report["max_exact_collision_half_gap_lower_bound"] is None
    assert "not a global identifiability lower bound" in report["claim_boundary"]


def test_exact_collision_exposes_the_half_gap_interface_lower_bound() -> None:
    report = empirical_decision_aliasing_audit(
        np.asarray([[0.0, 0.0], [2.0, 2.0]]), np.asarray([[0.0], [1.0]]), np.asarray(["fit-a", "fit-b"]),
        np.asarray([[0.0, 0.0]]), np.asarray([[2.0]]), np.asarray(["outcome-a"]),
        representation_radius=0.0, contrast_gap_threshold=0.5,
    )
    assert report["exact_collision_with_contrast_gap_count"] == 1
    assert report["max_exact_collision_half_gap_lower_bound"] == 1.0


def test_radius_calibration_uses_only_fit_representations_and_cross_root_neighbors() -> None:
    report = calibrate_representation_radius(
        np.asarray([[0.0], [0.1], [1.0]]), np.asarray(["a", "a", "b"]), quantile=0.5,
    )
    assert report["fit_row_count"] == 3
    assert report["cross_root_neighbor_count"] == 3
    assert report["representation_radius"] > 0.0


def test_alias_audit_rejects_root_overlap_between_reference_and_query_panels() -> None:
    with pytest.raises(ValueError, match="root panels must be disjoint"):
        empirical_decision_aliasing_audit(
            np.asarray([[0.0]]), np.asarray([[0.0]]), np.asarray(["same"]),
            np.asarray([[0.0]]), np.asarray([[1.0]]), np.asarray(["same"]),
            representation_radius=0.1, contrast_gap_threshold=0.1,
        )
