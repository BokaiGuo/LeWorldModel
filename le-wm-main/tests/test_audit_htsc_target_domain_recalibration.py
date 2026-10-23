from __future__ import annotations

import pytest

from experiments.decision_benchmark.audit_htsc_target_domain_recalibration import target_domain_recalibration


def _source(errors: list[float]) -> dict[str, object]:
    return {
        "schema": "htsc.finite_sample_calibration_audit.v1",
        "independent_external_root_coverage": {
            "roots": [
                {"group_id": f"target_{index:02d}", "max_contrast_error": error}
                for index, error in enumerate(errors)
            ]
        },
    }


def test_target_recalibration_uses_only_chronological_prefix_for_threshold() -> None:
    # n=9 at alpha=.1 uses the ninth order statistic, so the threshold is
    # frozen before the final evaluation root is read.
    result = target_domain_recalibration(
        _source([.1, .2, .3, .4, .5, .6, .7, .8, .9, .2]),
        target_calibration_root_count=9, alpha=.1,
    )
    assert result["status"] == "evaluated"
    assert result["target_calibration"]["bound"] == pytest.approx(.9)
    assert result["target_outcome"]["root_ids"] == ["target_09"]
    assert result["target_outcome"]["empirical_root_coverage"] == 1.0
    assert result["target_split_contract"]["selection_uses_target_outcome_errors"] is False


def test_target_recalibration_fails_closed_when_prefix_is_too_small() -> None:
    result = target_domain_recalibration(
        _source([.1] * 10), target_calibration_root_count=8, alpha=.1,
    )
    assert result["status"] == "fail_closed_insufficient_target_calibration"
    assert result["target_calibration"]["bound"] is None
    assert result["target_outcome"]["empirical_root_coverage"] is None
