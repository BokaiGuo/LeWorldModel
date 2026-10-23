from pathlib import Path

import numpy as np

from experiments.run_pusht_balanced_temporal_torque_confirmation import (
    offline_preference,
    signed_offline_preference,
)
from experiments.run_pusht_balanced_temporal_torque_confirmation import BalancedCohortUnavailable
from experiments.run_pusht_signed_temporal_torque_confirmation import (
    contact_frame_action_pair,
    write_pre_outcome_rejection,
)


def test_signed_torque_label_preserves_direction_lost_by_absolute_sensitivity() -> None:
    # +y produces a smaller but positive response; -y produces a larger
    # negative response.  Absolute sensitivity chooses -y, while a controller
    # targeting a positive turn must learn the signed +y preference.
    absolute = offline_preference(
        before_angle=0.0, zero_angle=0.1, plus_angle=0.4, minus_angle=-0.3,
        minimum_margin=1.0e-4,
    )
    signed = signed_offline_preference(
        before_angle=0.0, zero_angle=0.1, plus_angle=0.4, minus_angle=-0.3,
        minimum_margin=1.0e-4,
    )
    assert absolute[0] == 0
    assert signed[0] == 1
    assert signed[2] > 0.0 > signed[3]


def test_signed_torque_launcher_is_fresh_cuda_only() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "run_pusht_signed_temporal_torque_confirmation_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "run_with_torch_cudnn.sh .venv/bin/python -m" in text
    assert "run_pusht_signed_temporal_torque_confirmation" in text
    assert "Refusing to reuse immutable protocol output" in text
    assert "--train-per-class 64 --calibration-per-class 32 --test-per-class 64" in text
    assert "contact_frame_v2" in text
    assert "--contact-inward-magnitude 0.18 --contact-tangent-magnitude 0.55" in text


def test_contact_frame_pair_has_equal_norm_and_opposite_moments() -> None:
    actions, audit = contact_frame_action_pair(
        {"contact_radius": np.asarray([3.0, 4.0]), "inward": np.asarray([-0.6, -0.8])},
        inward_magnitude=0.18,
        tangent_magnitude=0.55,
    )
    radius = np.asarray([3.0, 4.0])
    assert audit["pair_valid"]
    assert np.isclose(np.linalg.norm(actions["plus"]), np.linalg.norm(actions["minus"]))
    assert float(radius[0] * actions["plus"][1] - radius[1] * actions["plus"][0]) > 0.0
    assert float(radius[0] * actions["minus"][1] - radius[1] * actions["minus"][0]) < 0.0


def test_signed_runner_persists_pre_outcome_cohort_rejection_without_score(tmp_path) -> None:
    error = BalancedCohortUnavailable(
        "test seed band has 0 eligible roots for label 1",
        collection={"split": "test", "eligible_by_label_before_sampling": {"0": 64, "1": 0}},
    )
    write_pre_outcome_rejection(
        tmp_path / "rejection",
        runtime={"device": "cuda"},
        bands={"train": 1, "calibration": 2, "test": 3},
        collection={"train": {"selected_roots": 128}, "test": error.collection},
        error=error,
        started=0.0,
    )
    import json

    report = json.loads((tmp_path / "rejection" / "signed_temporal_torque_confirmation.json").read_text())
    assert report["status"] == "rejected_pre_outcome_balanced_cohort"
    assert not report["sanity_checks"]["decoder_fit_attempted"]
    assert not report["sanity_checks"]["outcome_score_materialized"]
    assert "results" not in report
