import numpy as np

from experiments.audit_conditional_transfer_selection import audit


def _record(checkpoint: str, stage: int, mse: float, spectrum_shift: float, loss: float, alignment: float) -> dict:
    spectrum = [
        {"real": 0.8 + spectrum_shift, "imag": 0.0},
        {"real": 0.5, "imag": 0.0},
        {"real": 0.2, "imag": 0.0},
    ]
    diagnostics = {
        "prediction_mse": mse,
        "predictor_spectrum": spectrum,
        "task_metric_alignment": alignment,
        "task_metric_commutator": alignment,
        "task_metric_departure_from_normality": alignment,
        "task_metric_transient_amplification": alignment,
        "task_metric_spectral_radius": 0.8 + spectrum_shift,
    }
    return {
        "group": f"PushT/in_distribution/arch/seed_{checkpoint}",
        "checkpoint": f"arch/seed_{checkpoint}/epoch_{stage}",
        "model_factors": {
            "latent_dimension": 96,
            "training_stage": stage,
        },
        "diagnostics": diagnostics,
        "heldout": {"planning_loss": loss},
        "protocol": {
            "environment": "swm/PushT-v1",
            "coordinate_gauge": "decoded_observable_pose_raw",
            "prediction_quality": {
                "action_distribution": "fixed-actions",
            },
        },
    }


def test_requires_two_fixed_spectrum_quality_matched_candidates():
    report = audit([
        _record("1", 10, 1.00, 0.0, 0.8, 0.9),
        _record("2", 10, 1.02, 0.0, 0.2, 0.1),
        _record("3", 10, 1.01, 0.2, 0.1, 0.3),
    ])
    summary = report["summary"]
    assert summary["identifiable_strata"] == 1
    assert report["strata"][0]["eligible_count"] == 2
    assert report["strata"][0]["selector_indices"]["task_metric_alignment"] == 1


def test_nonmatched_spectrum_is_fail_closed():
    report = audit([
        _record("1", 10, 1.00, 0.0, 0.8, 0.9),
        _record("2", 10, 1.02, 0.2, 0.1, 0.1),
    ])
    assert report["summary"]["identifiable_strata"] == 0
    assert report["strata"][0]["selector_indices"]["task_metric_alignment"] is None
