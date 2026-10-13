import math

from experiments.empirical_spectral_audit.run_nonlinear_composition_aware_training import (
    run_report,
)
from experiments.empirical_spectral_audit.run_nonlinear_multihorizon_training_ablation import (
    TrainingConfig,
)


def test_composition_aware_smoke_has_paired_conditions():
    config = TrainingConfig(
        train_sequences=64,
        validation_sequences=32,
        test_sequences=64,
        epochs=3,
    )
    report = run_report(config, [0, 1], composition_weight=1.0)
    assert report["status"] == "complete"
    assert {row["condition"] for row in report["records"]} == {
        "one_step",
        "multi_horizon",
        "composition_aware",
    }
    assert len(report["records"]) == 6
    value = report["records"][0]["metrics"]["composition_switch_mse"]
    assert math.isfinite(value)
    assert "composition_aware_vs_multi_horizon" in report["paired_inference_vs_one_step"]

