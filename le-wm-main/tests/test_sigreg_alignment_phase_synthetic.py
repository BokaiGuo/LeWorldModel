import math

from experiments.empirical_spectral_audit.run_pixel_world_model_synthetic import (
    PixelBenchmarkConfig,
    run_one,
)
from experiments.empirical_spectral_audit.run_sigreg_alignment_phase_synthetic import (
    _summary,
)


def test_phase_summary_separates_sigreg_and_alignment_conditions():
    config = PixelBenchmarkConfig(
        train_samples=16,
        validation_samples=8,
        test_samples=16,
        rollout_trajectories=8,
        epochs=2,
        sigreg_num_proj=8,
    )
    records = []
    for sigreg_weight in (0.0, 0.1):
        record = run_one("conditional", 0.1, 0, config)
        record["sigreg_weight"] = sigreg_weight
        records.append(record)
    summary = _summary(records)
    assert set(summary) == {"0.0:conditional:0.1", "0.1:conditional:0.1"}
    for values in summary.values():
        assert values["n_seeds"] == 1
        assert math.isfinite(values["stable_rank"]["mean"])
        assert math.isfinite(values["conditional_commutator"]["mean"])


def test_pixel_phase_record_contains_isotropy_and_functional_readouts():
    config = PixelBenchmarkConfig(
        train_samples=16,
        validation_samples=8,
        test_samples=16,
        rollout_trajectories=8,
        epochs=2,
        sigreg_num_proj=8,
    )
    record = run_one("global", 0.1, 1, config)
    assert math.isfinite(record["test_one_step_mse"])
    assert set(record["rollout_mse"]) == {"1", "2", "4", "8"}
    assert math.isfinite(record["action_composition_error_mean"])
    assert record["diagnostics"]["stable_rank"] > 0
