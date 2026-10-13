import json

import torch

from experiments.empirical_spectral_audit.run_nonlinear_uncertainty_conditioned_horizon_ablation import (
    _adaptive_horizon_weights,
    run_report,
)
from experiments.empirical_spectral_audit.run_nonlinear_multihorizon_training_ablation import TrainingConfig


def test_adaptive_horizon_weights_prioritize_short_horizon_at_high_uncertainty():
    low = _adaptive_horizon_weights(torch.tensor([0.0]))
    high = _adaptive_horizon_weights(torch.tensor([1.0]))
    assert high[1].item() > low[1].item()
    assert high[4].item() < low[4].item()
    assert high[4].item() > 0.0


def test_uncertainty_conditioned_report_has_paired_matrix_and_boundary(tmp_path):
    output = tmp_path / "uncertainty.json"
    config = TrainingConfig(
        train_sequences=32,
        validation_sequences=16,
        test_sequences=32,
        epochs=4,
        planning_tasks=4,
        planning_candidates=6,
    )
    report = run_report(config, seeds=[0, 1], probe_members=2, probe_epochs=2)
    output.write_text(json.dumps(report))
    assert output.is_file()
    assert report["protocol"]["probe_fit_uses_training_rows_only"] is True
    assert report["protocol"]["same_data_and_initialization_across_profiles"] is True
    assert "ensemble_conditioned:planning_regret:shift=1.5:h=4" in report["paired_inference_vs_one_step"]
    assert "planning_regret:shift=1.5:h=4" in report["paired_inference_conditioned_vs_permuted"]
    assert len(report["selective_diagnostics"]) == 2 * 5 * 2
    assert "not end-to-end LeWM" in report["claim_boundary"][0]
