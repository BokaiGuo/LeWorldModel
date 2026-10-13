import pytest

from experiments.real_checkpoint_selection_protocol import evaluate_records


def _record(group: str, checkpoint: str, *, alignment: float, mse: float, loss: float) -> dict:
    return {
        "group": group,
        "checkpoint": checkpoint,
        "diagnostics": {
            "prediction_mse": mse,
            "alignment": alignment,
            "normalized_commutator": alignment,
            "spectral_radius": 1.0 + alignment / 10.0,
            "latent_variance": 1.0 / (1.0 + alignment),
            "schur_offdiagonal_energy": alignment,
            "departure_from_normality": alignment,
            "transient_amplification": 1.0 + alignment,
            "pseudospectral_sensitivity": 1.0 + alignment,
        },
        "heldout": {"planning_loss": loss},
    }


def _task_record(group: str, checkpoint: str, *, alignment: float, mse: float, task_alignment: float, loss: float) -> dict:
    record = _record(group, checkpoint, alignment=alignment, mse=mse, loss=loss)
    record["diagnostics"].update({
        "task_metric_alignment": task_alignment,
        "task_metric_commutator": task_alignment,
        "task_metric_departure_from_normality": task_alignment,
        "task_metric_transient_amplification": 1.0 + task_alignment,
        "task_metric_spectral_radius": 1.0 + task_alignment,
    })
    return record


def test_selection_regret_is_computed_only_against_heldout_oracle():
    report = evaluate_records([
        _record("env-a", "aligned", alignment=0.1, mse=2.0, loss=0.1),
        _record("env-a", "mse-only", alignment=1.0, mse=0.1, loss=0.8),
        _record("env-a", "bad", alignment=2.0, mse=3.0, loss=1.2),
        _record("env-b", "aligned", alignment=0.1, mse=2.0, loss=0.2),
        _record("env-b", "mse-only", alignment=1.0, mse=0.1, loss=0.9),
        _record("env-b", "bad", alignment=2.0, mse=3.0, loss=1.3),
    ])
    regrets = report["summary"]["selection_regret"]
    assert regrets["alignment_only"]["mean"] == 0.0
    assert regrets["prediction_mse_only"]["mean"] > 0.0
    assert regrets["random_expected"]["mean"] > 0.0
    assert report["groups"][0]["oracle_checkpoint"] == "aligned"


def test_one_candidate_group_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        evaluate_records([_record("env-a", "one", alignment=0.1, mse=0.1, loss=0.1)])


def test_missing_diagnostic_is_rejected():
    record = _record("env-a", "bad", alignment=1.0, mse=1.0, loss=1.0)
    del record["diagnostics"]["alignment"]
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_records([record, _record("env-a", "other", alignment=2.0, mse=2.0, loss=2.0)])


def test_singleton_mse_filter_is_not_mislabeled_as_geometry_selection():
    report = evaluate_records([
        _record("env-a", "epoch_10", alignment=2.0, mse=1.0, loss=0.4),
        _record("env-a", "epoch_20", alignment=0.1, mse=2.0, loss=0.1),
    ])
    group = report["groups"][0]
    assert group["mse_quality_filter"]["geometry_tie_break_identifiable"] is False
    assert group["selector_regret"]["mse_filtered_alignment"] is None
    assert report["summary"]["selector_unavailable_groups"]["mse_filtered_alignment"] == 1


def test_external_task_metric_selectors_are_reported_only_when_complete():
    report = evaluate_records([
        _task_record("env-a", "epoch_10", alignment=1.0, mse=1.0, task_alignment=0.8, loss=0.8),
        _task_record("env-a", "epoch_20", alignment=0.5, mse=1.02, task_alignment=0.1, loss=0.1),
        _task_record("env-a", "epoch_30", alignment=0.2, mse=2.0, task_alignment=0.3, loss=0.4),
    ])
    selectors = report["protocol"]["selectors"]
    assert "task_metric_alignment_only" in selectors
    assert "task_metric_geometry_composite" in selectors
    assert "mse_filtered_task_metric_alignment" in selectors
    assert "mse_filtered_task_metric_geometry_composite" in selectors
    assert "task_metric_alignment_only" in report["summary"]["selection_regret"]
    assert "task_metric_geometry_composite" in report["summary"]["selection_regret"]
