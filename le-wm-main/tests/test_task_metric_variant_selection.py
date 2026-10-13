import copy

from experiments.audit_task_metric_variant_selection import DIAGNOSTICS, VARIANTS, audit


def _record(stage: int, loss: float, offset: float) -> dict:
    variants = {
        name: {
            field: float(offset + index + field_index * 0.01)
            for field_index, field in enumerate(DIAGNOSTICS)
        }
        for index, name in enumerate(VARIANTS)
    }
    # Make the pre-registered default selector identify the oracle without
    # allowing it to see planning loss.
    variants["default_task_space"]["task_metric_alignment"] = float(loss)
    return {
        "group": "PushT/in_distribution/arch/seed_1",
        "checkpoint": f"arch/seed_1/epoch_{stage}",
        "diagnostics": {
            "task_metric_variants": variants,
            "task_metric_variant_metadata": {
                name: {"matrix": [[1.0, 0.0], [0.0, 1.0]]} for name in VARIANTS
            },
            "alignment": 0.3,
            "transient_amplification": 1.0,
            "prediction_mse": 1.0,
            "latent_variance": 1.0,
            "spectral_radius": 0.9,
        },
        "heldout": {"planning_loss": loss},
    }


def test_variant_audit_is_selector_only_and_reports_all_variants():
    report = audit([_record(10, 0.1, 0.5), _record(20, 0.8, 0.0)])
    assert set(report["summary"]) == {
        f"{name}/{field}" for name in VARIANTS for field in DIAGNOSTICS
    } | {f"{name}/geometry_composite" for name in VARIANTS} | {
        "random_expected", "legacy/alignment", "legacy/transient_amplification",
        "prediction_mse", "variance", "spectrum_radius",
    }
    assert report["groups"][0]["selectors"]["default_task_space/task_metric_alignment"]["is_oracle"]
    assert "held-out planning" in report["claim_boundary"]


def test_variant_audit_does_not_mutate_records():
    records = [_record(10, 0.2, 0.0), _record(20, 0.3, 0.1)]
    before = copy.deepcopy(records)
    audit(records)
    assert records == before
