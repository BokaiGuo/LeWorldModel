import pytest

from experiments.gdd_wm.audit_projection_execution_consequence import audit


def _report(delta: float) -> dict:
    fitted, baseline = [], []
    for index in range(24):
        root = f"root-{index}"
        baseline.append({
            "instance_id": root, "teacher_set_id": "set", "predicted_candidate_id": "baseline",
            "real_task_regret": 1.0,
        })
        fitted.append({
            "instance_id": root, "teacher_set_id": "set", "predicted_candidate_id": "fitted",
            "real_task_regret": 1.0 + delta,
        })
    return {
        "schema": "gdd_wm_d2_projection_distillation.v0.1",
        "metrics": {"held_out": {"outcome": {
            "fitted_projection": {"per_group": fitted}, "frozen_baseline": {"per_group": baseline},
        }}},
    }


def test_root_paired_audit_confirms_a_uniform_heldout_regret_improvement() -> None:
    result = audit(_report(-0.2), draws=400, seed=7)
    assert result["passed"] is True
    assert result["outcome"]["upper_95_fitted_minus_baseline_regret"] < 0.0


def test_root_paired_audit_rejects_a_uniform_heldout_regret_regression() -> None:
    result = audit(_report(0.2), draws=400, seed=7)
    assert result["passed"] is False
    assert result["outcome"]["upper_95_fitted_minus_baseline_regret"] > 0.0


def test_audit_rejects_an_improvement_confined_to_one_root() -> None:
    report = _report(0.0)
    fitted = report["metrics"]["held_out"]["outcome"]["fitted_projection"]["per_group"]
    for row in fitted:
        row["predicted_candidate_id"] = "baseline"
    fitted[0].update({
        "predicted_candidate_id": "changed", "real_task_regret": 0.0,
    })
    result = audit(report, draws=400, seed=7)
    assert result["passed"] is False
    assert result["outcome"]["changed_root_fraction"] < 0.10


def test_audit_requires_identity_bound_d2_group_traces() -> None:
    report = _report(-0.2)
    del report["metrics"]["held_out"]["outcome"]["fitted_projection"]["per_group"]
    with pytest.raises(ValueError, match="identity-bound"):
        audit(report)
