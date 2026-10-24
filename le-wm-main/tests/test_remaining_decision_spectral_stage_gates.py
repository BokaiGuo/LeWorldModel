from pathlib import Path


def test_remaining_stage_gate_script_requires_cuda_and_immutable_output() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "experiments/decision_benchmark/run_remaining_decision_spectral_stage_gates_cuda.sh").read_text(encoding="utf-8")
    audit = (root / "experiments/decision_benchmark/audit_remaining_decision_spectral_stages.py").read_text(encoding="utf-8")
    assert "refusing CPU fallback" in script
    assert "refusing to overwrite" in script
    assert "blocked_upstream" in audit
    assert "p3_replication_reports" in audit
    # The audit now separates the legacy replication readout from the
    # task-weighted, common-random-numbers mechanism gate.  Keep this
    # contract test aligned with the current explicit no-go outcome instead
    # of requiring the retired aggregate replication label.
    assert "no_go_synthetic_mechanism" in audit
    assert "conditional_task_critical_mode_calibration_pass" in audit
