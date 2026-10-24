from pathlib import Path


def test_v6r2_resume_is_idempotent_and_preserves_cuda_only_contract() -> None:
    script = (Path(__file__).resolve().parents[1] / "experiments/gdd_wm/resume_pointmaze_counterfactual_endpoint_v6r2_decision_spectral_cuda.sh").read_text(encoding="utf-8")
    assert "V6R2_RESUME_COMPLETE" in script
    assert "refusing CPU fallback" in script
    assert "pre_outcome_predictions.json" in script
    assert "decision_spectral_mismatch_audit.json" in script
    assert "[[ ! -e" in script
