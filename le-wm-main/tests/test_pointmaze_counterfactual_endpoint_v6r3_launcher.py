from pathlib import Path


def test_v6r3_launcher_uses_new_roots_and_cuda_only_paircontrast_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "experiments/gdd_wm/run_pointmaze_counterfactual_endpoint_v6r3_paircontrast_cuda.sh").read_text(encoding="utf-8")
    assert "ROOT_MIN=37192" in script
    assert "ROOT_MAX_EXCLUSIVE=37832" in script
    assert "v83_endpoint_v6r3_paircontrast" in script
    assert "decision_spectral_paircontrast_audit.json" in script
    assert "refusing CPU fallback" in script
    assert 'if [[ ! -e "${PREFLIGHT_OUTPUT}" ]]' in script
    assert "prior_context_sha256 | keys[]" in script
