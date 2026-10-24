from pathlib import Path


def test_p2_synthetic_launcher_is_cuda_only_and_immutable() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "experiments/decision_benchmark/run_decision_spectral_synthetic_p2_cuda.sh").read_text(encoding="utf-8")
    runner = (root / "experiments/decision_benchmark/run_decision_spectral_synthetic_p2_cuda.py").read_text(encoding="utf-8")
    assert "refusing CPU fallback" in script
    assert "--device cuda" in script
    assert "refusing to overwrite" in script
    assert 'choices=("cuda",)' in runner
    assert "gauge_invariant_to_tolerance" in runner
