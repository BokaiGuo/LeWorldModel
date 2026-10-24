from pathlib import Path


def test_p3_launcher_exposes_immutable_cuda_seeded_replicates() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "experiments/decision_benchmark/run_spectral_repair_synthetic_p3_cuda.sh").read_text(encoding="utf-8")
    runner = (root / "experiments/decision_benchmark/run_spectral_repair_synthetic_p3_cuda.py").read_text(encoding="utf-8")
    assert "refusing CPU fallback" in script
    assert "P3_OUTPUT" in script
    assert "P3_SEED" in script
    assert '"--seed", type=int' in runner
    assert '"seed": int(seed)' in runner
    assert "torch.as_tensor(values, device=d, dtype=torch.float64)" in runner
