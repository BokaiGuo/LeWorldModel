from pathlib import Path


def test_p3v2_is_cuda_only_and_excludes_full_effect_from_rank_matched_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "experiments/decision_benchmark/run_spectral_repair_synthetic_p3_v2_cuda.sh").read_text(encoding="utf-8")
    runner = (root / "experiments/decision_benchmark/run_spectral_repair_synthetic_p3_v2_cuda.py").read_text(encoding="utf-8")
    assert "refusing CPU fallback" in script
    assert "P3V2_OUTPUT" in script
    assert 'choices=("cuda",)' in runner
    assert "RANK_MATCHED_ARMS" in runner
    assert "full_effect_is_capacity_reference_not_primary_gate" in runner
    assert "decision_weighted_rank1_beats_all_rank_matched_controls" in runner
    assert '"--task-weight", type=float' in runner


def test_p3v2_metric_sweep_reports_every_declared_task_weight_on_cuda() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "experiments/decision_benchmark/run_spectral_repair_synthetic_p3_v2_weight_sweep_cuda.sh").read_text(encoding="utf-8")
    runner = (root / "experiments/decision_benchmark/run_spectral_repair_synthetic_p3_v2_weight_sweep_cuda.py").read_text(encoding="utf-8")
    assert "refusing CPU fallback" in script
    assert "WEIGHTS = (64.0, 256.0, 1024.0, 4096.0)" in runner
    assert "for weight in WEIGHTS" in runner
    assert "common_random_numbers" in runner
    assert "seed=int(args.seed), task_weight=weight" in runner
