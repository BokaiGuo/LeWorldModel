from __future__ import annotations

from pathlib import Path


def test_strict_active_probe_launchers_check_cudnn_before_creating_scientific_artifacts() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments"
    launchers = (
        root / "run_pointmaze_htsc_active_probe_fresh_v2_cuda.sh",
        root / "run_pointmaze_htsc_active_probe_decision_ambiguity_v3_cuda.sh",
        root / "run_pointmaze_htsc_affine_probe_v4_cuda.sh",
        root / "run_pointmaze_htsc_candidate_anchored_affine_probe_v5_cuda.sh",
    )
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        assert "experiments.gpu_preflight --device cuda --recurrent-smoke" in text
        # `nvidia-smi` is host telemetry, not proof that the PyTorch/cuDNN
        # process can run.  A launcher must depend on the stronger probe.
        assert "nvidia-smi --query-gpu" not in text
        assert "experiments.run_pointmaze_htsc_active_probe_strict" in text
