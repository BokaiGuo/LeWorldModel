from __future__ import annotations

from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_certified_repairability_v5_cuda.sh"


def test_v5_launcher_is_gpu_only_and_binds_the_fresh_protocol() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "nvidia-smi is unavailable; refusing CPU fallback" in text
    assert "experiments.gpu_preflight --device cuda" in text
    assert 'BUILDER="experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel"' in text
    assert 'ROUTER="experiments.gdd_wm.run_pointmaze_planning_shaped_repairability"' in text
    assert 'ALIAS_AUDIT="experiments.decision_benchmark.run_pointmaze_decision_aliasing_audit"' in text
    assert "--root-id-min 9336 --root-id-max-exclusive 9656 --device cuda" in text
    assert "--ensemble-size 4" in text
    assert "refusing to reuse a V5 panel or execution path" in text
    assert "decision_aliasing_calibration_outcome.json" in text
