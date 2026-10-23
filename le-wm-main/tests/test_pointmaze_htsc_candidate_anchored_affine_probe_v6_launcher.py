from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_pointmaze_htsc_candidate_anchored_affine_probe_v6_runtime_safety_cuda.sh"
)


def test_v6_launcher_uses_fresh_roots_and_runtime_safety_repair() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "point_maze_v31_candidate_anchored_runtime_safety_extension_20261022" in text
    assert "ROOT_MIN=12920" in text and "ROOT_MAX_EXCLUSIVE=13128" in text
    assert "--device cuda --recurrent-smoke" in text
    assert "refusing to reuse immutable active-probe output path" in text
    assert "--visual-calibration-roots \"${VISUAL_ROOTS}\"" in text
    assert "--test-roots \"${TEST_ROOTS}\"" in text
    assert "audit_pointmaze_active_probe_trace_contract" in text
    assert "audit_pointmaze_active_probe_identifiability" in text
