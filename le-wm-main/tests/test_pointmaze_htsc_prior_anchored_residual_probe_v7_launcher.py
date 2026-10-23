from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_pointmaze_htsc_prior_anchored_residual_probe_v7_runtime_safety_cuda.sh"
)


def test_v7_launcher_uses_fresh_roots_and_two_pulse_residual_design() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "point_maze_v32_prior_anchored_residual_extension_20261022" in text
    assert "ROOT_MIN=13128" in text and "ROOT_MAX_EXCLUSIVE=13336" in text
    assert "--probe-direction-grid 2" in text
    assert "effect_adapter_rgb_online_prior_anchored_candidate_residual_verified_local" in text
    assert "--device cuda --recurrent-smoke" in text
    assert "refusing to reuse immutable active-probe output path" in text
    assert "audit_pointmaze_active_probe_trace_contract" in text
    assert "audit_pointmaze_active_probe_identifiability" in text
