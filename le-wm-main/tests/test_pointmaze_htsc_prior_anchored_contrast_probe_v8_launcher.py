from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_pointmaze_htsc_prior_anchored_contrast_probe_v8_runtime_safety_cuda.sh"
)


def test_v8_launcher_uses_v7_disjoint_roots_and_calibrated_contrast_gate() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "point_maze_v33_prior_anchored_contrast_extension_20261022" in text
    assert "ROOT_MIN=13336" in text and "ROOT_MAX_EXCLUSIVE=13544" in text
    assert "point_maze_v32_prior_anchored_residual_extension_20261022" in text
    assert "--local-contrast-min-advantage-grid 0.00,0.02,0.05" in text
    assert "--probe-direction-grid 2" in text
    assert "effect_adapter_rgb_online_prior_anchored_candidate_residual_contrast_verified_local" in text
    assert "--device cuda --recurrent-smoke" in text
