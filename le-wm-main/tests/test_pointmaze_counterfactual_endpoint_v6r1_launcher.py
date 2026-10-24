from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_counterfactual_endpoint_v6r1_cuda.sh"
RESUME = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "resume_pointmaze_counterfactual_endpoint_v6r1_cuda.sh"


def test_v6r1_launcher_is_fresh_and_cuda_only() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "point_maze_v81_endpoint_v6r1_extension_20261023" in text
    assert "ROOT_MIN=35912" in text
    assert "ROOT_MAX_EXCLUSIVE=36552" in text
    assert "--device cuda" in text
    assert "refusing CPU fallback" in text
    assert "pointmaze_counterfactual_endpoint_v6r1_fresh_endpoint_20261023_cuda" in text


def test_v6r1_resume_uses_immutable_panel_and_cuda_only_runner() -> None:
    text = RESUME.read_text(encoding="utf-8")
    assert 'VALIDATOR="experiments.gdd_wm.validate_pointmaze_counterfactual_endpoint_v6_panel"' in text
    assert "--panel \"${PANEL}\" --frozen-preflight \"${PREFLIGHT}\"" in text
    assert "--device cuda" in text
    assert "refusing CPU fallback" in text
    assert "pointmaze_counterfactual_endpoint_v6r1_fresh_endpoint_20261023_cuda" in text
