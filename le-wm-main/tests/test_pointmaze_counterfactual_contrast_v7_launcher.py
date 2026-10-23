from __future__ import annotations

from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "decision_benchmark"
    / "run_pointmaze_counterfactual_contrast_v7_cuda.sh"
)


def test_counterfactual_contrast_launcher_is_cuda_only_and_seals_outcomes() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "run_pointmaze_counterfactual_contrast_v7_cuda" in text
    assert "exec" in text
    assert "--device cpu" not in text
