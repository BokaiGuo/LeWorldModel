from __future__ import annotations

from pathlib import Path


def test_launcher_keeps_calibration_and_outcome_panel_sizes_separate() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "experiments/gdd_wm/run_tworoom_verified_local_calibrated_v1_cuda.sh"
    ).read_text(encoding="utf-8")

    assert "CALIBRATION_EPISODES=64" in launcher
    assert "OUTCOME_EPISODES=128" in launcher
    assert 'CAL_COMMON=("${BASE_COMMON[@]}" --episodes "${CALIBRATION_EPISODES}")' in launcher
    assert 'OUTCOME_COMMON=("${BASE_COMMON[@]}" --episodes "${OUTCOME_EPISODES}")' in launcher
    assert launcher.count('"${CAL_COMMON[@]}"') == 2
    assert launcher.count('"${OUTCOME_COMMON[@]}"') == 2
