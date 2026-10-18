from __future__ import annotations

import json
import pytest

from experiments.run_live_cube_rgb_macro_panel import (
    ARMS,
    _arm_command,
    _paired_summary,
    _summarize,
)


def test_panel_declares_four_paired_arms_with_shared_runner() -> None:
    assert set(ARMS) == {
        "fullscale_reclose",
        "fullscale_open_loop",
        "lowgain_reclose",
        "lowgain_persistent",
        "lowgain_persistent_probe",
        "lowgain_effect_adaptive",
        "lowgain_backtrack",
        "lowgain_mode_reclose",
        "lowgain_effect_stop",
        "factorized_lowgain_backtrack",
        "multiphase_lowgain_backtrack",
    }
    command = _arm_command(
        arm="lowgain_effect_adaptive",
        output="/tmp/arm.json",
        device="cuda",
        seed=7,
        calibration_roots=8,
        test_roots=2,
        macro_cycles=5,
        action_block=1,
        approach_steps=8,
        descent_steps=15,
        probe_steps=4,
        tasks=[2, 3],
    )
    assert command[command.index("--device") + 1] == "cuda"
    assert "--effect-adaptation" in command
    assert "--push-scale" in command
    assert command[command.index("--calibration-target") + 1] == "privileged_task"
    assert command[command.index("--tasks") + 1:command.index("--tasks") + 3] == ["2", "3"]


def test_panel_can_request_rgb_realized_calibration_target() -> None:
    command = _arm_command(
        arm="lowgain_persistent",
        output="/tmp/arm.json",
        device="cuda",
        seed=7,
        calibration_roots=8,
        test_roots=2,
        macro_cycles=5,
        action_block=1,
        approach_steps=8,
        descent_steps=15,
        probe_steps=4,
        calibration_target="rgb_goal",
    )
    assert command[command.index("--calibration-target") + 1] == "rgb_goal"


def test_panel_factorized_arm_uses_independent_bank() -> None:
    command = _arm_command(
        arm="factorized_lowgain_backtrack",
        output="/tmp/factorized-arm.json",
        device="cuda",
        seed=7,
        calibration_roots=8,
        test_roots=2,
        macro_cycles=5,
        action_block=1,
        approach_steps=8,
        descent_steps=15,
        probe_steps=4,
        tasks=[2, 3],
    )
    assert "--factorized-bank" in command
    assert "--push-backtrack" in command


def test_panel_summary_preserves_distance_and_gate_fields() -> None:
    payload = {
        "aggregate": {
            "success_rate": 0.5,
            "contact_observed_rate": 1.0,
            "approach_valid_rate": 1.0,
            "zero_macro_rate": 0.25,
        },
        "test": [
            {
                "block_goal_distance_trace": [0.2, 0.1],
                "stopped_for_rgb_regression": True,
            },
            {
                "block_goal_distance_trace": [0.3, 0.25],
                "stopped_for_rgb_regression": False,
            },
        ],
    }
    summary = _summarize(payload)
    assert summary["roots"] == 2
    assert summary["final_distance_mean"] == 0.175
    assert summary["best_distance_mean"] == 0.175
    assert summary["stopped_for_rgb_regression"] == 1


def test_panel_summary_exposes_mode_gate_diagnostics() -> None:
    payload = {
        "aggregate": {
            "success_rate": 0.0,
            "contact_observed_rate": 1.0,
            "approach_valid_rate": 1.0,
            "zero_macro_rate": 0.0,
        },
        "test": [{
            "block_goal_distance_trace": [0.2, 0.1],
            "mode_reclose_trace": [
                {"reclose": True, "reason": "first_cycle", "rgb_progress": None},
                {"reclose": True, "reason": "insufficient_rgb_progress", "rgb_progress": 0.0},
                {"reclose": False, "reason": "progress_ok", "rgb_progress": 0.2},
            ],
        }],
    }
    summary = _summarize(payload)
    assert summary["mode_reclose_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["mode_reclose_reason_counts"] == {
        "first_cycle": 1,
        "insufficient_rgb_progress": 1,
        "progress_ok": 1,
    }
    assert summary["rgb_progress_valid_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["rgb_progress_mean"] == pytest.approx(0.1)
    assert summary["rgb_progress_last_values"] == [0.2]


def test_panel_reports_root_paired_arm_deltas() -> None:
    def payload(finals: list[float], successes: list[bool]) -> dict:
        return {
            "test": [
                {
                    "block_goal_distance_trace": [value + 0.1, value],
                    "success": success,
                    "contact_observed": True,
                }
                for value, success in zip(finals, successes)
            ]
        }

    result = _paired_summary({
        "fullscale_reclose": payload([0.5, 0.4], [False, False]),
        "lowgain_persistent": payload([0.3, 0.45], [True, False]),
    })
    row = result["lowgain_persistent"]
    assert row["paired_roots"] == 2
    assert row["mean_final_distance_reduction_vs_fullscale"] == pytest.approx(0.075)
    assert row["success_delta_sum_vs_fullscale"] == 1.0
    assert row["contact_delta_sum_vs_fullscale"] == 0.0
