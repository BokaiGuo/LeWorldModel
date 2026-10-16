from __future__ import annotations

import json

from experiments.compare_reacher_capacity_planning import compare


def _write(path, *, metric, successes, final, minimum, diagnostics=None):
    path.write_text(json.dumps({
        "schema": "live_reacher_sequence_candidate_planning.v1",
        "protocol": {
            "checkpoint": "checkpoint",
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 5,
            "candidate_bank": "capacity17",
            "candidate_count": 17,
            "final_action_selection": "best",
            "target_delta": [0.65, -0.65],
            "cost_metric": metric,
            "capacity_readout_artifact": "readout.npz" if metric != "euclidean" else None,
        },
        "frozen_lewm": {
            "seeds": [11, 12, 13],
            "episode_successes": successes,
            "final_qpos_distance": final,
            "min_qpos_distance": minimum,
            "runtime_diagnostics": diagnostics or {},
        },
    }))


def test_compare_reacher_capacity_planning_is_paired(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    readout = tmp_path / "readout.json"
    _write(
        baseline,
        metric="euclidean",
        successes=[0, 1, 0],
        final=[0.3, 0.2, 0.4],
        minimum=[0.2, 0.1, 0.3],
    )
    _write(
        readout,
        metric="capacity_safe_readout",
        successes=[1, 1, 0],
        final=[0.1, 0.2, 0.5],
        minimum=[0.1, 0.1, 0.2],
        diagnostics={
            "mlp_no_dynamics_gate_enabled": True,
            "mlp_active_rows": 6,
            "mlp_evaluated_rows": 4,
            "mlp_abstained_low_support_rows": 2,
            "mlp_min_support_fraction": 0.75,
            "mlp_proposal_support_rate": 0.25,
            "mlp_row_support_fraction_mean": 0.8,
            "mlp_row_support_fraction_min": 0.76,
            "mlp_row_support_fraction_max": 0.9,
            "mlp_argmin_changes": 3,
        },
    )
    result = compare(baseline, readout, draws=1000, seed=7)
    paired = result["paired_readout_minus_baseline"]
    assert paired["success"]["mean"] == 1 / 3
    assert paired["success_discordance"] == {
        "readout_only_wins": 1,
        "baseline_only_wins": 0,
        "ties": 2,
    }
    assert result["runtime_diagnostics"]["proposal_support_rate"] == 0.25
    assert result["runtime_diagnostics"]["active_rows"] == 6
    assert result["runtime_diagnostics"]["abstained_low_support_rows"] == 2
    assert result["runtime_diagnostics"]["min_support_fraction"] == 0.75
    assert result["runtime_diagnostics"]["row_support_fraction_mean"] == 0.8
    assert result["runtime_diagnostics"]["row_support_fraction_min"] == 0.76
    assert result["runtime_diagnostics"]["row_support_fraction_max"] == 0.9
    assert result["runtime_diagnostics"]["argmin_change_rate"] == 0.75
