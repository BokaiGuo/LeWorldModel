from __future__ import annotations

import json

import pytest

from experiments.aggregate_reacher_capacity_batches import aggregate
from experiments.compare_reacher_capacity_planning import compare


def _planning(path, *, seeds, metric, successes, final, minimum, diagnostics=None):
    path.write_text(
        json.dumps(
            {
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
                    "capacity_readout_artifact": (
                        "readout.npz" if metric == "capacity_safe_readout" else None
                    ),
                },
                "frozen_lewm": {
                    "seeds": seeds,
                    "episode_successes": successes,
                    "final_qpos_distance": final,
                    "min_qpos_distance": minimum,
                    "runtime_diagnostics": diagnostics or {},
                },
            }
        ),
        encoding="utf-8",
    )


def _comparison(tmp_path, name, *, seeds, base_success, readout_success):
    baseline = tmp_path / f"{name}_baseline.json"
    readout = tmp_path / f"{name}_readout.json"
    report_path = tmp_path / f"{name}_paired.json"
    _planning(
        baseline,
        seeds=seeds,
        metric="euclidean",
        successes=base_success,
        final=[0.4, 0.3],
        minimum=[0.2, 0.1],
    )
    _planning(
        readout,
        seeds=seeds,
        metric="capacity_safe_readout",
        successes=readout_success,
        final=[0.3, 0.4],
        minimum=[0.1, 0.2],
        diagnostics={
            "mlp_no_dynamics_gate_enabled": True,
            "mlp_active_rows": 2,
            "mlp_evaluated_rows": 2,
            "mlp_abstained_low_support_rows": 0,
            "mlp_proposal_support_rate": 0.9,
            "mlp_row_support_fraction_mean": 0.9,
            "mlp_row_support_fraction_min": 0.8,
            "mlp_row_support_fraction_max": 1.0,
            "mlp_argmin_changes": 1,
        },
    )
    report = compare(baseline, readout, draws=500, seed=3)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_aggregate_reacher_batches_preserves_strata(tmp_path) -> None:
    first = _comparison(
        tmp_path,
        "first",
        seeds=[1, 2],
        base_success=[0, 0],
        readout_success=[1, 0],
    )
    second = _comparison(
        tmp_path,
        "second",
        seeds=[3, 4],
        base_success=[1, 0],
        readout_success=[0, 0],
    )
    result = aggregate([first, second], draws=1000, seed=9)
    pooled = result["stratified_paired_readout_minus_baseline"]
    assert result["pooled_point_estimates"] == {
        "n": 4,
        "baseline_successes": 1,
        "readout_successes": 1,
    }
    assert pooled["success"]["mean"] == 0.0
    assert pooled["success"]["batch_ns"] == [2, 2]
    assert pooled["success_discordance"] == {
        "readout_only_wins": 1,
        "baseline_only_wins": 1,
        "ties": 2,
    }
    heterogeneity = result["batch_heterogeneity"]["success"]
    assert heterogeneity["batch_means"] == [0.5, -0.5]
    assert heterogeneity["range"] == 1.0
    assert heterogeneity["sign_consistent"] is False
    assert result["runtime_diagnostics"]["active_rows"] == 4
    assert result["runtime_diagnostics"]["argmin_change_rate"] == 0.5


def test_aggregate_reacher_batches_rejects_overlapping_roots(tmp_path) -> None:
    first = _comparison(
        tmp_path,
        "first",
        seeds=[1, 2],
        base_success=[0, 0],
        readout_success=[1, 0],
    )
    second = _comparison(
        tmp_path,
        "second",
        seeds=[2, 3],
        base_success=[1, 0],
        readout_success=[0, 0],
    )
    with pytest.raises(ValueError, match="disjoint"):
        aggregate([first, second], draws=100, seed=9)
