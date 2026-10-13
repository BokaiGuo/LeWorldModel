import json

import pytest

from experiments.summarize_directional_results import summarize_run, to_markdown


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_summary_reports_best_spectral_and_overall_planner_cost(tmp_path):
    run = tmp_path / "run"
    write_json(
        run / "lift_metrics.json",
        {
            "weighted_mean": {
                "dynamics_lift": {"32": 1.2},
                "predicted_dynamics_lift": {"32": 1.1},
                "planning_lift": {"32": 1.4},
                "prediction_residual_lift": {"32": 0.9},
                "nuisance_lift": {"32": 0.8},
            },
            "prediction_tangent_alignment": {"mean_cosine": 0.5},
        },
    )
    write_json(
        run / "planning_alignment.json",
        {
            "spearman_cost_success": -0.18,
            "spearman_cost_state_distance": 0.31,
            "num_points": 128,
            "spectral_planner_costs": {
                "spectral_top_8": {
                    "spearman_cost_success": 0.4,
                    "spearman_cost_state_distance": -0.9,
                },
                "spectral_top_16": {
                    "spearman_cost_success": -0.21,
                    "spearman_cost_state_distance": 0.42,
                },
            },
        },
    )
    write_json(run / "eval_metrics.json", {"success_rate": 0.75, "return": 123.0})

    row = summarize_run(run)

    assert row["residual_lift@32"] == 0.9
    assert row["best_spectral_cost"] == "spectral_top_16"
    assert row["best_spectral_success_spearman"] == -0.21
    assert row["best_spectral_state_spearman"] == 0.42
    assert row["best_spectral_state_delta_vs_latent"] == pytest.approx(0.11)
    assert row["best_planner_cost"] == "spectral_top_16"
    assert row["best_planner_state_spearman"] == 0.42

    markdown = to_markdown([row])
    assert "residual_lift@32" in markdown
    assert "best_planner_cost" in markdown
    assert "best_spectral_state_delta_vs_latent" in markdown
    assert "spectral_top_16" in markdown


def test_summary_keeps_latent_cost_as_best_when_spectral_is_worse(tmp_path):
    run = tmp_path / "run"
    write_json(run / "lift_metrics.json", {"weighted_mean": {}})
    write_json(
        run / "planning_alignment.json",
        {
            "spearman_cost_success": -0.3,
            "spearman_cost_state_distance": 0.5,
            "spectral_planner_costs": {
                "spectral_top_8": {
                    "spearman_cost_success": -0.1,
                    "spearman_cost_state_distance": 0.2,
                }
            },
        },
    )

    row = summarize_run(run)

    assert row["best_spectral_cost"] == "spectral_top_8"
    assert row["best_planner_cost"] == "latent_cost"
    assert row["best_planner_success_spearman"] == -0.3
    assert row["best_planner_state_spearman"] == 0.5
