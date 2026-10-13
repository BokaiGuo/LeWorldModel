import json

from experiments.build_six_goal_evidence_matrix import markdown, sweep_summary


def test_counterfactual_sweep_summary_keeps_seed_rows_separate(tmp_path):
    payload = {
        "results": [
            {
                "horizon": 1,
                "metrics": {
                    "latent_endpoint_prediction_error": {"estimate": 1.0},
                    "latent_counterfactual_separation": {"estimate": 2.0},
                    "physical_counterfactual_xy_separation": {"estimate": 3.0},
                    "decoded_position_error_plus": {"estimate": 4.0},
                    "decoded_position_error_minus": {"estimate": 5.0},
                },
            }
        ]
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(payload))
    second.write_text(json.dumps(payload | {"results": [{**payload["results"][0], "metrics": {**payload["results"][0]["metrics"], "latent_endpoint_prediction_error": {"estimate": 3.0}}}]}))
    summary = sweep_summary([first, second])
    assert summary["h1.latent_endpoint_prediction_error"]["seed_count"] == 2
    assert summary["h1.latent_endpoint_prediction_error"]["mean"] == 2.0


def test_matrix_markdown_is_reader_facing_and_bounded():
    report = {
        "goals": [
            {
                "id": 1,
                "name": "alignment",
                "status": "partial",
                "evidence_bucket": "partial",
                "establishes": "bounded result",
                "missing_acceptance": "independent holdout",
            }
        ]
    }
    text = markdown(report)
    assert "not a pooled meta-analysis" in text
    assert "partial" in text
    assert "independent holdout" in text
