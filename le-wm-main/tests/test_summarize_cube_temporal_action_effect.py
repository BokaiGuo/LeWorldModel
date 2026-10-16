import json

from experiments.summarize_cube_temporal_action_effect import summarize


def test_summary_aggregates_root_rows(tmp_path):
    payload = {
        "schema": "cube_temporal_action_effect.v1",
        "metrics": {
            "rgb_static": {"test_root_rmse": [{"root_id": 1, "rmse": 2.0}, {"root_id": 2, "rmse": 4.0}]},
            "rgb_temporal": {"test_root_rmse": [{"root_id": 1, "rmse": 1.0}, {"root_id": 2, "rmse": 3.0}]},
        },
    }
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(payload))
    result = summarize([path], seed=3)
    assert result["aggregate"]["rgb_temporal"]["n_test_roots"] == 2
    assert result["aggregate"]["rgb_temporal"]["wins_vs_baseline"] == 2
    assert result["aggregate"]["latent_temporal"]["baseline_method"] == "latent_static"
