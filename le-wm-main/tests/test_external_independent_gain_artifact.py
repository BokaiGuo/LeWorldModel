import json

from experiments.empirical_spectral_audit.build_external_independent_gain_artifact import build


def test_external_gain_builder_preserves_non_applicable_metric_as_blocked(tmp_path):
    source = tmp_path / "reacher.json"
    source.write_text(
        json.dumps(
            {
                "schema": "reacher_shared_candidate_ranking.v1",
                "protocol": {
                    "root_disjoint": True,
                    "runtime_state_access": False,
                    "test_planning_used_for_selection": False,
                    "checkpoint_frozen": True,
                },
                "metrics": {
                    "action_ranking_inversion": {"status": "available", "estimate": 0.5, "ci95": [0.4, 0.6]},
                    "multistep_task_error": {"status": "available", "estimate": 1.0, "ci95": [0.8, 1.2]},
                    "angle_closure": {"status": "available", "estimate": 1.0, "ci95": [0.8, 1.2]},
                    "planning_regret": {"status": "available", "estimate": 0.2, "ci95": [0.1, 0.3]},
                    "contact_retention": {"status": "not_applicable"},
                },
            }
        )
    )
    output = tmp_path / "external.json"
    report = build(source, output)
    assert report["schema"] == "external_independent_gain_bundle.v1"
    gate = report["external_independent_gain"]
    assert gate["metrics"]["contact_retention"]["status"] == "blocked"
    assert gate["coverage"]["status"] == "blocked_incomplete_cross_task_metric_panel"
