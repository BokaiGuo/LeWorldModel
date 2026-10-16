import json

from experiments.summarize_reacher_shared_candidate_ranking import main


def test_reacher_aggregate_requires_and_preserves_multiple_conditions(tmp_path, monkeypatch):
    def payload(delta):
        return {
            "schema": "reacher_shared_candidate_ranking.v1",
            "protocol": {"root_disjoint": True, "target_delta": delta},
            "roots": [
                {
                    "top1_hit": 1,
                    "top4_hit": 1,
                    "action_ranking_inversion": 0.2,
                    "multistep_task_error": 0.5,
                    "angle_closure": 0.5,
                    "planning_regret": 0.1,
                },
                {
                    "top1_hit": 0,
                    "top4_hit": 1,
                    "action_ranking_inversion": 0.4,
                    "multistep_task_error": 1.0,
                    "angle_closure": 1.0,
                    "planning_regret": 0.3,
                },
            ],
        }

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(payload([0.65, -0.65])))
    second.write_text(json.dumps(payload([0.65, 0.65])))
    output = tmp_path / "aggregate.json"
    # The first fixture predates the protocol field; both target geometries are
    # therefore supplied explicitly rather than inferred from filenames.
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize",
            "--input", str(first),
            "--input", str(second),
            "--condition-target-delta", "0.65", "-0.65",
            "--condition-target-delta", "0.65", "0.65",
            "--output", str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["schema"] == "reacher_shared_candidate_ranking.aggregate.v1"
    assert report["protocol"]["condition_count"] == 2
    assert report["pooled"]["n"] == 4
    assert report["contact_retention"]["status"] == "not_applicable"
