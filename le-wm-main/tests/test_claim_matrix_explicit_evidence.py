import json

from experiments.build_six_goal_evidence_matrix import build_matrix


def test_explicit_evidence_manifest_does_not_fallback_to_history(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    historical = repo / "results" / "pusht" / "lewm" / "visual_only_online_planning_20261008_seed1"
    historical.mkdir(parents=True)
    (historical / "online_planning.json").write_text(json.dumps({"reports": []}))
    sweep = repo / "results" / "pusht" / "lewm" / "counterfactual_horizon_sweep_old_20261008"
    sweep.mkdir(parents=True)
    (sweep / "counterfactual_horizon_sweep.json").write_text(json.dumps({"results": []}))
    manifest = tmp_path / "evidence_manifest.json"
    manifest.write_text(json.dumps({"records": {}}))
    report = build_matrix(repo, evidence_manifest_path=manifest)
    goal_one = report["goals"][0]
    assert goal_one["key_metrics"]["controlled_selection_win_rate"] is None
    assert report["goals"][2]["key_metrics"]["new_frozen_counterfactual_sweep"] == {}
    assert report["goals"][3]["key_metrics"]["visual_only_online"] == {}


def test_explicit_evidence_manifest_rejects_changed_record(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    artifact = repo / "controlled.json"
    artifact.write_text(json.dumps({"summary": {"alignment_selection_win_rate": 1.0}}))
    manifest = tmp_path / "evidence_manifest.json"
    manifest.write_text(json.dumps({
        "records": {
            "controlled_alignment": {
                "path": str(artifact),
                "sha256": "wrong",
            },
        },
    }))
    report = build_matrix(repo, evidence_manifest_path=manifest)
    assert report["goals"][0]["key_metrics"]["controlled_selection_win_rate"] is None
