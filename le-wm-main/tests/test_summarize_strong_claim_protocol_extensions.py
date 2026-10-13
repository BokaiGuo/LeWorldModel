import json
from pathlib import Path

from experiments.summarize_strong_claim_protocol import build


def test_extended_bundle_keeps_visual_and_trajectory_artifacts(tmp_path: Path):
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    manifest = {
        "commands": {"training": [], "selection": [], "causal_ood": []},
        "sha256_before_execution": "test",
    }
    (protocol / "manifest.json").write_text(json.dumps(manifest))
    visual = tmp_path / "visual.json"
    visual.write_text(json.dumps({"protocol": {"n_seed_clusters": 5}}))
    cross = tmp_path / "cross.json"
    cross.write_text(json.dumps({"reports": []}))
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(json.dumps({"artifact": "external_frozen_lewm_trajectory_operator_audit"}))
    report = build(protocol, None, visual, cross, trajectory)
    assert report["completion"]["visual_multiseed_planning_artifact"]
    assert report["completion"]["trajectory_local_operator_artifact"]
    assert report["visual_multiseed_planning"]["protocol"]["n_seed_clusters"] == 5
