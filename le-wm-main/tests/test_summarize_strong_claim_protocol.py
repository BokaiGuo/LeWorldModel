import json
from pathlib import Path

from experiments.summarize_strong_claim_protocol import (
    assess_alignment,
    assess_external_prediction,
    summarize_causal,
    summarize_selection,
    assess_visual_online_planning,
)


def test_missing_selection_provenance_is_not_valid(tmp_path):
    path = tmp_path / "checkpoint_record.json"
    path.write_text(json.dumps({"record_schema": "real_checkpoint_selection.v1", "protocol": {"interface": {}}, "record": {}}))
    report = summarize_selection([path])
    assert report["status"] == "partial_or_invalid"
    assert report["valid"] == 0


def test_missing_causal_artifact_is_explicit(tmp_path):
    report = summarize_causal([tmp_path / "missing.json"])
    assert report["status"] == "partial_or_invalid"
    assert report["valid"] == 0


def test_complete_selection_is_not_automatically_scientific_pass():
    selection = {
        "status": "complete",
        "benchmark": {"summary": {"selection_regret": {
            "alignment_only": {"mean": 0.08},
            "spectrum_only": {"mean": 0.06},
        }}},
    }
    passed, reason = assess_alignment(selection)
    assert passed is False
    assert reason == "standalone_alignment_does_not_beat_spectrum_only"


def test_external_prediction_requires_action_sensitive_behavior():
    payload = {"reports": [{
        "environment": "swm/TwoRoom-v1",
        "status": "completed",
        "summary": {"paired_action_effect": {
            "h1": {"zero_minus_observed_mse": {"ci95": [-0.01, -0.001]}},
        }},
    }]}
    passed, reason = assess_external_prediction(payload)
    assert passed is False
    assert reason == "observed_action_not_better_than_zero:swm/TwoRoom-v1"


def test_visual_planning_requires_paired_strict_gain_and_distance_gain():
    payload = {
        "aggregate": {
            "n_seed_clusters": 2,
            "vanilla_success_total": 1,
            "zero_success_total": 0,
            "vanilla_minus_zero_block_xy_distance": {"ci_high": 1.0},
        }
    }
    passed, reason = assess_visual_online_planning(payload)
    assert passed is False
    assert reason == "vanilla_block_xy_not_better_than_zero"
