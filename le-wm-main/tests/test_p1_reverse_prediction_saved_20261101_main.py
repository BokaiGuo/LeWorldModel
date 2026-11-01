"""Saved-array checks only; no model inference or physical execution."""
import json

import pytest

from experiments.spectral_reachability import p1_reverse_prediction_saved_20261101_main as saved

SOURCE = saved.control.ROOT / "results/p1_reverse_witness_prediction_20261101_main_MAIN01.json"


def test_actual_saved_vectors_recompute():
    assert saved.check(SOURCE)["status"] == "saved_numerics_verified"


@pytest.mark.parametrize("field", ["predicted_goal_q", "prediction_mse"])
def test_tampered_score_is_rejected(tmp_path, field):
    doc = json.loads(SOURCE.read_bytes())
    doc["rows"][0][field] += .1
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="mismatch"):
        saved.check(path)


def test_zero_physical_counterfactual_cannot_be_invented(tmp_path):
    doc = json.loads(SOURCE.read_bytes())
    doc["rows"][-1]["observed"] = doc["rows"][0]["observed"]
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="fabricated"):
        saved.check(path)
