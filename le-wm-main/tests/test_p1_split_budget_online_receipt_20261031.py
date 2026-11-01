import importlib
import json

import pytest


def test_runtime_receipt_is_persisted_and_digest_matches(tmp_path):
    online = importlib.import_module(
        "experiments.spectral_reachability.p1_split_budget_online_20261031"
    )
    path = tmp_path / "run_report.json"
    original = {"status": "recorded_not_consumed", "shards": [{"index": 0}]}
    path.write_text(json.dumps(original), encoding="utf-8")
    identity = {"original_state_dict": "model", "all_buffers": "buffers"}
    result = {"status": original["status"], "host": {"device": "cuda:0"},
              "model_before": identity, "model_after": identity, "online_events": 24}

    digest = online.persist_runtime_receipt(tmp_path, result)

    saved = json.loads(path.read_bytes())
    assert saved == {**original, **result}
    assert digest == online.sha(path)


def test_runtime_receipt_rejects_mismatched_status(tmp_path):
    online = importlib.import_module(
        "experiments.spectral_reachability.p1_split_budget_online_20261031"
    )
    path = tmp_path / "run_report.json"
    path.write_text('{"status":"partial_or_failed","shards":[]}', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="report/result identity"):
        online.persist_runtime_receipt(tmp_path, {"status": "recorded_not_consumed"})
    assert path.read_bytes() == before
