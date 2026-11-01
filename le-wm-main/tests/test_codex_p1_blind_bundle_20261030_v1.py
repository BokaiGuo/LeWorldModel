from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_blind_bundle_export_20261030_v1 as export
from experiments.spectral_reachability import codex_p1_blind_bundle_consumer_20261030_v1 as consume


@pytest.fixture(scope="module")
def bundle():
    prepared = export.adapter.saved.native.read(export.SOURCE, cap=32*1024**2)
    replay = export.adapter.saved.replay
    report = replay.one.saved.read_verified(replay.one.source.REPORT, replay.one.source.REPORT_SHA)
    original = replay.one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    return export.build(prepared, original)


def test_real_rows_self_contained_and_no_old_paths(bundle):
    assert consume.validate(bundle)["greedy_unknown"] == 8
    assert len(bundle["rows"]) == 16 and len(bundle["formal_roster"]) == 24
    encoded = json.dumps(bundle)
    assert "/home/" not in encoded and "snapshot" not in encoded and "native_status" not in encoded
    assert "original_suffix10" not in encoded and "reward" not in encoded
    assert all(len(row["history_rgb"]) == 3 for row in bundle["rows"])


@pytest.mark.parametrize("mutation", ["state", "outcome", "future", "path", "extra_task",
                                       "png", "pixels", "history", "action", "goal",
                                       "model", "roster", "duplicates"])
def test_unexpected_fields_and_tampering_rejected(bundle, mutation):
    b = deepcopy(bundle)
    row = b["rows"][0]
    if mutation == "state":
        row["physical_state"] = [1]
    elif mutation == "outcome":
        row["outcome"] = 1
    elif mutation == "future":
        row["cp10"] = "unavailable"
    elif mutation == "path":
        row["cp5_rgb"]["path"] = "/old/physical/result"
    elif mutation == "extra_task":
        row["task"]["root_context"]["snapshot"] = {}
    elif mutation == "png":
        row["cp5_rgb"]["rgb_base64"] = "AAAA"
    elif mutation == "pixels":
        row["cp5_rgb"]["pixel_sha256"] = "0"*64
    elif mutation == "history":
        row["history_rgb"][0]["rgb_base64"] = "AAAA"
    elif mutation == "action":
        row["task"]["root_raw_action_history"][0][0] += 1
    elif mutation == "goal":
        row["task"]["goal"][0] += 1
    elif mutation == "model":
        b["model_files"]["weights.pt"] = "0"*64
    elif mutation == "roster":
        b["formal_roster"].pop()
    else:
        b["rows"][1] = deepcopy(b["rows"][0])
    with pytest.raises((ValueError, AssertionError)):
        consume.validate(b)


def test_external_digest_and_json_duplicate_keys(tmp_path, bundle):
    path = tmp_path/"bundle.json"
    data = json.dumps(bundle, sort_keys=True, allow_nan=False).encode()
    path.write_bytes(data)
    _, result = consume.read_verified(path, consume.sha(data))
    assert result["rows"] == 16
    with pytest.raises(ValueError, match="SHA"):
        consume.read_verified(path, "0"*64)
    data = b'{"schema":"x","schema":"y"}'
    path.write_bytes(data)
    with pytest.raises(ValueError, match="duplicate"):
        consume.read_verified(path, consume.sha(data))


def test_blind_source_wrong_hash_rejected(bundle):
    prepared = export.adapter.saved.native.read(export.SOURCE, cap=32*1024**2)
    replay = export.adapter.saved.replay
    report = json.loads(replay.one.source.REPORT.read_text())
    original = json.loads(Path(report["prepared"]).read_text())
    compact = original["requests"][0]["task"]
    path = next(p for p, h in original["bindings"].items() if h == compact["binding"]["task_sha256"])
    entry = export.adapter.export_load(path, compact["binding"]["task_sha256"])
    entry["task_contract"]["blind_sha256"] = "0"*64
    with pytest.raises(ValueError, match="blind source SHA"):
        export.blind_inputs(entry, prepared["rows"][0]["task"])
