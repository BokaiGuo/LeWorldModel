import copy
import json

import pytest

from experiments.spectral_reachability import p1_diverse_h10_paired_20261030_main as paired


def setup_inputs(monkeypatch):
    p = json.loads((paired.h10.ROOT / "results/p1_diverse_h10_20261030_main_prepare_RUN01.json").read_bytes())
    old = json.loads(paired.OLD.read_bytes())
    report = {"shards": [{"anchor": a["index"], "sha256": "fixture"} for a in p["anchors"]]}
    monkeypatch.setattr(paired.h10.consumer, "check_shard",
                        lambda directory, anchor, bank, sha: {
                            "endpoints": [anchor["root_state"]] * 42,
                            "statuses": ["probe_completed"] * 42})
    monkeypatch.setattr(paired.h10.native, "read", lambda *args, **kwargs: {
        "rows": [{"responses": [{"state": [0] * 7}] * 10} for _ in range(42)]})
    return p, report, old


def test_complete_roster_and_last_probe_not_paired(monkeypatch, tmp_path):
    p, report, old = setup_inputs(monkeypatch)
    rows = paired.build(p, tmp_path, report, old)["rows"]
    assert len(rows) == 126
    assert sum(len(row["controls"]) for row in rows) == 3 * 41 * 2
    assert all(rows[i]["controls"] == {} for i in (41, 83, 125))
    assert all(row["selected"]["independent_suffix_result"] is None for row in rows)


def test_different_suffix_is_rejected(monkeypatch, tmp_path):
    p, report, old = setup_inputs(monkeypatch)
    old = copy.deepcopy(old)
    target = next(x for x in old["rows"] if x["logical_pair"] == 2 and x["role"] == "zero"
                  and x["probe"] == "zero")
    target["probe_sha256"] = "tampered"
    with pytest.raises(Exception, match="paired control identity"):
        paired.build(p, tmp_path, report, old)
