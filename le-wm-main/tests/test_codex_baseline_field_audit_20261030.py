import copy
import json
import sys

import pytest

from experiments.spectral_reachability import codex_baseline_field_audit_20261030 as m


def test_real_audit_separates_sources_and_keeps_missing():
    out = m.audit(m.read_inputs())
    assert not out["comparisons_allowed"]
    assert out["methods"]["GRASP"]["author_reported"]["value"]["value"] == 59.0
    assert out["methods"]["GRASP"]["local"]["root_resources"][0]["fields"]["total_FLOPs"]["value"] is None
    assert out["methods"]["Hi-LeWM"]["local"]["fields"]["full_policy"]["value"] is False
    assert out["methods"]["PlanWAM"]["local"]["result"]["value"] is None
    assert out["methods"]["LeWM"]["author_reported"]["uncertainty"]["value"]["kind"] == "reported_pm_unspecified"


def test_no_missing_to_zero_and_false_is_recorded():
    assert m.field(None, "/x", "s")["status"] == "unknown_in_source"
    assert m.field(False, "/x", "s")["status"] == "recorded"
    assert m.field(0, "/x", "s")["value"] == 0
    assert m.missing_leaves({"a": [0, None, {"b": "unknown"}]}) == ["/a/1", "/a/2/b"]


def test_reject_duplicate_paper_and_bad_roster():
    data = m.read_inputs()
    row = next(x for x in data["paper"]["rows"] if x["id"] == m.IDS["GRASP"])
    data["paper"]["rows"].append(copy.deepcopy(row))
    with pytest.raises(ValueError):
        m.audit(data)
    data = m.read_inputs()
    data["grasp"]["contract"]["order"][2][1] = "other"
    with pytest.raises(ValueError):
        m.audit(data)


def test_source_tamper_rejected(tmp_path, monkeypatch):
    p = tmp_path / "bad.json"
    p.write_text("{}")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "FILES", {"bad": ("bad.json", "0" * 64)})
    with pytest.raises(ValueError, match="SHA mismatch"):
        m.read_inputs()


def test_cli_exclusive_and_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(m, "read_inputs", lambda: {})
    monkeypatch.setattr(m, "audit", lambda _: {"test": True})
    path = tmp_path / "results" / (m.NAME + "_test.json")
    monkeypatch.setattr(sys, "argv", ["audit", "build", "--output", str(path)])
    m.main()
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", ["audit", "verify", "--output", str(path), "--sha256", sha])
    m.main()
    monkeypatch.setattr(sys, "argv", ["audit", "build", "--output", str(path)])
    with pytest.raises(FileExistsError):
        m.main()
    path.write_text(json.dumps({"test": False}))
    monkeypatch.setattr(sys, "argv", ["audit", "verify", "--output", str(path), "--sha256", sha])
    with pytest.raises(ValueError):
        m.main()
