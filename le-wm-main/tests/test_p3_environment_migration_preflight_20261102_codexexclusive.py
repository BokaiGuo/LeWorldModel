import hashlib
import json

import pytest

from experiments.spectral_reachability import p3_environment_migration_preflight_20261102_codexexclusive as m


def row(path, raw):
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_sources_match_missing_mismatch_and_symlink(tmp_path):
    good, missing, bad, link = [tmp_path / x for x in ("good", "missing", "bad", "link")]
    good.write_bytes(b"abc")
    bad.write_bytes(b"xyz")
    link.symlink_to(good)
    actual = m.audit_sources([row(good, b"abc"), row(missing, b"abc"),
                              row(bad, b"abc"), row(link, b"abc")])
    assert [x["status"] for x in actual] == ["match", "missing", "mismatch", "symlink_unverified"]


def test_duplicate_and_invalid_hash_rejected(tmp_path):
    item = row(tmp_path / "x", b"")
    with pytest.raises(ValueError):
        m.audit_sources([item, item])
    with pytest.raises(ValueError):
        m.audit_sources([{**item, "sha256": "x" * 64}])


def test_prepare_hash_and_non_authorization(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"source")
    prepare = tmp_path / "old.json"
    prepare.write_text(json.dumps({"schema": "codex_p3_online_budget_host_20261030_v1.prepare",
                                   "sources": [row(source, b"source")]}))
    monkeypatch.setattr(m, "current_environment", lambda: {"cuda_probed": False})
    monkeypatch.setattr(m, "legacy_closure_probe", lambda: {"status": "fixture"})
    report = m.build(prepare, m.sha_file(prepare))
    assert report["historical_source_counts"]["match"] == 1
    assert report["old_new_equivalence"] == "unknown"
    assert report["execution_authorized"] is report["cuda_used"] is False
    with pytest.raises(ValueError, match="SHA"):
        m.build(prepare, "0" * 64)


def test_historical_prepare_symlink_rejected(tmp_path):
    source, link = tmp_path / "source", tmp_path / "link"
    source.write_text("{}")
    link.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        m.build(link, m.sha_file(source))
