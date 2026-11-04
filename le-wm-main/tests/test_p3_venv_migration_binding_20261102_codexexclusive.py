import json

import pytest

from experiments.spectral_reachability import p3_venv_migration_binding_20261102_codexexclusive as m


def test_record_detects_content_and_rejects_symlink(tmp_path):
    file = tmp_path / "file"
    file.write_bytes(b"a")
    first = m.record(file)
    file.write_bytes(b"b")
    assert m.record(file)["sha256"] != first["sha256"]
    link = tmp_path / "link"
    link.symlink_to(file)
    with pytest.raises(ValueError, match="non-symlink"):
        m.record(link)


def test_verify_requires_external_hash_and_detects_drift(tmp_path, monkeypatch):
    file = tmp_path / "binding"
    value = {"execution_authorized": False, "old_new_environment_equivalence": "unknown"}
    file.write_text(json.dumps(value))
    monkeypatch.setattr(m, "build", lambda: value)
    digest = m.sha(file)
    assert not m.verify(file, digest)["execution_authorized"]
    with pytest.raises(ValueError, match="SHA"):
        m.verify(file, "0" * 64)
    monkeypatch.setattr(m, "build", lambda: {**value, "changed": True})
    with pytest.raises(ValueError, match="drift"):
        m.verify(file, digest)


def test_build_rejects_non_venv(monkeypatch):
    monkeypatch.setattr(m.sys, "prefix", m.sys.base_prefix)
    with pytest.raises(ValueError, match=".venv"):
        m.build()
