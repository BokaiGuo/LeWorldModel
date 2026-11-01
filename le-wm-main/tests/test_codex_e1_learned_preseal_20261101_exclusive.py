import hashlib
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import codex_e1_learned_preseal_20261101_exclusive as m


def test_cpu_preflight_rejects_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU preflight"):
        m.preflight(None)


def test_rejects_existing_paths_even_dangling_symlink(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    monkeypatch.setattr(m, "OUTPUT", tmp_path / "new-output")
    monkeypatch.setattr(m, "JOURNAL", tmp_path / "new-journal")
    m.JOURNAL.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        m.preflight(None)


def test_rejects_wrong_slot_order(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    runner = SimpleNamespace(verify=lambda *args: {"slots": [
        {"root_slot": 1, "condition": "off"}]})
    with pytest.raises(ValueError, match="slot"):
        m.preflight(runner)


def test_rejects_changed_instrumentation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(m, "PINS", {next(iter(m.PINS)): "0" * 64})
    with pytest.raises(ValueError, match="instrumentation changed"):
        m.preflight(None)


def isolated_preflight(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    monkeypatch.setattr(m, "OUTPUT", tmp_path / "new-output")
    monkeypatch.setattr(m, "JOURNAL", tmp_path / "new-journal")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"verified fixture")
    monkeypatch.setattr(m, "MANIFEST", manifest)
    monkeypatch.setattr(m, "MANIFEST_SHA", hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(m, "PINS", {})
    return {"slots": [{"root_slot": root, "condition": condition}
                      for root, condition in ((0, "off"), (0, "on"),
                                              (1, "off"), (1, "on"))]}


def test_rejects_manifest_tampering_before_verify(tmp_path, monkeypatch):
    isolated_preflight(tmp_path, monkeypatch)
    m.MANIFEST.write_bytes(b"tampered")
    runner = SimpleNamespace(verify=lambda *args: pytest.fail("verify must not run"))
    with pytest.raises(ValueError, match="manifest digest"):
        m.preflight(runner)


@pytest.mark.parametrize("occupied", ["OUTPUT", "JOURNAL"])
def test_rechecks_paths_after_verify(tmp_path, monkeypatch, occupied):
    manifest = isolated_preflight(tmp_path, monkeypatch)

    def verify(*args):
        getattr(m, occupied).touch()
        return manifest

    with pytest.raises(FileExistsError, match="appeared"):
        m.preflight(SimpleNamespace(verify=verify))


def test_isolated_success_does_not_create_outputs(tmp_path, monkeypatch):
    manifest = isolated_preflight(tmp_path, monkeypatch)
    result = m.preflight(SimpleNamespace(verify=lambda *args: manifest))
    assert result["status"] == "cpu_preflight_verified_not_authorized"
    assert not m.OUTPUT.exists()
    assert not m.JOURNAL.exists()
