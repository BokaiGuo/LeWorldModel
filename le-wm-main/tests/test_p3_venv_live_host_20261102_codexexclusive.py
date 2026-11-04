"""CPU-only preflight tests; no live neural or GPU operations."""
import json

import pytest

from experiments.spectral_reachability import p3_venv_live_host_20261102_codexexclusive as m


def test_verify_reconstructs_and_rejects_changed_manifest(tmp_path, monkeypatch):
    path = tmp_path / "prepare.json"
    plan = {"new_environment_binding": "binding", "new_environment_binding_sha256": "hash",
            "payload": 1}
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(m, "material", lambda *_: (plan, ["a", "b"], ["c", "d"]))
    sha = m.host.prior.sha(path)
    verified, tasks, banks = m.verify(path, sha)
    assert verified == plan and len(tasks) == len(banks) == 2
    monkeypatch.setattr(m, "material", lambda *_: ({**plan, "payload": 2}, [], []))
    with pytest.raises(ValueError, match="drift"):
        m.verify(path, sha)
    with pytest.raises(ValueError, match="SHA"):
        m.verify(path, "0" * 64)


def test_no_cuda_without_explicit_coordinated_authority(monkeypatch):
    def forbidden(*_):
        raise AssertionError("preflight or CUDA must not start")
    monkeypatch.setattr(m, "verify", forbidden)
    with pytest.raises(ValueError, match="authority"):
        m.launch("unused", "unused", "unused", "wrong")


def test_output_requires_exclusive_new_path(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / (m.NAME + "_PREP_01.json")
    assert m.output(path, "_PREP_") == path
    path.write_text("{}")
    with pytest.raises(ValueError, match="exclusive"):
        m.output(path, "_PREP_")
