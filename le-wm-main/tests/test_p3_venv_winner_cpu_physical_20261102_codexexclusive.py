import pytest

from experiments.spectral_reachability import p3_venv_winner_cpu_physical_20261102_codexexclusive as m


def test_run_requires_authority_and_hidden_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="authority"):
        m.run("x", "x", "x", "wrong")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(ValueError, match="hidden"):
        m.run("x", "x", "x", m.AUTH)


def test_output_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / (m.NAME + "_RUN_01")
    assert m.output(path, "_RUN_") == path
    path.mkdir()
    with pytest.raises(ValueError, match="exclusive"):
        m.output(path, "_RUN_")


def test_verify_detects_prepare_drift(tmp_path, monkeypatch):
    path = tmp_path / "prepare.json"
    path.write_text("{}")
    monkeypatch.setattr(m, "material", lambda: ({"changed": True}, None))
    with pytest.raises(ValueError, match="drift"):
        m.verify(path, m.moving.sha(path))
