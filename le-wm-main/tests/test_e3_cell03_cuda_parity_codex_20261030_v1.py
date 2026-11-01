import json
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import e3_cell03_cuda_parity_codex_20261030_v1 as m
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'OUTPUT', tmp_path / 'run')
    monkeypatch.setattr(m, 'LOCK', tmp_path / 'lock')
    m.LOCK.touch()
    monkeypatch.setattr(m, 'verify_manifest', lambda sha: {})
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    return tmp_path


def test_lock_conflict_no_output(fixture):
    with gpu_lock(m.LOCK):
        with pytest.raises(BlockingIOError):
            m.run('sha', m.CONFIRM)
    assert not m.OUTPUT.exists()


def test_nonzero_child_preserves_and_never_overwrites(fixture, monkeypatch):
    def child(args, **kwargs):
        assert kwargs['pass_fds'] and args[3] == m.MODULE
        with pytest.raises(BlockingIOError):
            with gpu_lock(m.LOCK):
                pass
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(m.subprocess, 'run', child)
    with pytest.raises(RuntimeError, match='failed'):
        m.run('sha', m.CONFIRM)
    assert json.loads((m.OUTPUT / 'parent.json').read_bytes())['child_returncode'] == 7
    with pytest.raises(FileExistsError):
        m.run('sha', m.CONFIRM)


def test_timeout_preserved(fixture, monkeypatch):
    def timeout(*a, **k):
        raise m.subprocess.TimeoutExpired('child', m.WALL)
    monkeypatch.setattr(m.subprocess, 'run', timeout)
    with pytest.raises(RuntimeError):
        m.run('sha', m.CONFIRM)
    assert json.loads((m.OUTPUT / 'parent.json').read_bytes())['timeout'] is True


def test_child_failure_preserved_and_child_claim_exclusive(fixture, monkeypatch):
    m.OUTPUT.mkdir()
    monkeypatch.setattr(m, 'verify_manifest', lambda _: (_ for _ in ()).throw(ValueError('drift')))
    with pytest.raises(ValueError, match='drift'):
        m.execute(m.OUTPUT, 'sha')
    assert json.loads((m.OUTPUT / 'failure.json').read_bytes())['error_type'] == 'ValueError'
    with pytest.raises(FileExistsError):
        m.execute(m.OUTPUT, 'sha')


def test_sink_cap(fixture, monkeypatch):
    m.OUTPUT.mkdir()
    monkeypatch.setattr(m, 'OUTPUT_CAP', 8)
    with pytest.raises(ValueError, match='cap'):
        m.Sink(m.OUTPUT).event({'stage': 'too large'})
