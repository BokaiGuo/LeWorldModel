import json
import fcntl
import os
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import e3_cell03_recovery_launcher_codex_20261030_v3 as m
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    for key, name in [('ROOT', 'root'), ('CLAIM', 'claim'), ('JOURNAL', 'journal'),
                      ('OUTPUT', 'output'), ('LOCK', 'lock')]:
        monkeypatch.setattr(m, key, tmp_path if key == 'ROOT' else tmp_path / name)
    m.LOCK.touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(m, 'verify', lambda sha: sha)
    return tmp_path


def test_denied_preclaim(fixture):
    with pytest.raises(ValueError, match='coordinated'):
        m.run('hash', 'not-confirmed')
    assert not m.CLAIM.exists()
    with gpu_lock(m.LOCK):
        with pytest.raises(BlockingIOError):
            m.run('hash', m.CONFIRM)
    assert not m.CLAIM.exists()


@pytest.mark.parametrize('code', [1, -9])
def test_child_failure_retained_and_no_retry(fixture, monkeypatch, code):
    def child(*args, **kwargs):
        assert kwargs['pass_fds']
        with pytest.raises(BlockingIOError):
            with gpu_lock(m.LOCK):
                pass
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(m.subprocess, 'run', child)
    with pytest.raises(RuntimeError, match='nonzero'):
        m.run('hash', m.CONFIRM)
    assert m.CLAIM.exists()
    assert [json.loads(line)['stage'] for line in m.JOURNAL.read_bytes().splitlines()] == [
        'claimed', 'child_intent', 'child_exit']
    with pytest.raises(FileExistsError):
        m.run('hash', m.CONFIRM)


def test_spawn_exception_preserved(fixture, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('exec failed')
    monkeypatch.setattr(m.subprocess, 'run', fail)
    with pytest.raises(OSError):
        m.run('hash', m.CONFIRM)
    assert json.loads(m.JOURNAL.read_bytes().splitlines()[-1])['stage'] == 'child_start_failure'
    assert m.CLAIM.exists()


def test_existing_symlink_blocks_before_claim(fixture):
    m.OUTPUT.symlink_to('missing')
    with pytest.raises(FileExistsError):
        m.run('hash', m.CONFIRM)
    assert not m.CLAIM.exists()


def test_zero_exit_missing_publication_fails_closed(fixture, monkeypatch):
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    with pytest.raises((ValueError, OSError)):
        m.run('hash', m.CONFIRM)
    assert json.loads(m.JOURNAL.read_bytes().splitlines()[-1])['stage'] == 'saved_inspect_failure'
    assert m.CLAIM.exists()


def test_manifest_sha_and_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'MANIFEST', tmp_path / 'manifest')
    body = {'candidate': {'environment_sha256': 'cpu', 'a': 1}, 'x': 3}
    sha = m.draft.write_exclusive(m.MANIFEST, body)
    monkeypatch.setattr(m, 'prepare_body',
                        lambda: {'candidate': {'environment_sha256': 'other', 'a': 1}, 'x': 3})
    assert m.verify(sha) == sha
    with pytest.raises(ValueError, match='SHA'):
        m.verify('0' * 64)
    monkeypatch.setattr(m, 'prepare_body',
                        lambda: {'candidate': {'environment_sha256': 'cpu', 'a': 2}, 'x': 3})
    with pytest.raises(ValueError, match='drift'):
        m.verify(sha)


def test_inherited_lock_survives_parent_descriptor_close(tmp_path):
    path = tmp_path / 'lock'
    path.touch()
    fd = os.open(path, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready_read, ready_write = os.pipe()
    stop_read, stop_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        os.close(stop_write)
        os.write(ready_write, b'1')
        os.read(stop_read, 1)
        os._exit(0)
    os.close(ready_write)
    os.close(stop_read)
    try:
        assert os.read(ready_read, 1) == b'1'
        os.close(fd)
        with pytest.raises(BlockingIOError):
            with gpu_lock(path):
                pass
    finally:
        os.write(stop_write, b'1')
        os.close(stop_write)
        os.close(ready_read)
        os.waitpid(pid, 0)
    with gpu_lock(path):
        pass
