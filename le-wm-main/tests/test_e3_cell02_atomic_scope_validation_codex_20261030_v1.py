import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from experiments.spectral_reachability import e3_cell02_atomic_scope_validation_codex_20261030_v1 as gate


def contender(directory, index, start, results):
    start.wait()
    try:
        gate.temporary_claim(directory, f'entry-{index}', f'{index:064x}', f'out-{index}')
        results.put('won')
    except FileExistsError:
        results.put('occupied')


def test_distinct_entrypoints_contend_atomically(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    start, results = ctx.Event(), ctx.Queue()
    processes = [ctx.Process(target=contender, args=(str(tmp_path), i, start, results))
                 for i in range(8)]
    for proc in processes:
        proc.start()
    start.set()
    for proc in processes:
        proc.join(20)
        assert proc.exitcode == 0
    outcomes = [results.get(timeout=5) for _ in processes]
    assert outcomes.count('won') == 1
    assert outcomes.count('occupied') == 7
    record = json.loads((tmp_path / gate.MARKER).read_bytes())
    assert record['scope'] == gate.SCOPE and record['test_only'] is True


@pytest.mark.parametrize('kind', ['empty', 'corrupt', 'symlink'])
def test_existing_marker_is_never_replaced(tmp_path, kind):
    marker = tmp_path / gate.MARKER
    if kind == 'symlink':
        marker.symlink_to(tmp_path / 'missing')
    else:
        marker.write_bytes(b'' if kind == 'empty' else b'corrupt')
    with pytest.raises(FileExistsError):
        gate.temporary_claim(tmp_path, 'other', 'a' * 64, 'other')
    assert os.path.lexists(marker)
    if kind != 'symlink':
        assert marker.read_bytes() == (b'' if kind == 'empty' else b'corrupt')


@pytest.mark.parametrize('failure_at', [1, 2])
def test_fsync_failure_preserves_marker(tmp_path, monkeypatch, failure_at):
    real = os.fsync
    count = 0

    def fsync(fd):
        nonlocal count
        count += 1
        if count == failure_at:
            raise OSError('injected fsync failure')
        real(fd)

    monkeypatch.setattr(os, 'fsync', fsync)
    with pytest.raises(OSError, match='injected'):
        gate.temporary_claim(tmp_path, 'entry', 'a' * 64, 'out')
    assert (tmp_path / gate.MARKER).exists()
    with pytest.raises(FileExistsError):
        gate.temporary_claim(tmp_path, 'other', 'b' * 64, 'other')


def test_outside_temp_rejected():
    with pytest.raises(ValueError):
        gate.temporary_claim(gate.candidate.ROOT, 'entry', 'a' * 64, 'out')


def test_real_isolated_draft_and_manifest_drift(tmp_path, monkeypatch):
    monkeypatch.chdir(gate.candidate.ROOT)
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.delenv('PYTHONHOME', raising=False)
    monkeypatch.delenv('PYTHONSTARTUP', raising=False)
    monkeypatch.setenv('PYTHONNOUSERSITE', '1')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setattr(gate.preflight, 'MANIFEST_DIRECTORY', tmp_path)
    data = gate.preflight.manifest_bytes()
    manifest = tmp_path / 'e3_cell02_manifest_test.json'
    manifest.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    result = gate.validate_manifest(manifest, digest)
    assert result['isolated_draft_matches'] and not result['production_authorized']
    monkeypatch.setenv('E3_TEST_ENV_DRIFT', '1')
    with pytest.raises(ValueError, match='differs'):
        gate.validate_manifest(manifest, digest)
    with pytest.raises(ValueError, match='SHA mismatch'):
        gate.validate_manifest(manifest, '0' * 64)
    assert not (tmp_path / gate.MARKER).exists()


def test_manifest_subprocess_error_never_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.review, 'read_manifest', lambda *args: (b'{}', {}))
    def fail():
        raise RuntimeError('draft failed')
    monkeypatch.setattr(gate.preflight, 'manifest_bytes', fail)
    with pytest.raises(RuntimeError, match='draft failed'):
        gate.validate_manifest('unused', 'a' * 64)
    assert not (tmp_path / gate.MARKER).exists()


@pytest.mark.parametrize('drift', ['source', 'environment', 'interpreter', 'manifest'])
def test_drift_during_validation_fails_closed(monkeypatch, drift):
    real_sha = gate.candidate.file_sha
    calls = 0
    def file_sha(path):
        nonlocal calls
        calls += 1
        value = real_sha(path)
        return '0' * 64 if drift == 'source' and calls > 6 else value

    reads = 0
    def read_manifest(*args):
        nonlocal reads
        reads += 1
        return ((b'changed' if drift == 'manifest' and reads == 2 else b'{}'), {})

    def fresh():
        if drift == 'environment':
            monkeypatch.setenv('E3_DRIFT', 'changed')
        if drift == 'interpreter':
            monkeypatch.setattr(gate.sys, 'executable', 'changed')
        return b'{}'

    monkeypatch.setattr(gate.candidate, 'file_sha', file_sha)
    monkeypatch.setattr(gate.review, 'read_manifest', read_manifest)
    monkeypatch.setattr(gate.preflight, 'manifest_bytes', fresh)
    with pytest.raises(ValueError, match='changed|drift'):
        gate.validate_manifest('unused', 'a' * 64)


def test_cli_failure_is_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(gate.sys, 'argv', ['validator', '--manifest', '/missing',
                                           '--manifest-sha256', 'a' * 64])
    assert gate.main() == 1
    assert json.loads(capsys.readouterr().err)['production_authorized'] is False
