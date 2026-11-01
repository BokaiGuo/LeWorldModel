import json
import multiprocessing
import os
import subprocess
import sys

import pytest

from experiments.spectral_reachability import e3_cell02_unified_gate_codex_20261030_v1 as gate


def compete(path, index, start, queue):
    start.wait()
    try:
        gate.temporary_gate(path, gate.ENTRYPOINTS[index % len(gate.ENTRYPOINTS)],
                            f'{index:064x}')
        queue.put('claimed')
    except FileExistsError:
        queue.put('blocked')


def test_cross_entry_atomic(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    start, queue = ctx.Event(), ctx.Queue()
    children = [ctx.Process(target=compete, args=(str(tmp_path), i, start, queue))
                for i in range(8)]
    for child in children:
        child.start()
    start.set()
    for child in children:
        child.join(30)
        assert child.exitcode == 0
    results = [queue.get(timeout=5) for _ in children]
    assert results.count('claimed') == 1
    assert results.count('blocked') == 7
    assert json.loads((tmp_path / gate.TEST_MARKER).read_bytes())['test_only'] is True


@pytest.mark.parametrize('name', ['old.claim', 'arbitrary-output', gate.TEST_MARKER])
@pytest.mark.parametrize('kind', ['file', 'symlink'])
def test_unknown_history_blocks_and_is_preserved(tmp_path, name, kind):
    target = tmp_path / name
    if kind == 'symlink':
        target.symlink_to(tmp_path / 'missing')
    else:
        target.write_bytes(b'historical')
    with pytest.raises(FileExistsError):
        gate.temporary_gate(tmp_path, gate.ENTRYPOINTS[0], 'a' * 64)
    assert os.path.lexists(target)
    assert not (tmp_path / gate.TEST_MARKER).exists() or name == gate.TEST_MARKER


def test_explicit_upstream_and_unknown_entrypoint(tmp_path):
    (tmp_path / 'known-upstream').write_bytes(b'fixture')
    with pytest.raises(ValueError):
        gate.temporary_gate(tmp_path, 'bypass.py', 'a' * 64, ('known-upstream',))
    assert gate.temporary_gate(tmp_path, gate.ENTRYPOINTS[1], 'a' * 64,
                               ('known-upstream',))['test_scope_claimed']


def test_artifact_appearing_after_claim_blocks_and_preserves(tmp_path, monkeypatch):
    actual = os.listdir
    count = 0
    def listdir(fd):
        nonlocal count
        count += 1
        if count == 2:
            (tmp_path / 'racing-output').write_bytes(b'fixture')
        return actual(fd)
    monkeypatch.setattr(gate.os, 'listdir', listdir)
    with pytest.raises(FileExistsError, match='appeared'):
        gate.temporary_gate(tmp_path, gate.ENTRYPOINTS[0], 'a' * 64)
    assert (tmp_path / gate.TEST_MARKER).exists()


@pytest.mark.parametrize('at', [1, 2])
def test_fsync_failure_preserves_claim(tmp_path, monkeypatch, at):
    actual, count = os.fsync, 0
    def fsync(fd):
        nonlocal count
        count += 1
        if count == at:
            raise OSError('injected')
        actual(fd)
    monkeypatch.setattr(gate.os, 'fsync', fsync)
    with pytest.raises(OSError, match='injected'):
        gate.temporary_gate(tmp_path, gate.ENTRYPOINTS[0], 'a' * 64)
    assert (tmp_path / gate.TEST_MARKER).exists()


def test_refuses_production_directory():
    with pytest.raises(ValueError):
        gate.temporary_gate(gate.candidate.RESULTS, gate.ENTRYPOINTS[0], 'a' * 64)


def test_audit_fails_closed_and_checks_fixed_paths(tmp_path, monkeypatch):
    claim, output = tmp_path / 'claim', tmp_path / 'output'
    claim.symlink_to(tmp_path / 'absent')
    monkeypatch.setattr(gate.candidate, 'CLAIM', claim)
    monkeypatch.setattr(gate.candidate, 'OUTPUT', output)
    monkeypatch.setattr(gate.dedup, 'candidate_names', lambda _: {'possible_conflicts': []})
    result = gate.audit()
    assert result['fixed_claim_present'] and not result['fixed_output_present']
    assert result['production_ready'] is False and result['fixed_denominator'] == 16


def test_entrypoint_drift_blocks(monkeypatch):
    census = iter([{'entry': 'a'}, {'entry': 'b'}])
    monkeypatch.setattr(gate, 'entrypoint_census', lambda: next(census))
    monkeypatch.setattr(gate.dedup, 'candidate_names', lambda _: {})
    with pytest.raises(ValueError, match='source changed'):
        gate.audit()


def test_real_audit_cli_nonzero():
    proc = subprocess.run([sys.executable, '-B', '-s', '-m', gate.__name__],
                          cwd=gate.candidate.ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    result = json.loads(proc.stdout)
    assert result['production_authorized'] is False
    assert result['bounded_names']['cell03_entries_opened_or_statted'] is False
    assert len(result['known_entrypoint_sha256']) == 6
