import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import e3_cell02_deployment_preflight_codex_20261030_v1 as d


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setattr(d.candidate, 'CLAIM', tmp_path/'claim')
    monkeypatch.setattr(d.candidate, 'OUTPUT', tmp_path/'output')
    monkeypatch.setattr(d, 'MANIFEST_DIRECTORY', tmp_path)
    monkeypatch.setattr(d.candidate, 'check_environment', lambda: None)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    return tmp_path/'e3_cell02_manifest_test.json'


def test_resource_report_uses_production_constants(local):
    result = d.observe()
    assert result['disk_minimum_bytes'] == d.transport.CAP + d.transport.FLOOR
    assert result['contract']['category_caps_bytes'] == d.transport.LIMITS
    assert not result['production_authorized'] and not result['scientific_admission']
    assert any('GPU' in s for s in result['unverified'])


@pytest.mark.parametrize('block', ['claim', 'claim_symlink', 'output', 'cuda', 'disk', 'inodes'])
def test_local_blockers_never_make_manifest(local, monkeypatch, block):
    if block == 'claim':
        d.candidate.CLAIM.write_text('existing')
    elif block == 'claim_symlink':
        d.candidate.CLAIM.symlink_to('missing')
    elif block == 'output':
        d.candidate.OUTPUT.symlink_to('missing')
    elif block == 'cuda':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    elif block == 'disk':
        monkeypatch.setattr(d.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    else:
        monkeypatch.setattr(d.shutil, 'disk_usage', lambda _: SimpleNamespace(free=d.transport.CAP + d.transport.FLOOR))
        monkeypatch.setattr(d.os, 'statvfs', lambda _: SimpleNamespace(f_favail=0))
    monkeypatch.setattr(d, 'manifest_bytes', lambda: pytest.fail('draft should not run'))
    with pytest.raises(ValueError, match='blocked'):
        d.generate(local)
    assert not local.exists()


def test_change_during_draft_blocks(local, monkeypatch):
    def draft():
        d.candidate.CLAIM.write_text('external')
        return b'{}\n'
    monkeypatch.setattr(d, 'manifest_bytes', draft)
    with pytest.raises(ValueError, match='changed'):
        d.generate(local)
    assert not local.exists()


def test_exclusive_write_and_symlink_rejected(local):
    d.create_exclusive(local, b'first')
    with pytest.raises(FileExistsError):
        d.create_exclusive(local, b'second')
    assert local.read_bytes() == b'first'
    other = local.parent/'e3_cell02_manifest_link.json'
    other.symlink_to(local)
    with pytest.raises(FileExistsError):
        d.create_exclusive(other, b'second')
    assert local.read_bytes() == b'first'
    with pytest.raises(ValueError, match='basename'):
        d.create_exclusive(local.parent/'unrelated.json', b'x')
    with pytest.raises(ValueError, match='dedicated'):
        d.create_exclusive(local.parent.parent/'e3_cell02_manifest_other.json', b'x')


@pytest.mark.parametrize('stdout,returncode', [
    (b'', 2), (b'{}', 0), (b'{}\n', 0),
])
def test_bad_draft_rejected(monkeypatch, stdout, returncode):
    calls = []
    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=stdout, stderr=b'fixture error', returncode=returncode)
    monkeypatch.setattr(d.subprocess, 'run', fake_run)
    with pytest.raises((ValueError, RuntimeError)):
        d.manifest_bytes()
    assert calls[0][0][1:5] == ['-B', '-s', '-m', d.MODULE]
    assert 'env' not in calls[0][1]


def test_real_draft_is_exact_launcher_manifest_in_clean_child(local, monkeypatch):
    monkeypatch.setattr(d.candidate, 'check_environment',
                        lambda: None)
    # Subprocess uses the real launcher checks and real authority. This is a
    # CPU-only draft; no GPU probing or model/physics execution takes place.
    result = d.generate(local)
    data = local.read_bytes()
    manifest = json.loads(data)
    assert data == d.candidate.canonical(manifest) + b'\n'
    assert hashlib.sha256(data).hexdigest() == result['manifest_sha256']
    assert manifest['authority_sha256'] == d.candidate.AUTHORITY_SHA
    assert manifest['audit_code_sha256'] == d.candidate.sha(d.candidate.AUDIT.encode())
    assert manifest['parent_sources'] and manifest['precheck_sources']
    assert not result['production_authorized']
