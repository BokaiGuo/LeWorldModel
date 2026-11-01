import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import e3_cell02_exclusive_candidate_codex_20261030_v1 as m
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub
from experiments.spectral_reachability import agent_e3_loaded_module_census_20261029_v1 as census


AUDIT_FIXTURE = """
import sys
sys.path.insert(0, sys.argv[1])
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as r
r.authority = lambda path, digest: {'cell': 2, 'prepared': {'bindings_sha256': {sys.argv[5]: sys.argv[6]}}, 'additional_sources': {}}
"""


def audit_fixture(output, source, digest, env):
    return subprocess.run([sys.executable, '-B', '-s', '-c', AUDIT_FIXTURE + m.AUDIT,
                           str(m.ROOT), str(output), m.AUTHORITY_SHA, str(m.AUTHORITY),
                           str(source), digest], env=env, capture_output=True, text=True)


def fixture_source(tmp_path):
    source = tmp_path/'source.py'
    source.write_text('fixture\n')
    return source, m.file_sha(source)


def record_census(store, source, digest, stages=('before', 'after')):
    entries = [{'module': 'fixture', 'path': str(source), 'sha256': digest}]
    for stage in stages:
        census.record(store, stage, entries)


@pytest.fixture
def fake_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'CLAIM', tmp_path/'claim')
    monkeypatch.setattr(m, 'OUTPUT', tmp_path/'output')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    manifest = {'parent_sources': {'source': 'hash'}, 'environment_sha256': 'environment',
                'python_invocation': m.sys.executable, 'python_executable': str(Path(m.sys.executable).resolve()),
                'python_sha256': m.file_sha(Path(m.sys.executable).resolve()), 'precheck_sources': {}}
    monkeypatch.setattr(m, 'draft', lambda: manifest)
    monkeypatch.setattr(m, 'module_sources', lambda: manifest['parent_sources'])
    monkeypatch.setattr(m, 'environment_sha', lambda: manifest['environment_sha256'])
    path = tmp_path/'manifest.json'
    path.write_bytes(m.canonical(manifest))
    calls = []
    monkeypatch.setattr(m.subprocess, 'run',
                        lambda args, **kw: calls.append(args) or SimpleNamespace(returncode=0))
    return path, m.file_sha(path), calls


def test_claim_is_one_shot_and_subprocesses_separate(fake_launch):
    path, digest, calls = fake_launch
    assert m.run(path, digest, m.CONFIRM)['transport_complete']
    assert len(calls) == 2 and calls[0][4] == m.CHILD and calls[1][4] == m.AUDIT
    assert json.loads(m.CLAIM.read_bytes())['manifest_sha256'] == digest
    with pytest.raises(FileExistsError):
        m.run(path, digest, m.CONFIRM)
    assert len(calls) == 2


@pytest.mark.parametrize('phase,code', [(0, 1), (0, -9), (1, 3)])
def test_failures_keep_claim_and_propagate(fake_launch, monkeypatch, phase, code):
    path, digest, calls = fake_launch
    def subprocess_run(args, **_):
        calls.append(args)
        return SimpleNamespace(returncode=code if len(calls)-1 == phase else 0)
    monkeypatch.setattr(m.subprocess, 'run', subprocess_run)
    with pytest.raises(SystemExit) as e:
        m.run(path, digest, m.CONFIRM)
    assert e.value.code == (code if code > 0 else 128-code)
    assert m.CLAIM.exists()
    with pytest.raises(FileExistsError):
        m.run(path, digest, m.CONFIRM)


def test_preclaim_failures_do_not_claim(fake_launch, monkeypatch):
    path, digest, calls = fake_launch
    with pytest.raises(ValueError):
        m.run(path, digest, 'no')
    with pytest.raises(ValueError, match='manifest hash'):
        m.run(path, '0'*64, m.CONFIRM)
    monkeypatch.setattr(m, 'draft', lambda: {'changed': True})
    with pytest.raises(ValueError, match='differs'):
        m.run(path, digest, m.CONFIRM)
    assert not m.CLAIM.exists() and not calls


def test_output_symlink_blocks(fake_launch):
    path, digest, calls = fake_launch
    m.OUTPUT.symlink_to('missing')
    with pytest.raises(FileExistsError):
        m.run(path, digest, m.CONFIRM)
    assert not m.CLAIM.exists() and not calls


def test_claim_race_only_one(tmp_path):
    path = tmp_path/'claim'
    def attempt(_):
        try:
            m.claim_once(path, {'a': 1})
            return True
        except FileExistsError:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(8))) == 1


def test_postclaim_source_drift_cannot_launch(fake_launch, monkeypatch):
    path, digest, calls = fake_launch
    monkeypatch.setattr(m, 'module_sources', lambda: {'other': 'hash'})
    with pytest.raises(ValueError, match='changed'):
        m.run(path, digest, m.CONFIRM)
    assert m.CLAIM.exists() and not calls


def test_precheck_source_drift_after_claim(fake_launch, monkeypatch, tmp_path):
    path, digest, calls = fake_launch
    source = tmp_path/'source.py'
    source.write_text('old')
    manifest = json.loads(path.read_bytes())
    manifest['precheck_sources'] = {str(source): m.file_sha(source)}
    path.write_bytes(m.canonical(manifest))
    digest = m.file_sha(path)
    monkeypatch.setattr(m, 'draft', lambda: manifest)
    original = m.claim_once
    def claim_then_change(claim, record):
        original(claim, record)
        source.write_text('new')
    monkeypatch.setattr(m, 'claim_once', claim_then_change)
    with pytest.raises(ValueError, match='precheck source changed'):
        m.run(path, digest, m.CONFIRM)
    assert m.CLAIM.exists() and not calls


def test_failed_claim_fsync_retains_marker_and_no_child(fake_launch, monkeypatch):
    path, digest, calls = fake_launch
    def fail(_):
        raise OSError('fsync failed')
    monkeypatch.setattr(m.os, 'fsync', fail)
    with pytest.raises(OSError, match='fsync failed'):
        m.run(path, digest, m.CONFIRM)
    assert m.CLAIM.exists() and not calls


@pytest.mark.parametrize('complete,expected', [(False, 1), (True, 0)])
def test_real_child_status_code_without_gpu(complete, expected):
    prefix = """
import sys
sys.path.insert(0, sys.argv[1])
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as r
r.run = lambda *args: {'complete': %s}
""" % complete
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-B', '-s', '-c', prefix + m.CHILD, str(m.ROOT),
                             str(m.AUTHORITY), m.AUTHORITY_SHA, '/unused'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stderr


@pytest.mark.parametrize('status,fail_record,cell,expected', [
    ('completed', False, 2, 0), ('partial', False, 2, 1),
    ('completed', True, 2, 1), ('completed', False, 1, 1)])
def test_real_transport_audit_in_cpu_subprocess(tmp_path, status, fail_record, cell, expected):
    source, digest = fixture_source(tmp_path)
    output = tmp_path/'job'
    store = pub.Store(output)
    store.event('physical', 'record', 'launcher_authority_verified', store.index,
                {'cell': cell, 'manifest_sha256': m.AUTHORITY_SHA, 'manifest_path': str(m.AUTHORITY)})
    record_census(store, source, digest)
    if fail_record:
        store.event('physical', 'record', 'launcher_failure', store.index, {})
    store.event('physical', 'record', 'launcher_guarded_return', store.index, {'cell': cell})
    store.publish(status)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1')
    result = audit_fixture(output, source, digest, env)
    assert (result.returncode == 0) == (expected == 0), result.stderr
    if expected == 0:
        assert json.loads(result.stdout)['census']['after'] == 1


@pytest.mark.parametrize('fault', ['missing_before', 'missing_after', 'wrong_binding', 'changed_source'])
def test_complete_transport_requires_valid_saved_census(tmp_path, fault):
    source, digest = fixture_source(tmp_path)
    output = tmp_path/'job'
    store = pub.Store(output)
    store.event('physical', 'record', 'launcher_authority_verified', store.index,
                {'cell': 2, 'manifest_sha256': m.AUTHORITY_SHA, 'manifest_path': str(m.AUTHORITY)})
    stages = tuple(s for s in ('before', 'after') if fault != 'missing_' + s)
    record_census(store, source, digest, stages)
    store.event('physical', 'record', 'launcher_guarded_return', store.index, {'cell': 2})
    store.publish('completed')
    assert pub.consume(str(output), m.file_sha(output/'report.json'),
                       m.file_sha(output/'completion.json'))['complete'] is True
    if fault == 'changed_source':
        source.write_text('changed\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1')
    audit = audit_fixture(output, source, '0'*64 if fault == 'wrong_binding' else digest, env)
    assert audit.returncode != 0
    assert 'census' in audit.stderr or 'module' in audit.stderr


@pytest.mark.parametrize('partial', [False, True])
def test_real_lifecycle_publication_child_exit_and_audit(tmp_path, partial):
    # The real lifecycle and Store are exercised, but episode and authority are
    # fixtures: no CUDA, model loading, neural inference, or native environment.
    source, digest = fixture_source(tmp_path)
    prefix = """
import sys
sys.path.insert(0, sys.argv[1])
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as r
def authority(path, digest):
    return {'cell': 2, 'manifest_path': path, 'manifest_sha': digest}
def perform(binding, store):
    if %s: raise ValueError('fixture episode failure')
    entries = [{'module': 'fixture', 'path': %r, 'sha256': %r}]
    r.launch.module_census.record(store, 'before', entries)
    r.launch.module_census.record(store, 'after', entries)
    return {'status': 'completed20'}
r.authority = authority
r.perform = perform
""" % (partial, str(source), digest)
    output = tmp_path/'job'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1')
    child = subprocess.run([sys.executable, '-B', '-s', '-c', prefix + m.CHILD,
                            str(m.ROOT), str(m.AUTHORITY), m.AUTHORITY_SHA, str(output)],
                           env=env, capture_output=True, text=True, timeout=60)
    assert child.returncode == (1 if partial else 0), child.stdout + child.stderr
    report = json.loads((output/'report.json').read_bytes())
    completion = json.loads((output/'completion.json').read_bytes())
    assert report['status'] == ('partial' if partial else 'completed')
    assert completion['complete'] is not partial
    audit = audit_fixture(output, source, digest, env)
    assert (audit.returncode != 0) is partial, audit.stdout + audit.stderr


@pytest.mark.parametrize('fault', ['missing_completion', 'tampered_journal', 'overrun_marker'])
def test_transport_audit_rejects_damaged_or_incomplete_publication(tmp_path, fault):
    source, digest = fixture_source(tmp_path)
    output = tmp_path/'job'
    store = pub.Store(output)
    store.event('physical', 'record', 'launcher_authority_verified', store.index,
                {'cell': 2, 'manifest_sha256': m.AUTHORITY_SHA, 'manifest_path': str(m.AUTHORITY)})
    record_census(store, source, digest)
    store.event('physical', 'record', 'launcher_guarded_return', store.index, {'cell': 2})
    store.publish('completed')
    if fault == 'missing_completion':
        (output/'completion.json').unlink()
    elif fault == 'tampered_journal':
        with (output/'events.jsonl').open('ab') as stream:
            stream.write(b'{}\n')
    else:
        (output/'publication_overrun.json').write_text('{}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1')
    audit = audit_fixture(output, source, digest, env)
    assert audit.returncode != 0


def test_cpu_real_loader_census_against_static_authority_and_precheck_manifest():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
        env.pop(key, None)
    draft = subprocess.run([sys.executable, '-B', '-s', '-m',
                            'experiments.spectral_reachability.e3_cell02_exclusive_candidate_codex_20261030_v1',
                            'draft'], cwd=m.ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert draft.returncode == 0, draft.stderr
    manifest = json.loads(draft.stdout)
    assert manifest['authority_sha256'] == m.AUTHORITY_SHA
    code = """
import json, sys, types, torch
sys.path.insert(0, sys.argv[1])
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as h
manifest = json.load(sys.stdin)
b = h.authority(sys.argv[2], sys.argv[3])
sources = dict(b['prepared']['bindings_sha256'])
h.launch.host.source.merge(sources, b['additional_sources'])
for path, digest in manifest['precheck_sources'].items():
    if path in sources:
        assert sources[path] == digest, 'manifest and authority conflict: ' + path
class Sink:
    def event(self, *a, **k): pass
    def put(self, *a, **k): pass
loader = h.launch.host.primitives.native.loader
# CPU-only fixture seam: retain actual checkpoint bytes and tensor hash, but
# replace the original CUDA-device assertion in this disposable child only.
def cpu_frozen(model):
    assert all(not mod.training for mod in model.modules())
    assert all(not v.requires_grad and v.device.type == 'cpu' for v in model.parameters())
loader.enc.frozen = cpu_frozen
model, digest = loader.loaded_model(b['prepared'], torch, torch.device('cpu'), Sink(), b['prepare_sha'])
assert digest == b['prepared']['expected_model_hash']
entries = h.launch.module_census.snapshot(sources)
assert entries and all(sources[e['path']] == e['sha256'] for e in entries)
assert all(not e['path'].endswith('e3_cell02_exclusive_candidate_codex_20261030_v1.py') for e in entries)
for action in ('missing', 'wrong_sha'):
    bad = dict(sources)
    key = entries[0]['path']
    if action == 'missing': del bad[key]
    else: bad[key] = '0' * 64
    try:
        h.launch.module_census.snapshot(bad)
        raise AssertionError('census accepted ' + action)
    except ValueError:
        pass
only_runtime = sorted({e['path'] for e in entries} - set(manifest['precheck_sources']))
assert only_runtime, 'expected actual loader-only runtime modules'
print(json.dumps({'census_count':len(entries), 'runtime_only_count':len(only_runtime),
                  'cpu_model_loaded':True, 'neural_forward_calls':0, 'physics_calls':0}))
"""
    proc = subprocess.run([sys.executable, '-B', '-s', '-c', code, str(m.ROOT),
                           str(m.AUTHORITY), m.AUTHORITY_SHA],
                          cwd=m.ROOT, env=env, input=json.dumps(manifest), capture_output=True,
                          text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    evidence = json.loads(proc.stdout.splitlines()[-1])
    assert evidence['runtime_only_count'] > 0
    print('CPU_CENSUS', json.dumps(evidence, sort_keys=True))
