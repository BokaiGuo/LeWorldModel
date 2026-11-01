import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import e3_cell02_final_review_codex_20261030_v1 as review


@pytest.fixture
def review_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(review.preflight, 'MANIFEST_DIRECTORY', tmp_path)
    path = tmp_path/'e3_cell02_manifest_fixture.json'
    data = review.candidate.canonical({'schema': 'fixture', 'cell': 2})+b'\n'
    path.write_bytes(data)
    obs = {'claim_exists': False, 'output_exists': False, 'cuda_explicitly_disabled': False,
           'disk_minimum_met': True, 'available_inodes': 10, 'contract': {'cap': 1},
           'tool_sha256': 'a'*64, 'transport_source_sha256': 'b'*64}
    monkeypatch.setattr(review.preflight, 'observe', lambda: dict(obs))
    monkeypatch.setattr(review.preflight, 'manifest_bytes', lambda: data)
    return path, review.candidate.sha(data), obs


def test_review_matches_bytes_but_never_authorizes(review_fixture):
    path, sha, _ = review_fixture
    result = review.review(str(path), sha)
    assert result['fresh_draft_matches'] and all(result['checks_before'].values())
    assert result['unresolved_blockers'] and not result['production_ready']
    assert not result['production_authorized'] and not result['scientific_admission']


@pytest.mark.parametrize('field,value', [('claim_exists', True), ('output_exists', True),
                                      ('cuda_explicitly_disabled', True), ('disk_minimum_met', False),
                                      ('available_inodes', 0)])
def test_local_resource_blockers_prevent_draft(review_fixture, monkeypatch, field, value):
    path, sha, obs = review_fixture
    obs[field] = value
    monkeypatch.setattr(review.preflight, 'manifest_bytes', lambda: pytest.fail('blocked before draft'))
    with pytest.raises(ValueError, match='local checks failed'):
        review.review(str(path), sha)


def test_drift_between_snapshots_and_bad_manifest(review_fixture, monkeypatch):
    path, sha, obs = review_fixture
    with pytest.raises(ValueError, match='SHA mismatch'):
        review.review(str(path), '0'*64)
    monkeypatch.setattr(review.preflight, 'manifest_bytes', lambda: b'{"cell":3}\n')
    with pytest.raises(ValueError, match='differs'):
        review.review(str(path), sha)
    monkeypatch.setattr(review.preflight, 'manifest_bytes', path.read_bytes)
    calls = []
    def observe():
        calls.append(1)
        return dict(obs, output_exists=len(calls) > 1)
    monkeypatch.setattr(review.preflight, 'observe', observe)
    with pytest.raises(ValueError, match='post-draft'):
        review.review(str(path), sha)


def test_noncanonical_and_symlink_manifest_rejected(review_fixture, tmp_path):
    path, _, _ = review_fixture
    path.write_text('{"cell": 2}')
    with pytest.raises(ValueError, match='noncanonical'):
        review.read_manifest(path, review.candidate.file_sha(path))
    link = tmp_path/'e3_cell02_manifest_link.json'
    link.symlink_to(path)
    with pytest.raises(ValueError, match='canonical'):
        review.read_manifest(link, review.candidate.file_sha(path))


@pytest.mark.parametrize('field', ['contract', 'tool_sha256', 'transport_source_sha256'])
def test_review_rejects_source_or_contract_drift(review_fixture, monkeypatch, field):
    path, sha, obs = review_fixture
    calls = []
    def observe():
        calls.append(1)
        value = dict(obs)
        if len(calls) == 2:
            value[field] = {'cap': 2} if field == 'contract' else 'c'*64
        return value
    monkeypatch.setattr(review.preflight, 'observe', observe)
    with pytest.raises(ValueError, match='source or contract drift'):
        review.review(str(path), sha)


def test_manifest_changed_after_draft_is_rejected(review_fixture, monkeypatch):
    path, sha, _ = review_fixture
    data = path.read_bytes()
    def draft():
        path.write_bytes(b'{}\n')
        return data
    monkeypatch.setattr(review.preflight, 'manifest_bytes', draft)
    with pytest.raises(ValueError, match='SHA mismatch'):
        review.review(str(path), sha)


def test_review_rejects_own_source_drift(review_fixture, monkeypatch):
    path, sha, _ = review_fixture
    original = review.candidate.file_sha
    calls = []
    def digest(path):
        if Path(path) == Path(review.__file__).resolve():
            calls.append(1)
            return 'a'*64 if len(calls) == 1 else 'b'*64
        return original(path)
    monkeypatch.setattr(review.candidate, 'file_sha', digest)
    with pytest.raises(ValueError, match='source or contract drift'):
        review.review(str(path), sha)


def test_real_launcher_run_wiring_in_isolated_cpu_process(tmp_path):
    # Actual draft, authority, precheck, manifest comparison, claim and source
    # checks run; only CHILD and AUDIT are intercepted before process creation.
    script = r"""
import json, os, pathlib, subprocess, sys
sys.path.insert(0, sys.argv[1])
from experiments.spectral_reachability import e3_cell02_exclusive_candidate_codex_20261030_v1 as m
t=pathlib.Path(sys.argv[2])
m.CLAIM=t/'claim'; m.OUTPUT=t/'output'
real_run=subprocess.run
calls=[]
def intercept(args, **kwargs):
    if len(args)>4 and args[4] in (m.CHILD,m.AUDIT):
        assert m.CLAIM.is_file()
        calls.append('child' if args[4] == m.CHILD else 'audit')
        assert args[5] == str(m.ROOT)
        if args[4] == m.CHILD:
            assert args[6:] == [str(m.AUTHORITY),m.AUTHORITY_SHA,str(m.OUTPUT)]
        else:
            assert args[6:] == [str(m.OUTPUT),m.AUTHORITY_SHA,str(m.AUTHORITY)]
        return subprocess.CompletedProcess(args,0)
    assert len(args)>4 and args[4] == m.PRECHECK
    return real_run(args, **kwargs)
m.subprocess.run=intercept
draft=m.draft()
manifest=t/'manifest.json'; manifest.write_bytes(m.canonical(draft))
digest=m.file_sha(manifest)
result=m.run(manifest,digest,m.CONFIRM)
assert result['transport_complete'] and not result['scientific_admission']
assert calls==['child','audit'] and not m.OUTPUT.exists()
try: m.run(manifest,digest,m.CONFIRM)
except FileExistsError: pass
else: raise AssertionError('repeat claim accepted')
assert calls==['child','audit']
print(json.dumps({'stubbed_child_and_audit':calls,'claim_retained':m.CLAIM.exists(),
                  'real_precheck':True,'gpu_calls':0,'physical_calls':0}))
"""
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'CUDA_VISIBLE_DEVICES'):
        env.pop(key, None)
    proc = subprocess.run([sys.executable, '-B', '-s', '-c', script, str(review.candidate.ROOT), str(tmp_path)],
                          cwd=review.candidate.ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout.splitlines()[-1])
    assert result['stubbed_child_and_audit'] == ['child', 'audit']
    assert result['claim_retained'] and result['gpu_calls'] == result['physical_calls'] == 0


def test_review_real_fresh_launcher_draft_with_temporary_resource_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(review.preflight, 'MANIFEST_DIRECTORY', tmp_path)
    monkeypatch.setattr(review.candidate, 'CLAIM', tmp_path/'claim')
    monkeypatch.setattr(review.candidate, 'OUTPUT', tmp_path/'output')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    data = review.preflight.manifest_bytes()
    path = tmp_path/'e3_cell02_manifest_real_draft.json'
    path.write_bytes(data)
    result = review.review(str(path), review.candidate.sha(data))
    assert result['fresh_draft_matches'] and not result['production_authorized']
    assert result['before']['claim_exists'] is False and result['after']['output_exists'] is False
