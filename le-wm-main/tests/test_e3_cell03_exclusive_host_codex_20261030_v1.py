import concurrent.futures
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import e3_cell03_exclusive_host_codex_20261030_v1 as m


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'CLAIM', tmp_path / 'claim')
    monkeypatch.setattr(m, 'OUTPUT', tmp_path / 'output')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(m.precheck, 'check', lambda *_: None)
    calls = []
    monkeypatch.setattr(m.subprocess, 'run',
                        lambda args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0))
    return calls


def test_one_shot_and_fixed_binding(setup):
    assert m.run(m.CONFIRM) == {'complete': True, 'child_exit_code': 0}
    assert len(setup) == 1
    assert setup[0] == [m.sys.executable, '-B', '-c', m.CHILD, str(m.AUTHORITY),
                        m.AUTHORITY_SHA, str(m.OUTPUT)]
    assert json.loads(m.CLAIM.read_text())['authority_sha256'] == m.AUTHORITY_SHA
    with pytest.raises(FileExistsError):
        m.run(m.CONFIRM)
    assert len(setup) == 1


def test_concurrent_claim_only_one_runner(setup):
    def attempt():
        try:
            m.run(m.CONFIRM)
            return True
        except FileExistsError:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: attempt(), range(8)))
    assert sum(results) == 1
    assert len(setup) == 1


def test_failure_retains_claim(setup, monkeypatch):
    def fail(*_, **__):
        raise RuntimeError('partial')
    monkeypatch.setattr(m.subprocess, 'run', fail)
    with pytest.raises(RuntimeError):
        m.run(m.CONFIRM)
    assert m.CLAIM.exists()
    with pytest.raises(FileExistsError):
        m.run(m.CONFIRM)


def test_checks_before_claim(setup, monkeypatch):
    with pytest.raises(ValueError):
        m.run('not coordinated')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    with pytest.raises(ValueError):
        m.run(m.CONFIRM)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    monkeypatch.setattr(m.precheck, 'check', lambda *_: (_ for _ in ()).throw(ValueError('source drift')))
    with pytest.raises(ValueError, match='source drift'):
        m.run(m.CONFIRM)
    assert not m.CLAIM.exists()
    assert not setup


@pytest.mark.parametrize('returncode,expected', [(1, 1), (7, 7), (-9, 137)])
def test_child_failure_is_nonzero_and_claim_retained(setup, monkeypatch, returncode, expected):
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=returncode))
    with pytest.raises(SystemExit) as exc:
        m.run(m.CONFIRM)
    assert exc.value.code == expected
    assert m.CLAIM.exists()


def test_existing_output_including_dangling_symlink_blocks(setup):
    m.OUTPUT.symlink_to('missing')
    with pytest.raises(FileExistsError):
        m.run(m.CONFIRM)
    assert not m.CLAIM.exists()
    assert not setup


def test_fresh_child_real_authority_and_census_then_partial_exit():
    # Real project/venv bindings and census, without model loading or GPU.
    prefix = """
import torch
from experiments import gpu_preflight
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as r
def fixture_run(path, sha, output, authorization):
    import sys, types
    binding = r.authority(path, sha)
    sources = dict(binding['prepared']['bindings_sha256'])
    r.launch.host.source.merge(sources, binding['additional_sources'])
    entries = r.launch.module_census.snapshot(sources)
    assert entries
    assert not any('e3_cell03_exclusive_host_codex_20261030_v1.py' in e['path'] for e in entries)
    assert not any('e3_cell03_host_cpu_precheck_codex_20261030_v1.py' in e['path'] for e in entries)
    injected = types.ModuleType('fixture_unsealed')
    injected.__file__ = str(__import__('pathlib').Path.cwd() / 'experiments/spectral_reachability/e3_cell03_exclusive_host_codex_20261030_v1.py')
    sys.modules['fixture_unsealed'] = injected
    try:
        r.launch.module_census.snapshot(sources)
        raise AssertionError('unsealed module accepted')
    except ValueError as exc:
        assert 'loaded module not presealed' in str(exc)
    finally:
        del sys.modules['fixture_unsealed']
    print('REAL_CENSUS_OK', len(entries), flush=True)
    return {'complete': False, 'error_type': 'fixture_partial'}
r.run = fixture_run
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([m.sys.executable, '-B', '-c', prefix + m.CHILD,
                             str(m.AUTHORITY), m.AUTHORITY_SHA, '/unused'],
                            env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'REAL_CENSUS_OK' in result.stdout
    assert '"complete": false' in result.stdout


def test_actual_child_exception_has_nonzero_exit():
    prefix = """
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as r
def fail(*args):
    raise RuntimeError('fixture child exception')
r.run = fail
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([m.sys.executable, '-B', '-c', prefix + m.CHILD,
                             str(m.AUTHORITY), m.AUTHORITY_SHA, '/unused'],
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0
    assert 'fixture child exception' in result.stderr
