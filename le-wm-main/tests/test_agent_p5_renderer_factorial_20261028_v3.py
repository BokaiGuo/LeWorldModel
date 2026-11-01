"""Exact origin checks only; no native imports/model/physics in these tests."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p5_renderer_factorial_20261028_v3 as m


@pytest.fixture
def origins(monkeypatch):
    paths = dict(environment='/env.py', pymunk='/overlay/pymunk/__init__.py',
                 pygame='/base/pygame/__init__.py')
    p = dict(origins=paths, bindings={v: 'a'*64 for v in paths.values()},
             pygame_distribution=dict(site='/base', version='2.6.1'))
    probe = dict(versions=dict(pygame=p['pygame_distribution']), native_source_sha256='a'*64,
                 imported_dependency_hashes={v: 'a'*64 for k, v in paths.items() if k != 'environment'})
    monkeypatch.setattr(m, 'sha', lambda path: 'a'*64)
    return p, probe


def test_inherited_pygame_and_overlay_pymunk(origins):
    p, probe = origins
    m.check_origins(p, *p['origins'].values(), probe)


@pytest.mark.parametrize('which', ['environment', 'pymunk', 'pygame'])
def test_arbitrary_module_origin_rejected(origins, which):
    p, probe = origins
    values = dict(p['origins'], **{which: '/different/__init__.py'})
    with pytest.raises(ValueError, match='exact bound module origins'):
        m.check_origins(p, *values.values(), probe)


@pytest.mark.parametrize('tamper', ['version', 'site', 'missing_hash', 'wrong_hash'])
def test_probe_disagreement_rejected(origins, tamper):
    p, original = origins
    probe = deepcopy(original)
    if tamper in ('version', 'site'):
        probe['versions']['pygame'][tamper] = 'wrong'
    elif tamper == 'missing_hash':
        probe['imported_dependency_hashes'].pop(p['origins']['pygame'])
    else:
        probe['imported_dependency_hashes'][p['origins']['pymunk']] = 'b'*64
    with pytest.raises(ValueError):
        m.check_origins(p, *p['origins'].values(), probe)


def test_live_hash_drift_rejected(origins, monkeypatch):
    p, probe = origins
    monkeypatch.setattr(m, 'sha', lambda path: 'b'*64)
    with pytest.raises(ValueError, match='imported file SHA'):
        m.check_origins(p, *p['origins'].values(), probe)


def test_authority_before_verify(monkeypatch):
    monkeypatch.setattr(m, 'verify', lambda *a: pytest.fail('must reject first'))
    with pytest.raises(ValueError, match='MAIN'):
        m.run('/manifest', 'a'*64, '/output', False)


def test_launch_binds_exact_manifest_output():
    b = m.launch_binding('/prepare', 'a'*64, '/output')
    assert b['manifest'] == '/prepare' and b['output'] == '/output'
    assert b['manifest_sha256'] == 'a'*64 and b['kernel_source_sha256'] == m.SOURCE_SHA
    assert b['neural_calls'] == 0


def test_actual_original_distribution_and_failure():
    p = m.read(m.PREP, m.PREP_SHA)
    runtime = m.read(p['runtime_manifest'])
    assert runtime['inherited']['pygame'] == dict(
        site=str(m.ROOT/'.venv/lib/python3.12/site-packages'), version='2.6.1')
    failed = m.read(m.FAILURE, m.FAILURE_SHA)
    assert failed['meter'] is None and failed['global_error'].endswith('overlay provenance')
    assert all(not row['shots'] for row in failed['rows'])
    assert m.sha(m.native.__file__) == m.SOURCE_SHA


def test_no_global_substitution():
    assert m.native.load_runtime.__module__ == m.native.__name__
    assert m.load_runtime is not m.native.load_runtime
    assert m.native.protocol()['Space_step'] == 0
