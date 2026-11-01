import os
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import e3_non03_host_readiness_codex_20261030_v1 as m


def test_complete_denominator_and_per_cell_fail_closed(monkeypatch):
    rows = [{'index': i, 'cell_id': str(i), 'online_jobs': [], 'feedback_authorities': []}
            for i in range(16) if i != 3]
    monkeypatch.setattr(m.coverage, 'inventory',
                        lambda _: {'rows': rows, 'unassigned_or_invalid': []})
    visited = []
    def check(i):
        visited.append(i)
        if i == 6:
            raise ValueError('corrupt source')
        return {'scientific_admission': False}
    monkeypatch.setattr(m, 'control5_saved', check)
    result = m.inspect_all()
    assert result['fixed_denominator'] == len(result['rows']) == 16
    assert result['rows'][3]['state'] == 'excluded_not_inspected'
    assert visited == list(m.MISSING) and 3 not in visited
    assert result['rows'][6]['missing_online_preconditions']['state'] == 'saved_control5_join_failed'
    assert not result['production_authorized'] and not result['scientific_admission']


def test_new_terminal_prevents_stale_missing_classification(monkeypatch):
    rows = [{'index': i, 'cell_id': str(i), 'online_jobs': [], 'feedback_authorities': []}
            for i in range(16) if i != 3]
    rows[2]['online_jobs'] = [{'terminal_status_recorded': 'completed20'}]
    monkeypatch.setattr(m.coverage, 'inventory',
                        lambda _: {'rows': rows, 'unassigned_or_invalid': []})
    monkeypatch.setattr(m, 'control5_saved', lambda _: {'scientific_admission': False})
    result = m.inspect_all()
    assert result['rows'][2]['missing_online_preconditions']['state'] == 'new_terminal_found_refresh_scope'


def test_cell02_saved_guard_and_wrong_inputs(monkeypatch):
    host = m.prep.host
    body = {'root_sha256': m.ROOT_SHA, 'horizon_packets': 3,
            'domain': {'name': 'Dxy'}, 'feedback_boundary': 5}
    first = np.zeros((5, 2), dtype=np.float32)
    binding = {'cell': 2, 'args': (), 'resolver': {'path': 'r', 'sha256': 's'}}
    setup = {'original_root_sha': m.ROOT_SHA,
             'expected_cp5': {'step': 5, 'pixel_sha256': m.CP5_SHA},
             'prefix5': np.zeros((5, 2), dtype=np.float32)}
    monkeypatch.setattr(m.prep, 'verify', lambda *_: {'task': 'r0.g0.h3.dDxy'})
    monkeypatch.setattr(host, 'authority', lambda *_: binding)
    monkeypatch.setattr(host.launch.episode.old.authority, 'load',
                        lambda *_: {'body': body, 'first5': first})
    monkeypatch.setattr(host.launch.episode.old.setup_authority, 'resolve', lambda *_: setup)
    monkeypatch.setattr(host.launch.host, 'sha', lambda *_: m.AUTHORITY_SHA)
    assert m.cell02()['saved_setup_joined']
    body['domain']['name'] = 'Dx'
    with pytest.raises(ValueError, match='H3 Dxy'):
        m.cell02()


def test_real_cell02_child_census_with_pinned_authority():
    code = """
import sys, types, torch
from experiments import gpu_preflight
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as h
b = h.authority(sys.argv[1], sys.argv[2])
assert b['cell'] == 2
sources = dict(b['prepared']['bindings_sha256'])
h.launch.host.source.merge(sources, b['additional_sources'])
entries = h.launch.module_census.snapshot(sources)
assert entries and not any('e3_non03_host_readiness_codex_20261030_v1.py' in x['path'] for x in entries)
x = types.ModuleType('unsealed_fixture')
x.__file__ = sys.argv[3]
sys.modules['unsealed_fixture'] = x
try:
    h.launch.module_census.snapshot(sources)
    raise AssertionError('unsealed module accepted')
except ValueError as e:
    assert 'loaded module not presealed' in str(e)
print('REAL_CELL02_CENSUS_OK')
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1')
    p = subprocess.run([sys.executable, '-B', '-c', code, str(m.AUTHORITY), m.AUTHORITY_SHA,
                        str(m.ROOT/'experiments/spectral_reachability/e3_non03_host_readiness_codex_20261030_v1.py')],
                       env=env, text=True, capture_output=True, timeout=180)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'REAL_CELL02_CENSUS_OK' in p.stdout
