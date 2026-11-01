from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from experiments.spectral_reachability import e3_cell02_caps_dedup_codex_20261030_v1 as c


@pytest.fixture(scope='module')
def actual_inputs():
    return (c.pinned_json(c.candidate.AUTHORITY, c.candidate.AUTHORITY_SHA),
            c.pinned_json(c.PREPARED, c.PREPARED_SHA))


def manifest():
    m = c.candidate
    return {'schema': m.SCHEMA, 'cell': 2, 'root': str(m.ROOT), 'claim': str(m.CLAIM),
            'output': str(m.OUTPUT), 'authority': str(m.AUTHORITY), 'authority_sha256': m.AUTHORITY_SHA,
            'precheck_code_sha256': m.sha(m.PRECHECK.encode()), 'child_code_sha256': m.sha(m.CHILD.encode()),
            'audit_code_sha256': m.sha(m.AUDIT.encode()),
            'python_invocation': 'fixture', 'python_executable': 'fixture',
            'python_sha256': '0'*64, 'environment_sha256': '0'*64, 'precheck_sources': {},
            'parent_sources': {str(Path(m.__file__).resolve()): m.file_sha(Path(m.__file__).resolve())},
            'production_authorization': 'external_decision_required', 'scientific_admission': False}


def test_actual_authority_prepared_caps_and_binding(actual_inputs):
    result = c.contract(*actual_inputs)
    assert result['physical_predispatch_caps']['public_step'] == 30
    assert result['physical_predispatch_caps']['physics_step'] == 304
    assert result['numerical_per_node_success_expectation']['predict'] == 204
    assert result['online_two_node_success_expectation'] == {'predict': 408, 'reverse': 64}
    assert result['prepared_three_node_transport_envelope']['packet_cap'] == 612
    assert result['operation_caps_present_directly_in_launcher_manifest'] is False
    assert result['numerical_counts_are_success_expectations_not_independently_proved_predispatch_caps']


@pytest.mark.parametrize('fault', ['cell', 'horizon', 'budget', 'source_missing', 'source_conflict'])
def test_prepared_or_source_tampering_rejected(actual_inputs, fault):
    authority, prepared = deepcopy(actual_inputs)
    path = str(Path(c.physical.__file__).resolve())
    if fault == 'cell':
        authority['cell'] = 1
    elif fault == 'horizon':
        prepared['bodies'][2]['horizon_packets'] = 1
    elif fault == 'budget':
        prepared['budgets']['3']['packet_cap'] += 1
    elif fault == 'source_missing':
        prepared['bindings_sha256'].pop(path, None)
        authority['additional_sources'].pop(path, None)
    else:
        prepared['bindings_sha256'][path] = '0'*64
        authority['additional_sources'][path] = '1'*64
    with pytest.raises(ValueError):
        c.contract(authority, prepared)


@pytest.mark.parametrize('field', ['authority_sha256', 'child_code_sha256', 'cell', 'output'])
def test_manifest_tampering_rejected(field):
    m = manifest()
    m[field] = None
    with pytest.raises(ValueError, match='manifest binding'):
        c.manifest_binding(m)


def test_extra_manifest_cap_is_not_treated_as_original_contract():
    m = manifest()
    m['physical_caps'] = {'public_step': 999}
    with pytest.raises(ValueError, match='manifest binding'):
        c.manifest_binding(m)


def test_bounded_dedup_flags_unknown_and_preserves_control5(tmp_path):
    (tmp_path/c.candidate.AUTHORITY.name).write_text('fixture')
    (tmp_path/'agent_e3_gd_host_cuda_20261029_v1_cell02_MAIN01').mkdir()
    (tmp_path/'other_cell02_attempt.claim').symlink_to('missing')
    result = c.candidate_names(tmp_path)
    assert not result['enumerated_names_clear'] and not result['cross_entrypoint_unique_proven']
    assert [x['name'] for x in result['possible_conflicts']] == ['other_cell02_attempt.claim']
    assert any(x['classification'] == 'upstream_control5' for x in result['entries'])


def test_only_names_examined_excludes_cell03_without_stat(monkeypatch):
    class Entry:
        def __init__(self, name): self.name = name
        def __getattr__(self, name): raise AssertionError('entry inspected: '+name)
    class Scan:
        def __enter__(self):
            return iter([Entry('cell03_secret'), Entry('cell02_and_cell03'), Entry('cell02_attempt'),
                         Entry('arbitrary_unidentified')])
        def __exit__(self, *args): return False
    monkeypatch.setattr(c.os, 'scandir', lambda _: Scan())
    result = c.candidate_names('/unused')
    assert [x['name'] for x in result['entries']] == ['cell02_attempt']
    assert result['cross_entrypoint_unique_proven'] is False


def test_real_authority_manifest_review_with_fixture_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(c.final_review.preflight, 'MANIFEST_DIRECTORY', tmp_path)
    results = tmp_path/'results'
    results.mkdir()
    monkeypatch.setattr(c.candidate, 'RESULTS', results)
    path = tmp_path/'e3_cell02_manifest_fixture.json'
    data = c.candidate.canonical(manifest())+b'\n'
    path.write_bytes(data)
    result = c.review(str(path), c.candidate.sha(data))
    assert result['caps']['cell'] == 2
    assert result['bounded_duplicate_review']['enumerated_names_clear']
    assert not result['bounded_duplicate_review']['cross_entrypoint_unique_proven']
    assert not result['production_authorized'] and not result['final_environment_revalidated']


def test_cli_reports_conflict_and_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(c, 'review', lambda *args: {'bounded_duplicate_review': {
        'possible_conflicts': [{'name': 'cell02_unknown'}]}, 'production_ready': False})
    monkeypatch.setattr(c.sys, 'argv', ['audit', '--manifest', '/unused', '--manifest-sha256', '0'*64])
    assert c.main() == 1
    assert json.loads(capsys.readouterr().out)['production_ready'] is False


def test_dedup_snapshot_drift_is_not_silently_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(c.final_review.preflight, 'MANIFEST_DIRECTORY', tmp_path)
    results = tmp_path/'results'
    results.mkdir()
    monkeypatch.setattr(c.candidate, 'RESULTS', results)
    path = tmp_path/'e3_cell02_manifest_fixture.json'
    data = c.candidate.canonical(manifest())+b'\n'
    path.write_bytes(data)
    original = c.candidate_names
    calls = []
    def changing(path):
        calls.append(1)
        if len(calls) == 2:
            (results/'cell02_unknown.claim').touch()
        return original(path)
    monkeypatch.setattr(c, 'candidate_names', changing)
    with pytest.raises(ValueError, match='observations changed'):
        c.review(str(path), c.candidate.sha(data))


def test_real_cpu_draft_binds_real_prepared_contract(tmp_path, monkeypatch):
    # No candidate run: the original isolated draft only checks saved inputs.
    monkeypatch.setattr(c.final_review.preflight, 'MANIFEST_DIRECTORY', tmp_path)
    results = tmp_path/'results'
    results.mkdir()
    monkeypatch.setattr(c.candidate, 'RESULTS', results)
    data = c.final_review.preflight.manifest_bytes()
    path = tmp_path/'e3_cell02_manifest_cpu_fixture.json'
    path.write_bytes(data)
    result = c.review(str(path), c.candidate.sha(data))
    assert result['caps']['numerical_per_node_success_expectation']['predict'] == 204
    assert not result['production_authorized'] and not result['final_environment_revalidated']
