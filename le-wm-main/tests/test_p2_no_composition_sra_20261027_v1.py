"""No neural forwards: saved-contract validation and refusal integrity only."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import p2_no_composition_sra_20261027_v1 as m


@pytest.fixture
def prepared(tmp_path):
    if not (m.DEFAULT_REFERENCE/'report.json').is_file():
        pytest.skip('saved calibration inputs unavailable')
    output = tmp_path/m.TAG
    result = m.prepare(m.DEFAULT_REFERENCE, output)
    assert result['evidence_integrity_valid'], result
    return output


def test_refusal_has_no_fake_actions_or_matching_claim():
    result = m.refusal()
    assert result['status'] == 'rejected_missing_native_sra_contract'
    assert result['selected_actions'] is None and result['candidate_receipts'] == []
    for name in ('executable_baseline','P1_admission','P2_admission',
                 'prediction_budget_matched','fair_total_budget_comparison','physical_efficacy_verified'):
        assert result[name] is False
    assert result['new_neural_calls'] == result['new_physical_steps'] == 0
    assert set(result['gaps']) == set(m.GAPS)
    with pytest.raises(ValueError, match='no executable SRA'):
        m.selected_actions({'passed': True})


def test_resources_distinguish_unknown_limits_and_unexecuted_work():
    resources = m.resource_contract()
    assert all(v is None for v in resources['proposed_sra_limits'].values())
    assert {'jacobian_calls','scalar_output_vjp_directions','svd_calls','bounded_solve_calls',
        'compilation_predict_calls','selection_evaluations','flops','wall_time_limit_s'} <= set(m.RESOURCES)
    assert resources['reference_prediction_only']['candidates']*2 == 80
    assert resources['reference_prediction_only']['replay_predict_samples'] == 4
    assert resources['unknown_is_not_zero'] and resources['compilation_included_in_total']
    assert not resources['fair_total_budget_comparison']


def test_live_prepare_preserves_all_known_identities_and_sources(prepared):
    protocol, receipt, result = m.load_prepared(prepared)
    old = m.audit.read_json(m.DEFAULT_REFERENCE/'protocol.json')
    assert [r['root_index'] for r in protocol['context']['cases']] == list(range(4))
    old_by_id = {r['root_sha256']:r for r in old['prepared']}
    for case in protocol['context']['cases']:
        assert {k:case[k] for k in old_by_id[case['root_sha256']]} == old_by_id[case['root_sha256']]
        assert len(case['goal_vector_sha256']) == len(case['projection_sha256']) == 64
    assert protocol['context']['normalization'] == old['normalization']
    assert receipt['status'] == 'prepared_rejection_only'
    assert result['known_calibration_roots'] == 4 and result['independent_outcome_roots'] == 0
    assert not (prepared/'started.json').exists()
    manifest = m.audit.read_json(Path(receipt['source_seals'][0]['output_dir'])/'manifest.json')
    assert {e['source'] for e in manifest['files']} == {str(p) for p in m.source_files()}
    # Historical source seals include physical *code*, but no physical outcome
    # report is opened or used by this consumer.
    assert all('physical' not in str(Path(p).parent.name)
        for p in protocol['bindings'] if Path(p).name == 'report.json')
    with pytest.raises(FileExistsError):
        m.prepare(m.DEFAULT_REFERENCE, prepared)


@pytest.mark.parametrize('filename', ['protocol.json','prepared.json','report.json'])
def test_tampered_artifacts_fail_closed(prepared, filename):
    path = prepared/filename
    data = m.audit.read_json(path)
    data['P2_admission'] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        m.load_prepared(prepared)


def test_archive_tampering_fails(prepared):
    receipt = m.audit.read_json(prepared/'prepared.json')
    path = Path(receipt['source_seals'][0]['output_dir'])/'sources/000.source'
    path.write_text('altered only fresh test archive')
    with pytest.raises(ValueError, match='mismatch'):
        m.load_prepared(prepared)


def test_invalid_reference_writes_machine_readable_refusal(tmp_path):
    output = tmp_path/m.TAG
    result = m.prepare(tmp_path/'absent', output)
    assert result['status'] == 'invalid_evidence' and not result['evidence_integrity_valid']
    assert m.audit.read_json(output/'invalid.json') == result
    with pytest.raises(ValueError, match='invalid/incomplete'):
        m.load_prepared(output)


def test_contract_mutation_refused_before_source_sealing(tmp_path, monkeypatch):
    original = m.audit.EvidenceReader.load
    def changed(reader, path, expected=None):
        data = original(reader, path, expected)
        if Path(path) == m.DEFAULT_REFERENCE/'protocol.json':
            data = deepcopy(data)
            data['schedule']['candidates'] = 80
        return data
    monkeypatch.setattr(m.audit.EvidenceReader, 'load', changed)
    result = m.prepare(m.DEFAULT_REFERENCE, tmp_path/m.TAG)
    assert result['status'] == 'invalid_evidence'
    assert 'protocol digest' in result['integrity_errors'][0]


@pytest.mark.parametrize('phase', ['verify','cuda'])
def test_cli_refuses_without_allocating_or_loading_neural_stack(prepared, phase, monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        assert not name.startswith(('torch','stable_worldmodel','stable_pretraining')), name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    assert m.main([phase, '--output', str(prepared)]) == 2
    assert not (prepared/'started.json').exists()


def test_clean_process_prepare_imports_no_torch(tmp_path):
    output = tmp_path/m.TAG
    script = """
import builtins, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.startswith(('torch', 'stable_worldmodel', 'stable_pretraining')):
        raise AssertionError('neural stack import: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments.spectral_reachability import p2_no_composition_sra_20261027_v1 as m
raise SystemExit(m.main(['prepare', '--output', sys.argv[1]]))
"""
    completed = subprocess.run([sys.executable, '-B', '-c', script, str(output)],
        cwd=m.PROJECT, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 2, completed.stdout+completed.stderr
    assert m.audit.read_json(output/'report.json')['new_neural_calls'] == 0


@pytest.mark.parametrize('text', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'])
def test_strict_json_rejects_ambiguous_or_nonfinite_input(tmp_path, text):
    path = tmp_path/'bad.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        m.audit.read_json(path)
