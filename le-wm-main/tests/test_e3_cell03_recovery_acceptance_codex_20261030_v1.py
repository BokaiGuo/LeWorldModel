import hashlib
import json

import pytest

from experiments.spectral_reachability import e3_cell03_recovery_acceptance_codex_20261030_v1 as m


def test_real_predecessor_read_only():
    result = m.audit_predecessor()
    assert result == {'predecessor_transport_verified': True, 'failure_index': 15,
                      'cuda_loader_recorded': True, 'physical_episode_recorded': False,
                      'scientific_admission': False}


def test_real_proposal_is_deterministic_and_not_authorized():
    result = m.proposal()
    assert result == m.proposal()
    body = result['proposal']
    assert not body['recovery_authorized'] and not body['production_enabled']
    assert body['proposed_claim'] != str(m.PREVIOUS_CLAIM)
    assert body['proposed_output'] != str(m.PREVIOUS)
    assert len(body['sources_sha256']['child_code_utf8']) == 64
    assert result['proposal_sha256'] == m.sha_bytes(
        json.dumps(body, sort_keys=True, separators=(',', ':')).encode())


def fixture_predecessor(tmp_path, monkeypatch, *, extra=False, failure='expected', status='partial'):
    output = tmp_path / 'output'
    store = m.transport.Store(output)
    resources = (['launcher_started', 'launcher_authority_verified',
                  'chunked_identity_host_started', 'launcher_gpu_preflight'] +
                 ['loader_event'] * 4 + ['loader_receipt'] +
                 ['preflight_binding_chunk'] * 5 + ['preflight_binding_seal', 'launcher_failure'])
    for name in resources:
        detail = {}
        if name == 'loader_receipt':
            detail = {'receipt': {'device': 'cuda:0'}}
        if name == 'launcher_failure':
            detail = {'type': 'ValueError', 'message': ('loaded module not presealed: ' +
                      (m.FAILURE_PATH if failure == 'expected' else failure))}
        store.event('physical', 'record', name, store.index, detail)
    if extra:
        store.event('physical', 'record', 'native_step', store.index, {})
    publication = store.publish(status)
    claim = tmp_path / 'claim'
    claim.write_bytes(b'fixture claim')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(m, 'PREVIOUS', output)
    monkeypatch.setattr(m, 'PREVIOUS_CLAIM', claim)
    monkeypatch.setattr(m, 'EXPECTED', {
        'claim': digest(claim), 'report': publication['report_sha256'],
        'completion': publication['completion_sha256'], 'journal': digest(output / 'events.jsonl')})
    return output, claim


@pytest.mark.parametrize('kwargs', [
    {'extra': True}, {'failure': 'another source'}, {'status': 'completed'}])
def test_reject_transport_valid_but_wrong_failure_semantics(tmp_path, monkeypatch, kwargs):
    fixture_predecessor(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(ValueError):
        m.audit_predecessor()


@pytest.mark.parametrize('file', ['claim', 'journal', 'report', 'completion'])
def test_reject_tampered_predecessor(tmp_path, monkeypatch, file):
    output, claim = fixture_predecessor(tmp_path, monkeypatch)
    assert m.audit_predecessor()['failure_index'] == 15
    path = claim if file == 'claim' else output / ('events.jsonl' if file == 'journal' else file + '.json')
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        m.audit_predecessor()


def test_proposal_detects_source_drift(monkeypatch):
    monkeypatch.setattr(m, 'audit_predecessor', lambda: None)
    monkeypatch.setattr(m, 'file_sha', lambda _: 'changed')
    with pytest.raises(ValueError, match='authority drift'):
        m.proposal()
