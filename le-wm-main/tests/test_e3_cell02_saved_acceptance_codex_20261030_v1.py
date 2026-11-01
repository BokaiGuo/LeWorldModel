from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import e3_cell02_saved_acceptance_codex_20261030_v1 as a
from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as cap
from test_agent_e3_continuous_saved_consumer_20261029_v1 import exercise_fake_episode

REAL_FEEDBACK_CONSUME = a.feedback.consume


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'DIRECTORY', tmp_path)
    monkeypatch.setattr(a.candidate, 'CLAIM', tmp_path/'claim')
    monkeypatch.setattr(a.candidate, 'OUTPUT', tmp_path/'job')
    a.candidate.OUTPUT.mkdir()
    report = a.candidate.OUTPUT/'report.json'
    completion = a.candidate.OUTPUT/'completion.json'
    (a.candidate.OUTPUT/'events.jsonl').write_bytes(b'journal')
    report.write_bytes(a.base.raw({'journal_sha256': a.base.fsha(b'journal')}))
    completion.write_bytes(b'completion')
    m = {'schema': a.candidate.SCHEMA, 'cell': 2, 'authority': str(a.candidate.AUTHORITY),
         'authority_sha256': a.candidate.AUTHORITY_SHA, 'output': str(a.candidate.OUTPUT),
         'claim': str(a.candidate.CLAIM), 'child_code_sha256': a.candidate.sha(a.candidate.CHILD.encode()),
         'audit_code_sha256': a.candidate.sha(a.candidate.AUDIT.encode()),
         'precheck_code_sha256': a.candidate.sha(a.candidate.PRECHECK.encode()),
         'production_authorization': 'external_decision_required', 'scientific_admission': False,
         'parent_sources': {str(Path(a.candidate.__file__).resolve()): a.candidate.file_sha(Path(a.candidate.__file__).resolve())}}
    manifest = tmp_path/'e3_cell02_manifest_fixture.json'
    manifest.write_bytes(a.candidate.canonical(m)+b'\n')
    msha = a.candidate.file_sha(manifest)
    claim = {'schema': a.candidate.SCHEMA+'.claim', 'manifest_sha256': msha,
             'authority_sha256': a.candidate.AUTHORITY_SHA, 'output': str(a.candidate.OUTPUT),
             'pid': 123, 'meaning': 'attempt claimed, outcome unknown'}
    a.candidate.CLAIM.write_bytes(a.candidate.canonical(claim))
    pins = {'consumer': a.candidate.file_sha(Path(a.__file__).resolve()),
            'native': a.candidate.file_sha(Path(a.native.__file__).resolve()),
            'feedback': a.candidate.file_sha(Path(a.feedback.__file__).resolve())}
    inspection = {'base': {'cell': 2, 'status': 'completed20'},
                  'binding_chunks_complete': True, 'identity': {'encodes': [{'boundary': 10}, {'boundary': 15}]}}
    monkeypatch.setattr(a.host, 'inspect', lambda *args: deepcopy(inspection))
    monkeypatch.setattr(a.native, 'build', lambda *args: {'transport_complete': True,
                                                          'native': {'terminal_status': inspection['base']['status']}})
    monkeypatch.setattr(a.host, 'authority', lambda *args: {'cell': 2, 'args': (), 'resolver': {'path': '', 'sha256': ''}})
    monkeypatch.setattr(a.host.launch.episode.old.authority, 'load', lambda *args: {'body': {}, 'first5': []})
    monkeypatch.setattr(a.host.launch.episode.old.setup_authority, 'resolve', lambda *args: {'prefix5': []})
    monkeypatch.setattr(a.feedback, 'consume', lambda *args: {'completed20_feedback_io_joined': True})
    args = (a.candidate.file_sha(report), a.candidate.file_sha(completion), str(manifest), msha, pins, a.ACK)
    return args, inspection


def test_build_and_exclusive_receipt(fixture, tmp_path):
    args, _ = fixture
    result = a.build(*args)
    assert result['complete_feedback_chain_checked'] and not result['scientific_admission']
    receipt = tmp_path/'e3_cell02_acceptance_fixture.json'
    sha = a.save(receipt, result)
    assert a.read_pinned(receipt, sha) == a.base.raw(result)
    with pytest.raises(FileExistsError):
        a.save(receipt, result)


@pytest.mark.parametrize('fault', ['claim', 'manifest', 'report', 'completion', 'checker', 'identity', 'partial', 'terminal'])
def test_acceptance_rejects_tampering(fixture, fault):
    args, inspection = fixture
    args = list(args)
    if fault == 'claim':
        a.candidate.CLAIM.write_bytes(b'{}')
    elif fault == 'manifest':
        Path(args[2]).write_bytes(b'{}')
    elif fault == 'report':
        (a.candidate.OUTPUT/'report.json').write_bytes(b'changed')
    elif fault == 'completion':
        (a.candidate.OUTPUT/'completion.json').write_bytes(b'changed')
    elif fault == 'checker':
        args[4] = dict(args[4], feedback='0'*64)
    elif fault == 'identity':
        inspection['identity']['encodes'].pop()
    elif fault == 'partial':
        inspection['base']['status'] = 'partial'
    else:
        inspection['base']['cell'] = 3
    with pytest.raises((ValueError, KeyError)):
        a.build(*args)


def test_early_stop_is_not_full_feedback(fixture):
    args, inspection = fixture
    inspection['base']['status'] = 'native_initial_success_control5'
    result = a.build(*args)
    assert result['full_feedback_join'] is None and not result['complete_feedback_chain_checked']
    assert result['terminal_status'] == 'native_initial_success_control5'


def test_build_calls_real_feedback_join_on_complete_fixture(fixture, tmp_path, monkeypatch):
    import test_agent_e3_feedback_setup_20261029_v2 as cases
    original_case = cases.case
    def cell02_case(*args, **kwargs):
        raw, kw, unused, row = original_case(*args, **kwargs)
        row['episode_id'] = 'cell02'
        return raw, kw, unused, row
    monkeypatch.setattr(cases, 'case', cell02_case)
    args, _ = fixture
    args = list(args)
    store = cap.pub.Store(tmp_path/'job-real')
    row, kw, body, first, _ = exercise_fake_episode(store, None)
    seal = store.publish('completed')
    monkeypatch.setattr(a.candidate, 'OUTPUT', store.output)
    manifest = Path(args[2])
    value = json.loads(manifest.read_bytes())
    value['output'] = str(store.output)
    manifest.write_bytes(a.candidate.canonical(value)+b'\n')
    args[3] = a.candidate.file_sha(manifest)
    claim = json.loads(a.candidate.CLAIM.read_bytes())
    claim.update(output=str(store.output), manifest_sha256=args[3])
    a.candidate.CLAIM.write_bytes(a.candidate.canonical(claim))
    args[0], args[1] = seal['report_sha256'], seal['completion_sha256']
    monkeypatch.setattr(a.host.launch.episode.old.authority, 'load',
                        lambda *unused: {'body': body, 'first5': first})
    monkeypatch.setattr(a.host.launch.episode.old.setup_authority, 'resolve',
                        lambda *unused: {'prefix5': kw['prefix5']})
    monkeypatch.setattr(a.feedback, 'consume', REAL_FEEDBACK_CONSUME)
    assert row['episode_id'] == 'cell02'
    assert a.build(*args)['complete_feedback_chain_checked'] is True


def reseal(store, seal, mutate):
    p = store.output
    events = [a.base.decode(line) for line in (p/'events.jsonl').read_bytes().splitlines()]
    mutate(events)
    previous = None
    for e in events:
        e['previous'] = previous
        e['sha256'] = a.base.digest({k: v for k, v in e.items() if k != 'sha256'})
        previous = e['sha256']
    journal = b''.join(a.base.raw(e)+b'\n' for e in events)
    # Preserve transport byte accounting after mutation with same-sized values.
    report = a.base.decode((p/'report.json').read_bytes())
    assert len(journal) == (p/'events.jsonl').stat().st_size
    report['tail'] = previous
    report['journal_sha256'] = a.base.fsha(journal)
    rb = a.base.raw(report)
    completion = a.base.decode((p/'completion.json').read_bytes())
    completion['report_sha256'] = a.base.fsha(rb)
    cb = a.base.raw(completion)
    (p/'events.jsonl').write_bytes(journal)
    (p/'report.json').write_bytes(rb)
    (p/'completion.json').write_bytes(cb)
    return a.base.fsha(rb), a.base.fsha(cb)


@pytest.mark.parametrize('fault', ['none', 'dispatch', 'rgb', 'latent', 'node'])
def test_real_feedback_consumer_with_resealed_fixture_tampering(tmp_path, fault):
    store = cap.pub.Store(tmp_path/'job')
    row, kw, body, first, _ = exercise_fake_episode(store, None)
    seal = store.publish('completed')
    def mutate(events):
        def event(resource, boundary=None):
            return next(e for e in events if e['resource'] == resource and
                        (boundary is None or e['detail'].get('boundary') == boundary))
        if fault == 'dispatch':
            d = next(e for e in events if e['resource'] == 'native_authorized_dispatch'
                     and e['detail']['step'] == 6)['detail']
            d['raw_bits'] = 'f' + d['raw_bits'][1:]
        elif fault == 'rgb':
            d = event('native_e3_encode_intent', 10)['detail']['observation']
            d['pixel_sha256'] = 'f' * 64
        elif fault == 'latent':
            d = event('native_e3_encode_return', 10)['detail']
            d['latent'][0] = 11.0
        elif fault == 'node':
            d = event('native_e3_node_return', 10)['detail']
            d['raw_bits'] = 'f' + d['raw_bits'][1:]
    if fault != 'none':
        report_sha, completion_sha = reseal(store, seal, mutate)
        assert cap.pub.consume(store.output, report_sha, completion_sha)['complete']
    else:
        report_sha, completion_sha = seal['report_sha256'], seal['completion_sha256']
    args = (str(store.output), report_sha, completion_sha, body, kw['prefix5'], first, row['episode_id'])
    if fault == 'none':
        assert a.feedback.consume(*args)['completed20_feedback_io_joined'] is True
    else:
        with pytest.raises((ValueError, KeyError)):
            a.feedback.consume(*args)
