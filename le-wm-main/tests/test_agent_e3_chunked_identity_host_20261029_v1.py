from copy import deepcopy
import pytest

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as h
from experiments.spectral_reachability import agent_e3_chunked_host_authority_20261029_v1 as a
from experiments.spectral_reachability import agent_e3_identity_authority_prepare_20261029_v1 as old_authority
from experiments.spectral_reachability import agent_e3_preflight_array_transport_20261029_v1 as transport
from test_agent_e3_identity_authority_prepare_20261029_v1 import refs, RESOLVER
from test_agent_e3_encoder_identity_20261029_v1 import prepared, FakeTorch, DEVICE, implementation, read_events


def test_strict_source_prepare_verify_and_no_overwrite(tmp_path):
    src = old_authority.prepare(refs(), RESOLVER, 0, str(tmp_path/'source.json'))
    out = str(tmp_path/'new.json')
    result = a.prepare(src['path'], src['sha256'], out)
    assert a.verify(src['path'], src['sha256'], out, result['sha256'])['verified']
    assert h.authority(out, result['sha256'])['cell'] == 0
    with pytest.raises((ValueError, FileExistsError)):
        a.prepare(src['path'], src['sha256'], out)
    with pytest.raises(ValueError):
        a.verify(src['path'], '0'*64, out, result['sha256'])


def test_loader_two_nodes_saved_and_tamper(tmp_path):
    p = prepared(tmp_path); s = transport.Store(tmp_path/'job')
    bound = h.chunks.load_bound(p, FakeTorch, DEVICE, s, 'a'*64)
    for boundary in (10, 15):
        bound._encode(FakeTorch, None, boundary, lambda: None, implementation)
    seal = s.publish('completed')
    assert h.launch.pub.consume(s.output, seal['report_sha256'], seal['completion_sha256'])['complete']
    events = read_events(s)
    result = h.chunks.verify(events, p, 'a'*64)
    assert [x['boundary'] for x in result['encodes']] == [10, 15]
    assert result['binding']['model_instance'] == hex(id(bound.model))
    bad = deepcopy(events)
    next(e for e in bad if e['resource']=='preflight_binding_chunk')['detail']['data'] = 'AAAA'
    with pytest.raises(ValueError): h.chunks.verify(bad, p, 'a'*64)
    partial = [e for e in events if e['resource'] != 'preflight_binding_seal']
    assert h.chunks.reconstruct(partial, allow_partial=True) is None
    with pytest.raises(ValueError): h.chunks.reconstruct(partial)


def test_no_execution_without_authorization(tmp_path):
    with pytest.raises(ValueError): h.run('/missing', '0'*64, str(tmp_path/'out'), 'no')
    assert not (tmp_path/'out').exists()


def test_new_inspector_partial_binding_and_old_marker_rejection(tmp_path, monkeypatch):
    s = transport.Store(tmp_path/'job')
    h.launch._record(s, 'chunked_identity_host_started', dict(schema=h.NAME))
    seal = s.publish('partial')
    base = dict(status='partial', encoding_ledger=dict(boundaries=[]))
    monkeypatch.setattr(h.host.prior, 'inspect', lambda *args: base)
    monkeypatch.setattr(h, 'authority', lambda *args: {})
    result = h.inspect(str(s.output), seal['report_sha256'], seal['completion_sha256'], '/authority', 'a'*64)
    assert result['identity'] is None and not result['binding_chunks_complete']
    assert not result['full_saved_strict'] and not result['scientific_admission']
    old = transport.Store(tmp_path/'old')
    h.launch._record(old, 'identity_host_started', dict(schema='old'))
    prior = old.publish('partial')
    with pytest.raises(ValueError):
        h.inspect(str(old.output), prior['report_sha256'], prior['completion_sha256'], '/authority', 'a'*64)


def test_new_inspector_joins_complete_bound_identity(tmp_path, monkeypatch):
    p = prepared(tmp_path); s = transport.Store(tmp_path/'job')
    h.launch._record(s, 'chunked_identity_host_started', dict(schema=h.NAME))
    bound = h.chunks.load_bound(p, FakeTorch, DEVICE, s, 'a'*64)
    for boundary in (10, 15): bound._encode(FakeTorch, None, boundary, lambda: None, implementation)
    seal = s.publish('completed')
    base = dict(status='completed', encoding_ledger=dict(boundaries=[dict(boundary=10),dict(boundary=15)]))
    monkeypatch.setattr(h.host.prior, 'inspect', lambda *args: base)
    monkeypatch.setattr(h, 'authority', lambda *args: dict(prepared=p, prepare_sha='a'*64))
    result = h.inspect(str(s.output), seal['report_sha256'], seal['completion_sha256'], '/authority', 'a'*64)
    assert result['binding_chunks_complete'] and len(result['identity']['encodes']) == 2
    base['encoding_ledger']['boundaries'] = [dict(boundary=10)]
    with pytest.raises(ValueError):
        h.inspect(str(s.output), seal['report_sha256'], seal['completion_sha256'], '/authority', 'a'*64)
