"""Development-only K2 HOST tests; no neural model or physics execution."""
import pytest
from copy import deepcopy
import runpy
from pathlib import Path
from experiments.spectral_reachability import agent_p3_c3_k2_host_cuda_20261029_v2 as m


def test_production_requires_explicit_main_authority_before_read():
    with pytest.raises(ValueError, match='MAIN-only'):
        m.run('/not-read', '0'*64, '/not-created', authorization='NO')


def test_six_slot_caps_and_frozen_budget():
    assert len(m.k2.roster()) == 6
    assert all(row['K'] == 2 for row in m.k2.roster())
    assert m.profile()['resource_caps'] == {'2':m.k2.caps()}
    assert m.rules()['roster'] == m.k2.roster()
    assert m.rules()['inherited_original_protocol_sha256'] == m.original.PROTOCOL_SHA
    p = m.proof()
    assert p['output_support'] == dict(slots=6, request_identities=864, axes=27)
    assert p['live_bytes_upper'] <= p['cap'] - p['failure_reserve']
    assert p['fits'] and p['transport']['complete_transport_schema_fit_proved']
    counts = m.ktransport.count_envelope()
    assert counts['request_denominator'] == 864 and not counts['serialized_byte_fit_proved']
    assert all(counts['new_totals'][a] <= counts['old_totals'][a] for a in counts['old_totals'])


def test_actual_failure_publish_preserves_six_rows(tmp_path):
    f = runpy.run_path(str(m.ROOT/'tests/test_agent_p3_c3_common_cap_cuda_20261029_v1.py'))
    h = runpy.run_path(str(m.ROOT/'tests/test_agent_p3_c3_compact_host_cuda_20261029_v1.py'))
    # Reuse only the RAM static fixture builder; no HOST model is loaded.
    p = h['ram_prepare'](f); before = h['before'](p)
    _, store, catalog, _ = h['m'].make_seal(p,before,fixture=True)
    online = m.ktransport.OnlineStore(tmp_path/'failure',store,catalog,p['tasks'])
    try:
        rows = [dict(**identity,status='blocked_before_model',result=None) for identity in m.k2.roster()]
        saved = online.transport.preserve_failure(rows,RuntimeError('fixture failure'))
        failure = m.durable.read_json(saved['path'],m.durable.RESERVE)
        assert failure['rows'] == rows and failure['full_transport_verified'] is False
        assert not (tmp_path/'failure'/'manifest.json').exists()
        assert m.codec.file_sha(saved['path']) == saved['sha256']
    finally:
        online.transport.close()


def test_six_slot_full_ram_disk_saved_reader(tmp_path):
    f = runpy.run_path(str(m.ROOT/'tests/test_agent_p3_c3_common_cap_cuda_20261029_v1.py'))
    sources = {**m.PINS, **{str(p):m.codec.file_sha(p) for p in m.own_files()}}
    sources.update({str(Path(x.__file__).resolve()):m.codec.file_sha(x.__file__)
                    for x in (m.k2, m.engine, m.ktransport, m.kbound)})
    t = f['task']()
    p = dict(schema=m.NAME+'.ram_prepare', tasks=[t, deepcopy(t)],
             policy='RAM_ONLY_NOT_LOADABLE', expected_model_hash='1'*64,
             authority=dict(kind='pure_injected_fixture_NOT_original_authority'), sources=sources,
             roster=m.k2.roster(), profile=m.profile(), precision=m.old.precision_contract(),
             rules=m.rules(), k2_protocol=m.k2.protocol(),
             selection_history='exploratory K2 after K1 queue refusal', byte_proof=m.proof(), **m.FLAGS)
    path = str(m.own_files()[0])
    before = dict(model=dict(original_state_dict='1'*64,all_buffers='2'*64),
        classes=[dict(name='',class_name='fixture.NotRealModel',path=path,sha256=sources[path])],
        precision=m.old.precision_contract(),host=dict(device='cuda:0',allocation_sum=4.,visible_nvidia_nodes=['FAKE_RAM_ONLY']))
    result = m.drive(p, tmp_path/'prepare.json', '0'*64, tmp_path/'bundle', before,
        lambda task,sink:f['Native'](task,sink), lambda:deepcopy(before),
        m.Guard(m.time.monotonic(),lambda:[0,0]), fixture=True)
    reader = m.Reader(result['output'], result['manifest_sha256'], result['completion_sha256'],allow_fixture=True)
    try:
        receipt = m.validate_saved(reader,p,tmp_path/'prepare.json','0'*64,allow_fixture=True)
        assert receipt['denominator'] == 6 and receipt['total_request_denominator'] == 864
        assert len(receipt['arms']) == 6
    finally:
        reader.close()
    # A saved consumer must reject mutation of each durable layer; restore the
    # fixture bytes after each trial, never repairing production artifacts.
    for name in ('report.json','static.json','journal.bin','cas/objects.pack','cas/objects.idx'):
        path = Path(result['output'])/name
        with path.open('r+b') as stream:
            first = stream.read(1); assert first
            stream.seek(0); stream.write(bytes([first[0] ^ 1]))
        try:
            with pytest.raises((ValueError,KeyError)):
                m.Reader(result['output'], result['manifest_sha256'],result['completion_sha256'],allow_fixture=True)
        finally:
            with path.open('r+b') as stream:
                stream.write(first)
