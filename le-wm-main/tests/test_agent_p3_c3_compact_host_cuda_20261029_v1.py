"""Only RAM callbacks/fakeCUDA/tempdisk. Never production build/consume."""
from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_c3_compact_host_cuda_20261029_v1 as m


@pytest.fixture(scope='module')
def fx():
    return runpy.run_path(str(m.ROOT/'tests/test_agent_p3_c3_common_cap_cuda_20261029_v1.py'))


def ram_prepare(fx):
    sources = {**m.PINS, **{str(p):m.codec.file_sha(p) for p in m.own_files()}}
    t = fx['task']()
    return dict(schema=m.NAME+'.ram_prepare',tasks=[t,deepcopy(t)],policy='RAM_ONLY_NOT_LOADABLE',
        expected_model_hash='1'*64,authority=dict(kind='pure_injected_fixture_NOT_original_authority'),
        sources=sources,roster=m.original.roster(),profile=m.original.profile(),precision=m.old.precision_contract(),
        rules=m.original.rules(),byte_proof=m.proof(),**m.FLAGS)


def before(p):
    path = str(m.own_files()[0])
    return dict(model=dict(original_state_dict=p['expected_model_hash'],all_buffers='2'*64),
        classes=[dict(name='',class_name='fixture.NotRealModel',path=path,sha256=p['sources'][path])],
        precision=m.old.precision_contract(),host=dict(device='cuda:0',allocation_sum=4.,visible_nvidia_nodes=['FAKE_RAM_ONLY']))


def test_integer_complete_HOST_budget_no_repeated_event_blanket():
    p = m.proof()
    assert p['live_bytes_upper'] == 3112701428
    assert p['total_bytes_upper'] == 3649572340
    assert p['live_remaining'] == 645394956
    assert p['additional_physical_files'] == dict(external_prepare=16*1024**2,host_report=8*1024**2,external_receipt=16*1024**2)
    assert p['output_support'] == dict(slots=12,request_identities=1728,axes=27)
    assert p['failure_reserve'] == 512*1024**2 and p['cap'] == 4*1024**3
    assert (p['wall_seconds'],p['slot_wall_seconds'],p['cuda_allocated_bytes'],p['cuda_reserved_bytes']) == (7200,900,3*1024**3,4*1024**3)
    assert p['fits'] and not p['wall_or_GPU_fit_predicted'] and not p['production_ready']
    assert m.PINS[str(Path(m.durable.__file__).resolve())] == '5142bbee6d829aa245ea4019ff55c4d050d28c5c6fa9472dfef874bd05c42041'


def test_RAM_seal_explicit_schema_original_fullshape_static_measure(fx):
    p = ram_prepare(fx); b = before(p)
    seal,store,cat,c = m.make_seal(p,b,fixture=True)
    assert c['source_bytes_checked'] is True and c['total_static_bytes'] <= 32*1024**2
    assert seal['value']['extra']['prepared'] == p
    assert seal['value']['extra']['before'] == b
    assert seal['value']['extra']['execution_backend'] == 'RAM_FIXTURE'
    assert all(cat.contains('context',t['root_context']) for t in p['tasks'])
    assert m.canonical(store.get(store.roots['static'])) == m.canonical(seal['value'])
    assert seal['value']['extra']['bank'] == m.bank_literals()
    with pytest.raises(ValueError,match='authority schema'): m.make_seal(p,b)


def test_large_input_is_stream_verified_not_mistaken_for_static_output_cap(tmp_path,fx):
    p = ram_prepare(fx); b = before(p); asset = tmp_path/'owned-large-input.bin'
    with asset.open('wb') as stream: stream.truncate(m.codec.STATIC+1)
    p['sources'][str(asset)] = m.codec.file_sha(asset)
    seal,_,_,c = m.make_seal(p,b,fixture=True)
    assert seal['value']['extra']['large_input_assets'] == [dict(path=str(asset),
        sha256=p['sources'][str(asset)],bytes=m.codec.STATIC+1)]
    assert seal['value']['extra']['all_source_bytes_checked'] is True
    assert c['total_static_bytes'] < m.codec.STATIC
    with asset.open('r+b') as stream: stream.write(b'X')
    with pytest.raises(ValueError,match='source bytes'): m.make_seal(p,b,fixture=True)


def test_original_order_static_root_restores_exact_typed_census(fx):
    p = ram_prepare(fx); b = before(p)
    seal,store,_,c = m.make_seal(p,b,fixture=True)
    root = store.get(store.roots['static']); raw = m.canonical(root)
    recovered = dict(value=root,fields=seal['fields'],canonical_sha256=m.codec.h(raw).hex(),canonical_bytes=len(raw))
    _,_,actual = m.routing.install_static(recovered)
    assert m.canonical(root) == m.canonical(seal['value'])
    assert actual == c


@pytest.mark.parametrize('field', ['model','classes','precision','host'])
def test_actual_before_metadata_reject_bad_ABI_or_unbound_class(fx,field):
    p = ram_prepare(fx); b = before(p)
    if field == 'model': b[field]['original_state_dict'] = '0'*64
    elif field == 'classes': b[field][0]['path'] = '/foreign/unbound.py'
    elif field == 'precision': b[field]['SDP'] = 'FLASH'
    else: b[field]['allocation_sum'] = True
    with pytest.raises(ValueError): m.make_seal(p,b,fixture=True)


def test_before_classes_duplicate_names_and_finite_budget(fx):
    p = ram_prepare(fx); b = before(p); b['classes'] *= 2
    with pytest.raises(ValueError,match='unique'): m.validate_before(p,b)
    b = before(p); b['classes'][0]['name'] = 'a'*(m.CLASS_CAP+1)
    with pytest.raises(ValueError,match='finite'): m.validate_before(p,b)


@pytest.mark.parametrize('elapsed,peak,valid', [(0,[0,0],True),(7200,[3*1024**3,4*1024**3],True),
    (7201,[0,0],False),(-1,[0,0],False),(1,[3*1024**3+1,0],False),(1,[0,4*1024**3+1],False),(1,[True,0],False)])
def test_fakeCUDA_guard_exact_caps_not_predicted_fit(elapsed,peak,valid):
    g = m.Guard(10,lambda:peak,clock=lambda:10+elapsed)
    seen = []; g.extra_check = lambda:seen.append('actual_precision_guard')
    if valid: assert g() == peak and seen == ['actual_precision_guard']
    else:
        with pytest.raises(ValueError): g()


def test_authorization_before_any_loader_OR_authority_discovery():
    with pytest.raises(ValueError,match='MAIN-only'):
        m.run('/never-read','not-a-sha','/never-create',authorization='NO')


def test_host_runtime_scalar_subclass_is_literal_without_changing_json():
    class FakeVersion(str): pass
    host = dict(device='cuda:0',torch_version=FakeVersion('2.8.0'),allocation_sum=4.,
                visible_nvidia_nodes=['/dev/fake'])
    literal = m.json_literal(host)
    assert type(literal['torch_version']) is str and m.canonical(literal) == m.canonical(host)
    assert type(literal['allocation_sum']) is float
    with pytest.raises(TypeError): m.json_literal(dict(foreign=object()))


def test_actual_online_native_failure_retains_all12_no_success_manifest(tmp_path,fx):
    p = ram_prepare(fx); b = before(p)
    class Failing(fx['Native']):
        def __init__(self,t,s): super().__init__(t,s,fail=True)
    g = m.Guard(m.time.monotonic(),lambda:[0,0]); out = tmp_path/'native_failure'
    with pytest.raises(ValueError,match='complete settled12'):
        m.drive(p,tmp_path/'ram_prepare.json','0'*64,out,b,lambda t,s:Failing(t,s),lambda:deepcopy(b),g,fixture=True)
    value = m.durable.read_json(out/'failure.json',m.durable.RESERVE)
    assert len(value['rows']) == 12 and value['rows'][0]['result'] is not None
    assert all(x['status'] == 'blocked_after_runtime_failure' for x in value['rows'][1:])
    assert not (out/'manifest.json').exists() and (out/'journal.bin').stat().st_size > 0
    assert value['full_transport_verified'] is False
    assert g.transport.used() < m.durable.TOTAL-m.durable.RESERVE
    assert g.transport.extra_bytes >= m.EXTERNAL_RESERVATION


def test_census_and_receipt_are_bounded_without_discarding_ledger():
    c = dict(schema='x',canonical_bytes=1,object_and_index_bytes=2,field_and_metadata_bytes=3,
        total_static_bytes=5,cap=32*1024**2,source_bytes_checked=True,
        source_entries=[dict(path='/large/a',sha256='0'*64,bytes=2**30)],production_ready=False)
    summary = m.census_summary(c)
    assert 'source_entries' not in summary and summary['source_entries_count'] == 1
    assert summary['source_entries_sha256'] == m.codec.semantic(c['source_entries'])
    events = [dict(index=i,payload='unabridged ledger') for i in range(1000)]
    original = dict(slot=0,resources=dict(caps={a:1 for a in m.core.AXES},events=events))
    reduced = m.receipt_arm(original)
    assert original['resources']['events'] == events
    assert reduced['resources']['events_count'] == 1000
    assert reduced['resources']['events_semantic_sha256'] == m.codec.semantic(events)
    assert 'events' not in reduced['resources'] and len(m.canonical(reduced)) < 16384


def test_help_and_proof_do_not_import_neural_or_run_build():
    code = ('import sys; from experiments.spectral_reachability import '+m.NAME+' as m; '
        'assert "torch" not in sys.modules; assert m.proof()["fits"]; '
        'assert "torch" not in sys.modules; m.main(["--help"])')
    r = subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,text=True,capture_output=True,timeout=30)
    assert r.returncode == 0 and 'consume' in r.stdout and 'prepare' in r.stdout


@pytest.fixture(scope='module')
def complete12(tmp_path_factory,fx):
    """NEW wrapper entire original12 RAM flow, NOT fixture×12 extrapolation.

    Deliberately expensive, never calls producer build/NN/production strict.
    Original fixedH1/H3 paths/caps unchanged; numerical Native callbacks are
    real RAM buffer witnesses. One run shared by saved/mutation tests.
    """
    path = tmp_path_factory.mktemp('compact_HOST_RAM_full12')
    p = ram_prepare(fx); b = before(p)
    result = m.drive(p,path/'ram_prepare.json','0'*64,path/'bundle',b,
        lambda t,s:fx['Nonlinear'](t,s),lambda:deepcopy(b),
        m.Guard(m.time.monotonic(),lambda:[0,0]),fixture=True)
    return p,b,path,result


def test_complete12_true_online_HOST_wrapper_tempdisk_saved_reconstruction(complete12):
    p,b,path,r = complete12
    with pytest.raises(ValueError,match='fixture'):
        m.Reader(r['output'],r['manifest_sha256'],r['completion_sha256'])
    reader = m.Reader(r['output'],r['manifest_sha256'],r['completion_sha256'],allow_fixture=True)
    try:
        saved = m.validate_saved(reader,p,path/'ram_prepare.json','0'*64,allow_fixture=True)
        assert saved['schema'] == m.NAME+'.ram_consume' and len(saved['arms']) == 12
        assert saved['total_request_denominator'] == 1728
        assert all(len(x['resources']['caps']) == 27 for x in saved['arms'])
        assert all(saved[k] is False for k in m.FLAGS)
        assert reader.report['before'] == reader.report['after'] == b
        assert reader.bytes+m.PREPARE_CAP+m.RECEIPT_CAP < m.durable.TOTAL-m.durable.RESERVE
    finally: reader.close()


def test_complete12_coherent_completion_clock_edit_requires_external_SHA(complete12):
    p,b,path,r = complete12; file = Path(r['output'])/'completion.json'; raw = file.read_bytes()
    value = m.durable.read_json(file,m.durable.COMPLETION_CAP)
    value['wall_seconds_through_manifest_fsync'] = 0.
    try:
        file.write_bytes(m.canonical(value))
        with pytest.raises(ValueError,match='external completion SHA'):
            m.Reader(r['output'],r['manifest_sha256'],r['completion_sha256'],allow_fixture=True)
    finally: file.write_bytes(raw)


@pytest.mark.parametrize('name',['report.json','static.json','journal.bin','cas/objects.pack','cas/objects.idx'])
def test_complete12_each_entire_file_mutation_rejected(complete12,name):
    p,b,path,r = complete12; file = Path(r['output'])/name
    with file.open('r+b') as f:
        old = f.read(1); f.seek(0); f.write(bytes([old[0]^1]))
    try:
        with pytest.raises(ValueError):
            m.Reader(r['output'],r['manifest_sha256'],r['completion_sha256'],allow_fixture=True)
    finally:
        with file.open('r+b') as f: f.write(old)
