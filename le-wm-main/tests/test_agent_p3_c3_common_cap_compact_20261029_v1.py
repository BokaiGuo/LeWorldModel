"""RAM/stdlib/real native-callback fixture ABI only; no production execution."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_c3_common_cap_compact_20261029_v1 as m


def blob(shape, dtype='<f4', value=-0.):
    n = 1
    for d in shape: n *= d
    raw = struct.pack('<f' if dtype == '<f4' else '<d',value)*n
    return dict(shape=list(shape),dtype=dtype,sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode())


def wrapped(event, slot=11, call=4883):
    return dict(kind='arm_event',slot=slot,payload=dict(kind='io',call=call,payload=event))


def events(h=3):
    scope = dict(mode='adaptive',K=17,phase='program',parent_context_sha256='a'*64)
    fixed = {
        'origin': dict(latent=blob((1,3,192)),history=blob((1,2,10)),raw_flat=blob((10*h,)),mean=blob((2,)),std=blob((2,),value=1.)),
        'packet': dict(raw=blob((5,2)),token=blob((10,)),outgoing=blob((1,3,10))),
        'program_packet': dict(latent=blob((1,3,192)),history=blob((1,2,10)),raw=blob((5,2)),mean=blob((2,)),
            std=blob((2,),value=1.),token=blob((10,)),outgoing=blob((1,3,10))),
        'boundary': dict(latent=blob((1,3,192)),history=blob((1,2,10))),
        'program_boundary': dict(latent=blob((1,3,192)),history=blob((1,2,10))),
        'nominal': dict(value=blob((192,))), 'AD_terminal': dict(value=blob((192,))),
        'jacobian': dict(value=blob((192,10*h))), 'raw_backward': dict(value=blob((10*h,)))}
    out = []
    for kind,arrays in fixed.items():
        phase = 'whole' if kind.startswith('program_') else 'AD'
        out.append(wrapped(dict(kind=kind,scope=scope,phase=phase,step=8,operation_index=2**63,arrays=arrays)))
    ins = [{'action_encoder:0':blob((1,3,10))},
        {'predict:latent':blob((1,3,192)),'predict:act_emb':blob((1,3,192))},
        {'predictor:0':blob((1,3,192)),'predictor:1':blob((1,3,192))},
        {'pred_proj:0':blob((3,192))},
        {'grad:output':blob((192,)),'grad:raw':blob((10*h,)),'grad:basis':blob((192,))}]
    for arrays in ins:
        out.append(wrapped(dict(kind='io_input',scope=scope,phase='AD',step=None,operation_index=2**64-1,arrays=arrays)))
    for op in ('action_encoder','predict','predictor','pred_proj','grad'):
        shape = (10*h,) if op == 'grad' else (3,192) if op == 'pred_proj' else (1,3,192)
        out.append(wrapped(dict(kind='io_return',scope=scope,phase='AD',step=None,operation_index=0,arrays={op+':output':blob(shape)})))
    out.append(wrapped(dict(kind='direction',scope=scope,direction=191)))
    for kind,status in zip(m.RESOURCES,m.STATUSES):
        out.append(wrapped(dict(kind=kind,entry=dict(index=2**64-1,scope=scope,resource='vjp',shape=[1,3,192],
            directions=2**64-1,delta=dict.fromkeys(m.AXES,2**64-1),status=status,started_ns=0,
            ended_ns=None if status in ('attempted','blocked_before_dispatch') else 2**64-1,
            error='complete failure text' if status == 'failed' else None))))
    out += [wrapped(dict(kind='native_call_attempt',method='signal',task={'RAM':True},request={'RAM':False})),
            wrapped(dict(kind='native_call_return',result=[-0.,1.])),
            wrapped(dict(kind='native_call_failure',error=dict(type='RuntimeError',message='unmodified')))]
    for kind,names in m.AUX.items(): out.append(wrapped(dict(kind=kind,**{k:dict(RAM=True) for k in names})))
    return out


def archive(values):
    c = m.Codec(m.keys_of(values),static=dict(label='RAM_not_production'))
    for e in values: c.append_event(e)
    return c


@pytest.mark.parametrize('h', [1,3])
def test_every_real_event_variant_fullshape_rows_and_byteproof(h):
    es = events(h); c = archive(es)
    assert [m.canonical(c.event(i)) for i in range(len(es))] == [m.canonical(x) for x in es]
    assert set(c.kind_counts) == set(m.KINDS)
    stats = c.verify_events(); c.verify_closure()
    for kind,length in stats['row_max'].items():
        cap = 512 if kind in m.ARRAYS else 192 if kind in m.RESOURCES else 64 if kind == 'direction' else 128 if kind in m.NATIVE else 256
        assert length <= cap
    assert stats['tensor_header_and_index_max'] <= 128
    p = m.proof(); assert p['live_bytes_upper'] == 2930867648 < m.LIVE
    assert p['request_denominator'] == 1728 and p['denominator'] == 12 and not p['production_ready']


@pytest.mark.parametrize('v', [None,True,False,0,1,-1,-2**63,2**64-1,0.,-0.,1.,-1.,
    1.7976931348623157e308,5e-324,'汉字\nsource',[],[1,True,1.,-0.],[[0.,-0.],[1.,2.]],{'v':[1,True,-0.]}])
def test_literal_type_and_float_bits_exact(v):
    c = m.Codec(m.keys_of(v)); key = c.put(v); c.roots['literal'] = key
    got = c.get(key)
    assert type(got) is type(v) and m.canonical(got) == m.canonical(v)
    if type(v) is float: assert struct.pack('<d',got) == struct.pack('<d',v)
    c.verify_closure()


def test_int_float_bool_not_collapsed_and_dense_lists_do_not_coerce():
    c = m.Codec([])
    refs = [c.put(x) for x in [True,1,1.,False,0,0.,-0.]]
    assert len(set(refs)) == 7
    a = c.get(c.put([1,True,1.,-0.])); assert list(map(type,a)) == [int,bool,float,float]
    assert struct.pack('<d',a[-1]) == struct.pack('<d',-0.)


@pytest.mark.parametrize('dtype', ['<f4','<f8'])
@pytest.mark.parametrize('shape', [(192,30),(1,3,192),(3,192),(45,2),(2,)])
def test_raw_tensor_exact_original_dtype_shape_bytes_once(dtype,shape):
    b = blob(shape,dtype); c = m.Codec(m.keys_of(b)); one = c.put(b)
    assert c.put(deepcopy(b)) == one and len(c.objects) == 1
    assert c.get(one) == b
    assert c.objects[one][:1] == b'B' and len(c.objects[one])+48-len(base64.b64decode(b['data'])) <= 128
    c.roots['tensor'] = one; c.verify_closure()


@pytest.mark.parametrize('bad', ['kind','extra','missing','wrongset','shape','dtype','sha','base64','boolindex','negativeindex','oversizeindex','status'])
def test_variant_layout_mutations_refused(bad):
    e = events()[0]; e = deepcopy(e); core = e['payload']['payload']
    if bad == 'kind': core['kind'] = 'unknown'
    elif bad == 'extra': core['extra'] = 1
    elif bad == 'missing': del core['scope']
    elif bad == 'wrongset': del core['arrays']['mean']
    elif bad == 'shape': core['arrays']['raw_flat'] = blob((20,))
    elif bad == 'dtype': core['arrays']['latent'] = blob((1,3,192),'<f8')
    elif bad == 'sha': core['arrays']['latent']['sha256'] = 'f'*64
    elif bad == 'base64': core['arrays']['latent']['data'] += '\n'
    elif bad == 'boolindex': core['operation_index'] = True
    elif bad == 'negativeindex': core['step'] = -1
    elif bad == 'oversizeindex': core['operation_index'] = 2**64
    else:
        e = next(x for x in events() if x['payload']['payload']['kind'] == 'resource_return')
        e['payload']['payload']['entry']['status'] = 'failed'
    with pytest.raises((ValueError,KeyError)): archive([e])


@pytest.mark.parametrize('v', [float('nan'),float('inf'),2**64,-2**63-1,(1,2),{'a':object()}])
def test_unknown_nonfinite_foreign_or_unbounded_types_failclosed(v):
    with pytest.raises(ValueError): m.Codec({'a'}).put(v)


def test_unknown_fields_static_cap_and_cycle_refuse():
    c = m.Codec({'known'})
    with pytest.raises(ValueError): c.put({'unknown':1})
    v = []; v.append(v)
    with pytest.raises(ValueError,match='acyclic'): c.put(v)
    with pytest.raises(ValueError,match='static'): m.Codec([],static='x'*m.STATIC)


def test_reference_spine_views_actual_proof_no_duplicate_event_copies():
    es = [wrapped(dict(kind='native_call_attempt',method='replay',task={'RAM':True},request={'RAM':True}))]
    es += [wrapped(dict(kind='direction',scope={'RAM':True},direction=i)) for i in range(4)]
    es += [wrapped(dict(kind='resource_return',entry=dict(index=0,scope={'RAM':True},resource='metric',shape=None,
        directions=None,delta=dict.fromkeys(m.AXES,0),status='returned',started_ns=0,ended_ns=1,error=None)))]
    es += [wrapped(dict(kind='native_call_return',result=[]))]
    c = archive(es); indices = list(range(len(es)))
    c.roots['slot'] = c.add_view('slot',indices)
    c.roots['native_calls'] = c.add_view('native',indices)
    c.roots['ledger'] = c.add_view('ledger',[len(es)-2])
    raw = [x['payload']['payload'] for x in es]
    entry = c.virtual_capsule(11,4883,raw)
    assert c.virtual_capsule(11,4883,raw) == entry
    c.roots['bank_evidence_alias'] = bytes.fromhex(entry['object_sha256'])
    reader = m.VirtualCapsuleReader(c); pointer = reader.callback(11,4883,raw)
    assert pointer['sha256'] == m.semantic(raw) and '__virtual__' in pointer['path']
    assert not Path(pointer['path']).exists() and entry['materialized'] is False
    stats = c.verify_events(); assert stats['actual_max_spine_memberships'] == 5 <= 8
    assert stats['view_budget_fits'] and c.read_virtual(entry) == raw
    with pytest.raises(ValueError,match='delimiter'): c.add_view('native',indices[:-1])
    with pytest.raises(ValueError): c.add_view('arbitrary_alias',indices)
    altered = dict(entry,materialized=True)
    with pytest.raises(ValueError,match='NONmaterial'): c.read_virtual(altered)
    altered = deepcopy(raw); altered[1]['direction'] = 123
    with pytest.raises(ValueError): reader.callback(11,4883,altered)
    c.verify_closure()


def test_spine_bound_for_all_sizes_not_small_fixture_extrapolation():
    p = m.spine_proof()
    assert p['maximum_memberships_including_journal'] == 5 <= 8
    assert sum(p['per_event_bounds'].values()) == 194 <= 256
    for family,minimum in p['family_min_members'].items():
        for n in (minimum,2,3,192,1443564):
            if n >= minimum:
                assert 6+48+8*n <= p['per_event_bounds'][family]*n


@pytest.fixture(scope='module',params=[1,3])
def actual_native_RAM(request):
    # Executes only existing NumPy callbacks. No model/weights/production source
    # discovery, no producer build/prepare or fullstrict on actual artifacts.
    f = runpy.run_path(str(m.ROOT/'tests/test_agent_p3_r_native_orchestrator_20261029_v1.py'))
    tree = f['m']; native = f['fixtures']['NumpyNative']; task = f['fixtures']['original']['task']()
    class Sink:
        def __init__(self): self.events = []
        def event(self,e): self.events.append(wrapped(deepcopy(e),slot=0,call=0) if e.get('kind') != 'io'
            else dict(kind='arm_event',slot=0,payload=deepcopy(e)))
    sink = Sink(); ledger = tree.core.Ledger(dict.fromkeys(tree.core.AXES,10**9)); ledger.scope = dict(phase='RAM_signal')
    session = tree.Session(task,lambda t,s:native(t,s),sink,ledger,'RAM_FIXTURE',lambda:None,lambda *x:None)
    ledger.sink = session.event
    ledger.call('signal_compile',lambda:session.signal(task,request.param))
    return tree,session,sink,ledger,request.param


def test_real_native_H1_H3_all192VJP_exact_expand_and_audit(actual_native_RAM,tmp_path):
    tree,session,sink,ledger,horizon = actual_native_RAM
    es = sink.events; receipt = ledger.receipt()
    c = m.Codec(m.keys_of([es,receipt]),static=dict(label='real_RAM_callback_not_neural'))
    for e in es: c.append_event(e)
    old = session.records[0]['transcript']; entry = c.virtual_capsule(0,0,old)
    expanded = c.read_virtual(entry)
    assert tree.old.audit_native(expanded) is True
    assert len([e for e in expanded if e['kind'] == 'direction']) == 192
    assert expanded[-1]['result']['horizon'] == horizon
    assert m.canonical(expanded) == m.canonical(old)
    indices = [i for i,e in enumerate(es) if c.peel(e)[1]['kind'] in ('resource_return','resource_failure','resource_blocked')]
    ledger_view = c.add_view('ledger',indices)
    c.roots['full_receipt'] = c.put_with_refs(receipt,dict(events=ledger_view))
    assert m.canonical(c.get(c.roots['full_receipt'])) == m.canonical(receipt)
    tree.core.audit_ledger(c.get(c.roots['full_receipt']))
    stats = c.verify_events(); assert stats['actual_max_spine_memberships'] <= 5 and stats['view_budget_fits']
    info = m.write_fixture(c,tmp_path/'native')
    loaded = m.read_fixture(tmp_path/'native',info['fixture_sha256'])
    reader = m.VirtualCapsuleReader(loaded)
    assert reader.callback(0,0,old)['sha256'] == m.semantic(old)
    assert reader.provenance(0,0)['physical_storage_binding']['artifacts']['objects.pack']
    assert tree.old.audit_native(loaded.read_virtual(entry)) is True


@pytest.mark.parametrize('damage',['none','pack','index','journal','extra','missing','manifest','unused','dangling','trailing'])
def test_saved_fixture_full_census_hashes_and_internal_mutations(tmp_path,damage):
    c = archive(events()); folder = tmp_path/'codec'
    info = m.write_fixture(c,folder)
    if damage == 'none':
        got = m.read_fixture(folder,info['fixture_sha256'])
        assert [m.canonical(got.event(i)) for i in range(len(c.events))] == [m.canonical(c.event(i)) for i in range(len(c.events))]
        return
    if damage == 'extra': (folder/'extra').write_bytes(b'x')
    elif damage == 'missing': (folder/'objects.idx').unlink()
    elif damage == 'manifest': (folder/'fixture.json').write_bytes(b'{}')
    elif damage in ('pack','index','journal'):
        name = dict(pack='objects.pack',index='objects.idx',journal='journal.bin')[damage]
        raw = bytearray((folder/name).read_bytes()); raw[-1] ^= 1; (folder/name).write_bytes(raw)
    else:
        meta = json.loads((folder/'fixture.json').read_bytes())
        if damage == 'unused':
            raw = b'Shidden'; key = m.h(m.DOMAIN+raw); offset = (folder/'objects.pack').stat().st_size
            with (folder/'objects.pack').open('ab') as f: f.write(raw)
            with (folder/'objects.idx').open('ab') as f: f.write(m.INDEX.pack(key,offset,len(raw)))
        elif damage == 'trailing':
            with (folder/'objects.pack').open('ab') as f: f.write(b'extra')
        else: meta['roots']['missing'] = 'f'*64
        meta['artifacts'] = {n:m.h((folder/n).read_bytes()).hex() for n in meta['artifacts']}
        raw = m.canonical(meta); (folder/'fixture.json').write_bytes(raw); info['fixture_sha256'] = m.h(raw).hex()
    with pytest.raises((ValueError,KeyError)): m.read_fixture(folder,info['fixture_sha256'])


def test_cold_stdlib_only_CLI_and_frozen_protocol():
    assert hashlib.sha256(m.PROTOCOL.read_bytes()).hexdigest() == m.PROTOCOL_SHA
    code = f'import sys; import {m.__name__}; assert not ({{"torch","numpy","pickle","gymnasium"}} & set(sys.modules))'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,check=True)
    result = subprocess.run([sys.executable,'-m',m.__name__,'proof'],cwd=m.ROOT,check=True,text=True,capture_output=True)
    assert json.loads(result.stdout)['production_ready'] is False
    result = subprocess.run([sys.executable,'-m',m.__name__,'--help'],cwd=m.ROOT,check=True,text=True,capture_output=True)
    assert 'roundtrip-demo' in result.stdout and '{prepare' not in result.stdout


def test_fullshape_element_and_record_census_no_dedup_savings_assumption():
    cores = [x['payload']['payload'] for x in events(3)]
    arrays = [e for e in cores if e['kind'] in m.ARRAYS]
    def measure(es):
        tensors = [b for e in es for b in e['arrays'].values()]
        return len(tensors),sum(len(base64.b64decode(b['data']))//4 for b in tensors)
    forward = [e for e in arrays if e['kind'] in ('program_packet','program_boundary') or
        e['kind'] in ('io_input','io_return') and not any(k.startswith('grad:') for k in e['arrays'])]
    vjp = [e for e in arrays if e['kind'] == 'raw_backward' or
        e['kind'] in ('io_input','io_return') and any(k.startswith('grad:') for k in e['arrays'])]
    origin = next(e for e in arrays if e['kind'] == 'origin')
    extra = [origin,origin]+[e for e in arrays if e['kind'] in ('nominal','AD_terminal','jacobian')]
    assert measure(forward) == (19,6460) and measure(vjp) == (5,474) and measure(extra) == (13,7404)
    assert 6460 <= 8192 and 7404 <= 8192
    c = archive(events()); s = c.stats()
    assert sum(s['physical_object_and_index_bytes_by_tag'].values()) == s['stored_object_and_index_bytes']


def test_twelve_slots_1728_literal_identities_roundtrip_no_partial_plan_release(tmp_path):
    rows = []
    for root in (0,1):
        for k in (1,17):
            for mode in ('fixedH1','staticH3_progress','adaptive'):
                rows.append(dict(slot=len(rows),root=root,K=k,mode=mode,full_C3_demonstrated=False,
                    model_plan=None,jobs=[dict(slot=i,status='not_requested') for i in range(144)]))
    c = m.Codec(m.keys_of(rows)); c.roots['rows'] = c.put(rows)
    info = m.write_fixture(c,tmp_path/'roster')
    loaded = m.read_fixture(tmp_path/'roster',info['fixture_sha256'])
    got = loaded.get(loaded.roots['rows'])
    assert m.canonical(got) == m.canonical(rows) and len(got) == 12
    assert sum(len(row['jobs']) for row in got) == 1728
    assert all(row['model_plan'] is None and row['full_C3_demonstrated'] is False for row in got)


@pytest.mark.parametrize('damage',['dtype','shape','trailing','ref','rawhash','static','delta27','capsuleidentity'])
def test_internal_raw_and_authority_mutations_failclosed(damage):
    es = events(); c = archive(es)
    if damage in ('dtype','shape','rawhash'):
        key = next(k for k,v in c.objects.items() if v[:1] == b'B')
        raw = bytearray(c.objects[key]); raw[1 if damage == 'dtype' else 3 if damage == 'shape' else -1] ^= 1
        c.objects[key] = bytes(raw)
        with pytest.raises(ValueError): c.verify_events()
    elif damage == 'trailing':
        key = c.events[0]; c.objects[key] += b'x'
        with pytest.raises(ValueError): c.verify_events()
    elif damage == 'ref':
        raw = b'L'+struct.pack('<I',1)+b'f'*32; key = c.intern(raw)
        with pytest.raises(ValueError,match='ref'): c.get(key)
    elif damage == 'static':
        c.static['label'] = 'different'
        with pytest.raises(ValueError,match='sealed'): c.verify_events()
    elif damage == 'delta27':
        e = next(e for e in es if e['payload']['payload']['kind'] == 'resource_return')
        del e['payload']['payload']['entry']['delta']['grad_API']
        with pytest.raises(ValueError): archive([e])
    else:
        a = [wrapped(dict(kind='native_call_attempt',method='replay',task={'RAM':True},request={'RAM':True}),slot=0,call=0),
             wrapped(dict(kind='native_call_return',result=[]),slot=0,call=0)]
        c = archive(a); entry = c.virtual_capsule(0,0,[x['payload']['payload'] for x in a])
        altered = dict(entry,slot=1); c.capsules[entry['path']] = altered
        with pytest.raises(ValueError,match='identity'): c.read_virtual(altered)


def test_command_roundtrip_and_verify_are_explicit_RAM_only(tmp_path):
    a = subprocess.run([sys.executable,'-m',m.__name__,'roundtrip-demo','--output',str(tmp_path/'demo')],
        cwd=m.ROOT,check=True,text=True,capture_output=True)
    info = json.loads(a.stdout)
    b = subprocess.run([sys.executable,'-m',m.__name__,'verify-fixture','--input',info['path'],
        '--fixture-sha',info['fixture_sha256']],cwd=m.ROOT,check=True,text=True,capture_output=True)
    result = json.loads(b.stdout)
    assert result['events'] == 2 and result['objects'] < 4 and result['production_ready'] is False
