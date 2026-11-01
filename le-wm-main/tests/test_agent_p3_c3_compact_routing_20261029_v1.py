"""Only RAM callbacks/literal source metadata. No production build or strict."""
from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_c3_compact_routing_20261029_v1 as m


@pytest.fixture(scope='module')
def upstream():
    return runpy.run_path(str(Path(__file__).resolve().parent/'test_agent_p3_c3_common_cap_cuda_20261029_v1.py'))


def static(upstream,extra=None,sources=None):
    t=upstream['task']()
    return m.seal_static([t,deepcopy(t)],sources or {},{} if extra is None else extra)


def bind(catalog,kind,value):
    return catalog.bind(kind,value,catalog.store.put(value))


def external_tree(catalog,r,t):
    """RAM *reference* fixture, not certification of native journal ownership.

    Real values come from NumpyNative complete callbacks. This helper encodes
    the external side separately; structural routing must not copy its data.
    It intentionally does not pretend generic transcript lists are codec views.
    """
    bind(catalog,'task',t)
    def visit(value):
        if type(value) is dict:
            keys=set(value)
            if keys=={'layout','latent_dim','context_length','action_dim','latent_history','action_history'}:
                bind(catalog,'context',value)
            if keys=={'binding','root_context','goal','projection','action_mean','action_std'}:
                bind(catalog,'task',value)
            if keys=={'binding_sha256','parent_context_sha256','horizon','nominal','jacobian','producer_sha256'}:
                bind(catalog,'native_literal',value['nominal']); bind(catalog,'native_literal',value['jacobian'])
            for v in value.values(): visit(v)
        elif type(value) is list:
            if m.codec.rectangle(value) is None:
                for v in value: visit(v)
    visit(r)
    for call in r['native_calls']: bind(catalog,'transcript',call['transcript'])
    bind(catalog,'ledger',r['resource_receipt']['events'])
    bind(catalog,'inclusive',r['resource_receipt']['inclusive_operation_wall_ns'])
    for group in r['R_groups']:
        b=group['bank']
        for v in (b['rule'],b['authority_manifest']['sources'],b['authority_manifest']['numeric_environment']): bind(catalog,'static',v)
    for event in r['resource_receipt']['events']: bind(catalog,'scope',event['scope'])


@pytest.fixture(scope='module')
def saved(upstream):
    return upstream['one']()


@pytest.fixture(scope='module')
def routed(upstream,saved):
    return router_for(upstream,saved)


def router_for(upstream,saved):
    t,r,s=saved
    seal=static(upstream,extra=dict(fields=sorted(m.codec.keys_of(r))))
    # This is explicit RAM discovery before the seal, not permission to extend
    # an installed field table when the producer emits an unexpected key.
    store,catalog,budget=m.install_static(seal)
    external_tree(catalog,r,t)
    return m.Routing(store,catalog),budget


def test_static_source_actual_read_length_sha_and_pack_budget(upstream):
    source=Path(m.__file__).resolve(); source_sha=m.codec.file_sha(source)
    seal=static(upstream,sources={str(source):source_sha})
    store,catalog,budget=m.install_static(seal)
    assert budget['source_bytes_checked'] is True
    assert budget['source_entries']==[dict(path=str(source),sha256=source_sha,bytes=source.stat().st_size)]
    assert budget['total_static_bytes']==store.nodes_bytes+store.static_bytes<=32*1024**2
    assert m.canonical(store.get(catalog.get('task',seal['value']['tasks'][0])))==m.canonical(seal['value']['tasks'][0])
    with pytest.raises(ValueError,match='source bytes exact SHA'):
        static(upstream,sources={str(source):'0'*64})


def test_static_overflow_and_mutation_no_default_authority(upstream):
    with pytest.raises(ValueError,match='static/field seal'):
        static(upstream,extra={'too_large':'x'*(32*1024**2)})
    seal=static(upstream); seal['value']['tasks'][0]['goal'][0]+=1.
    with pytest.raises(ValueError,match='static exact bytes'): m.install_static(seal)


def test_static_source_symlink_rejected(tmp_path,upstream):
    target=tmp_path/'source.py'; target.write_bytes(b'RAM source\n')
    link=tmp_path/'link.py'; link.symlink_to(target)
    with pytest.raises(ValueError,match='regular source path'):
        static(upstream,sources={str(link):m.codec.file_sha(target)})


def test_static_field_table_cannot_be_extended_after_seal(upstream):
    seal=static(upstream); seal['fields'].append('unexpected_after_seal')
    with pytest.raises(ValueError,match='exact static field'): m.install_static(seal)


def test_real_full_R_tree_every_field_routed_and_exact_expanded(saved,routed):
    t,r,s=saved; routing,budget=routed
    result=routing.tree(0,r)
    key=bytes.fromhex(result['object_sha256'])
    assert m.canonical(routing.store.get(key))==m.canonical(r)
    assert result['candidate_identity_count']==144 and result['parent_count']<=17
    census=routing.census()
    assert all(b['bytes']<=b['cap'] for b in census['buckets'])
    # All 144 jobs including never-dispatched identities are structural records.
    assert all(routing.catalog.contains('job',j) for j in r['jobs'])
    refs=[x for x in routing.trace if x['owner'] is None]
    assert {'context','native_literal','task','transcript','ledger','inclusive','static'}<={x['route'] for x in refs}
    assert any('.native_calls[' in x['path'] and x['route']=='transcript' for x in refs)
    assert not census['production_ready'] and not census['full_C3_demonstrated']
    print(m.canonical(dict(label='REAL_DIMENSIONAL_RAM_ONLY',static_budget=budget,
        structural_bytes=census['unique_structural_bytes'],structural_objects=census['unique_structural_objects'],
        max_by_bucket={k:max((b['bytes'] for b in census['buckets'] if b['owner'][0]==k),default=0)
                       for k in m.LIMITS},routed_fields=len(routing.trace))).decode())


def test_same_parent_queue_cannot_hide_both_bank_types(saved,routed):
    routing,_=routed
    with pytest.raises(ValueError,match='exclusive queue branch'):
        routing.tree(0,saved[1],progress_banks=[(0,{})])


def test_duplicate_context_must_not_merge_distinct_parent_owners(saved,routed):
    routing,_=routed
    r=saved[1]
    result=routing.tree(0,r)
    # Parent owner ID is queue ordinal, even if future trajectories happen to
    # revisit identical context bytes. No assumption of numerical noncollision.
    parents=[b['owner'] for b in routing.census()['buckets'] if b['owner'][0]=='parent']
    assert len(parents)==result['parent_count']
    assert all(type(o[2]) is int and 0<=o[2]<17 for o in parents)


def test_actual_fixture_static_budget_is_concrete_not_32MiB_envelope(routed):
    routing,budget=routed
    assert budget['total_static_bytes']<2*1024**2
    assert budget['canonical_bytes']>100000
    assert budget['object_and_index_bytes']>0 and budget['field_and_metadata_bytes']>0


def test_failed_native_prefix_no_fabricated_normal_tree(upstream):
    # A second, deliberately failed callback fixture has no parent maps and no
    # plan; it still carries all144 requests and the real attempted ledger.
    class Fail(upstream['Native']):
        def __init__(self,t,s): super().__init__(t,s,fail=True)
    t,r,s=upstream['one'](cls=Fail)
    routing,_=router_for(upstream,(t,r,s))
    ref=routing.tree(0,r)
    assert r['status']=='failed_or_budget_exhausted' and r['model_plan'] is None
    assert len(r['jobs'])==144 and r['resource_receipt']['attempted']['predict_API']==1
    assert m.canonical(routing.store.get(bytes.fromhex(ref['object_sha256'])))==m.canonical(r)


def test_actual_progress_bank_schema_and_K1_cap_saved_all144(upstream):
    t=upstream['f']['fixtures']['original']['task']()
    sample=upstream['one'](mode='fixedH1',K=1,cls=upstream['Native'],t=t)
    _,r,s=sample
    banks=[e['payload']['bank'] for e in s.events if e['kind']=='io' and e['payload']['kind']=='progress_bank']
    ordinals=[i for i,q in enumerate(r['queues']) if q['status']=='queued' and q['decision']['extension_type']=='progress']
    assert len(banks)==len(ordinals)>0
    # Include fields from the actual progress-bank value BEFORE installation,
    # as a fixed RAM source snapshot, never by extending a live field table.
    routing,_=router_for(upstream,sample)
    ref=routing.tree(0,r,progress_banks=list(zip(ordinals,banks,strict=True)))
    assert r['status'].startswith('resource_cap') and r['model_plan'] is None
    assert ref['bank_kinds_by_queue_ordinal']=={i:'progress' for i in ordinals}
    assert all(routing.catalog.contains('progress',b) for b in banks)
    assert m.canonical(routing.store.get(bytes.fromhex(ref['object_sha256'])))==m.canonical(r)


def test_actual_partial_R_control_postcontext_all_eight_null_rows_retained(upstream):
    count=[0]
    class FailPost(upstream['Nonlinear']):
        def __init__(self,t,s):
            count[0]+=1
            super().__init__(t,s,fail=count[0]==4)
        def call(self,name,fn,inputs,shape=None):
            if self.fail and name=='predictor':
                def failure(): raise RuntimeError('RAM injected postcontext predictor failure')
                return upstream['Native'].call(self,name,failure,inputs,shape)
            return super().call(name,fn,inputs,shape)
    sample=upstream['one'](cls=FailPost)
    _,r,_=sample
    assert r['status']=='failed_or_budget_exhausted' and r['model_plan'] is None
    assert len(r['R_groups'])==1
    group=r['R_groups'][0]
    assert group['status']=='partial_runtime_failure'
    assert group['control']['status']=='failed' and 'postcontext' in group['control']
    assert len(group['rows'])==8 and all(x['status']=='not_requested' for x in group['rows'])
    routing,_=router_for(upstream,sample)
    ref=routing.tree(0,r)
    assert all(routing.catalog.contains('proposal',row) for row in group['rows'])
    assert m.canonical(routing.store.get(bytes.fromhex(ref['object_sha256'])))==m.canonical(r)


@pytest.mark.parametrize('damage',['extra','missing','request_field','row_field','candidateid','rootflag','unboundJ','longstring'])
def test_real_shape_reject_unclassified_or_mutated_fields(saved,routed,damage):
    _,original,_=saved; r=deepcopy(original); routing,_=routed
    if damage=='extra': r['unclassified']='no free metadata'
    elif damage=='missing': del r['fixed_denominator']
    elif damage=='request_field': r['jobs'][0]['request']['unclassified']=0
    elif damage=='row_field': r['jobs'][0]['payload']['new_scalar']=0.
    elif damage=='candidateid': r['jobs'][0]['slot']=1
    elif damage=='rootflag': r['P1_admission']=1
    elif damage=='unboundJ': r['signal_mappings'][0]['native_response']['jacobian'][0][0]+=.125
    else: r['status']='x'*257
    with pytest.raises((ValueError,KeyError)): m.validate(r,m.TREE,routing.catalog) if damage not in ('request_field','row_field','candidateid','unboundJ') else routing.tree(0,r)


def test_missing_external_native_reference_refuses_no_fallback(saved,routed):
    routing,_=routed
    identity=('ledger',m.semantic(saved[1]['resource_receipt']['events']))
    key=routing.catalog.values.pop(identity)
    try:
        with pytest.raises(ValueError): m.validate(saved[1],m.TREE,routing.catalog)
    finally: routing.catalog.values[identity]=key


def test_same_reference_target_cannot_be_replaced_by_same_shape(saved,routed):
    routing,_=routed
    value=saved[1]['signal_mappings'][0]['native_response']['nominal']
    wrong=list(value); wrong[0]+=.125
    key=routing.store.put(wrong)
    with pytest.raises(ValueError,match='expanded exact bytes'): routing.catalog.bind('native_literal',value,key)


def maximum(spec,catalog,counter):
    """Adversarial finite fullshape, no optimistic all-zero dedup assumption."""
    k=spec[0]; counter[0]+=1; n=counter[0]
    if k=='ref':
        value='RAM_external_'+spec[1]
        bind(catalog,spec[1],value)
        return value
    if k=='record': return {name:maximum(s,catalog,counter) for name,s in spec[1].items()}
    if k=='list': return [maximum(spec[1],catalog,counter) for _ in range(spec[2])]
    if k=='optional': return maximum(spec[1],catalog,counter)
    if k=='variant': return maximum(max(spec[1],key=m.bound),catalog,counter)
    if k=='array':
        def build(dims):
            if not dims:
                counter[0]+=1; return float(counter[0])+0.25
            return [build(dims[1:]) for _ in range(dims[0])]
        return build(spec[1])
    if k=='str': return ('s'*spec[1])
    if k=='enum': return max(spec[1],key=lambda s:len(s.encode()))
    if k=='uint': return 2**64-1-n
    if k=='signed': return -n
    if k=='float': return float(n)+0.25
    if k=='bool': return True
    return None


@pytest.mark.parametrize('kind',tuple(m.SCHEMAS))
def test_every_schema_maxshape_actual_encoded_bytes_below_analytic_bound(kind):
    fields=set().union(*(m.fields_of(s) for s in m.SCHEMAS.values()))
    store=m.codec.Codec(fields); catalog=m.Catalog(store); routing=m.Routing(store,catalog)
    value=maximum(m.SCHEMAS[kind],catalog,[0]); before=set(store.objects)
    bucket='slot' if kind=='tree' else 'candidate' if kind in ('request','node','proposal','job') else 'parent'
    key=routing.record(kind,value,(bucket,0,0))
    exact=sum(len(store.objects[k])+48 for k in set(store.objects)-before)
    assert exact<=m.bound(m.SCHEMAS[kind])<=m.LIMITS[bucket]
    assert m.canonical(store.get(key))==m.canonical(value)
    assert routing.census()['unique_structural_bytes']==exact


def test_parent_combined_fullshape_bucket_not_one_record_average():
    fields=set().union(*(m.fields_of(s) for s in m.SCHEMAS.values()))
    store=m.codec.Codec(fields); cat=m.Catalog(store); routing=m.Routing(store,cat)
    counter=[0]
    # Two complete parent mappings + both J signal diagnostics + queue + whole
    # SVD bank + explicit zero row + R group. No production outputs read.
    for kind in ('mapping','mapping','signal','queue','bank','control','group'):
        value=maximum(m.SCHEMAS[kind],cat,counter)
        routing.record(kind,value,('parent',0,0))
    assert routing.bytes['parent',0,0]<=m.proof()['per_owner_bytes_upper']['parent']<=512*1024


def test_integer_full12_proof_does_not_increase_cap():
    p=m.proof(); base=m.codec.proof()
    assert p['all_structural_buckets_fit']
    assert p['per_owner_bytes_upper']==dict(parent=242544,candidate=28200,slot=215424)
    assert base['total_bytes_upper']==3467738560<4*1024**3
    assert not p['production_control_flow_connected'] and not p['production_ready']


def test_cold_import_and_only_RAM_CLI():
    script=f'import sys; import {m.__name__}; assert not any(n in sys.modules for n in ("torch","numpy","pickle","gymnasium"))'
    subprocess.run([sys.executable,'-c',script],check=True)
    out=subprocess.run([sys.executable,'-m',m.__name__,'proof'],capture_output=True,text=True,check=True)
    assert '"production_ready":false' in out.stdout


def test_actual_maxshape_diskCAS_routing_roundtrip_and_exclusive_reopen(tmp_path):
    fields=set().union(*(m.fields_of(s) for s in m.SCHEMAS.values()))
    store=m.codec.Codec(fields); catalog=m.Catalog(store); routing=m.Routing(store,catalog)
    value=maximum(m.BANK,catalog,[0])
    directory=tmp_path/'exclusive_CAS'
    disk=m.attach_disk(store,directory,budget=8*1024**2)
    key=routing.record('bank',value,('parent',0,0)); descriptor=disk.finish()
    assert m.canonical(store.get(key))==m.canonical(value)
    assert disk.offset+48*len(disk)==store.nodes_bytes
    assert descriptor['objects']==len(disk)
    disk.close()
    with pytest.raises(ValueError,match='new exclusive'): m.DiskCAS(directory,budget=8*1024**2)
    saved=m.DiskCAS.open_saved(directory,descriptor)
    store.objects=saved
    try:
        assert m.canonical(store.get(key))==m.canonical(value)
        with pytest.raises(ValueError,match='writable'): saved[key]=saved[key]
        with pytest.raises(ValueError,match='no deletion'): del saved[key]
    finally: saved.close()


@pytest.mark.parametrize('damage',['pack','index','extra','length','boundary'])
def test_diskCAS_saved_corruption_census_size_refuses(tmp_path,damage):
    directory=tmp_path/'CAS'; disk=m.DiskCAS(directory,budget=4096)
    raw=b'Sactual raw bytes'; key=m.codec.h(m.codec.DOMAIN+raw); disk[key]=raw
    descriptor=disk.finish(); disk.close()
    if damage=='pack': (directory/'objects.pack').write_bytes(b'X'+raw[1:])
    elif damage=='index': (directory/'objects.idx').write_bytes(b'X'+(directory/'objects.idx').read_bytes()[1:])
    elif damage=='extra': (directory/'fake_capsule.json').write_bytes(b'{}')
    elif damage=='length': descriptor['objects']+=1
    else: descriptor['boundary']='report sampled before fsync'
    with pytest.raises(ValueError): m.DiskCAS.open_saved(directory,descriptor)


def test_diskCAS_preappend_bound_no_refund_preserves_prefix(tmp_path):
    disk=m.DiskCAS(tmp_path/'CAS',budget=128)
    try:
        raw=b'Skept'; key=m.codec.h(m.codec.DOMAIN+raw); disk[key]=raw
        with pytest.raises(ValueError,match='byte allowance'): disk[m.codec.h(m.codec.DOMAIN+b'S'+b'x'*128)]=b'S'+b'x'*128
        assert len(disk)==1 and disk[key]==raw
        disk[key]=raw; assert len(disk)==1
        assert disk.finish()['pack_bytes']==len(raw)
    finally: disk.close()
