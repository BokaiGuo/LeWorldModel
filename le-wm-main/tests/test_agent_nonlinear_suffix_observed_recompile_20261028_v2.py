"""RAM arithmetic + real saved schema only; never load a model/environment."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import pytest
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v2 as m
import test_agent_nonlinear_suffix_observed_recompile_20261028_v1 as n

inputs=n.inputs
complete=n.complete


class RAM(n.parent.RAM):
    def step(self,c,a,label):return super().step(c,a,label.removeprefix('observed_'))
    def gradient(self,c,a,label):return super().gradient(c,a,label.removeprefix('observed_'))


def spec_for(complete,i=0,overcast=False):
    q,_,_=n.make_request(complete,i,overcast)
    after=n.Arithmetic(q['task']).after(q['observed_context'],q['incumbent']['commands'])
    return dict(index=i,root_slot=q['root_slot'],condition=q['condition'],status='ready',request=q,
        parent_selection_digest='a'*64,parent_plan_sha256='b'*64,midpoint_row_binding_sha256='c'*64,
        observed_incumbent_prediction=after,observed_incumbent_Q=m.F.cost(q['task']['goal'],after))


@pytest.fixture(scope='module')
def compiled(complete):
    spec=spec_for(complete);journal=n.parent.Events();ledger=m.core.Ledger(m.F.caps(18,8),sink=journal.event)
    result=m.compile_row(spec,RAM(spec['request']['task'],journal,ledger),journal)
    return spec,result,ledger.receipt(),journal.events


def test_native18_8_plus_independent_replay(compiled):
    s,r,receipt,events=compiled;plan=m.audit_row(s,r,receipt,events)
    assert receipt['attempted']==receipt['completed']==m.F.caps(18,8)
    assert len([e for e in events if e['kind']=='forward_started'])==18
    assert plan['new_suffix_Q']<=plan['observed_incumbent_Q']
    assert plan['commands'][:5]==s['request']['incumbent']['prefix']
    assert plan['new_witness_sha256']==m.digest(r['numeric'])
    assert plan['parent_prediction_not_new_claim'] and not plan['physical_release']
    assert r['numeric']['actual_native_resources'] is None


@pytest.mark.parametrize('field,value',[
    ('slot',True),('root_slot',False),('parent_iteration',True),('parent_prefix_index',1.),
    ('cap',True),('solver_float64_norm',float('nan')),('raw',True),('latent',float('inf')),
    ('Q',True),('Q',float('nan')),('winner',False),('iteration',True),('cost',True)])
def test_strict_literal_scalar_types(compiled,field,value):
    s,r,_,_=deepcopy(compiled);q=s['request']
    if field in ('slot','root_slot'):q[field]=value
    elif field in ('parent_iteration','parent_prefix_index'):q['incumbent'][field]=value
    elif field in ('cap','solver_float64_norm'):q['incumbent']['original_ridge_domain'][field]=value
    elif field=='raw':q['observed_raw10'][0][0]=value
    elif field=='latent':q['observed_context']['latent_history'][0][0]=value
    elif field=='Q':r['numeric']['new_suffix_Q']=value
    elif field=='winner':r['numeric']['winner_iteration']=value
    elif field=='iteration':r['numeric']['iterates'][1]['iteration']=value
    else:r['numeric']['scheduled_cost']['predict']=value
    with pytest.raises(ValueError):m.strict_numeric(q,r['numeric'],r['numeric_events'])


@pytest.mark.parametrize('bad',['unknown','module','gradient','scalar_vector','AD_fields','drop','Q','parent','context','extra'])
def test_full_native_and_plan_mutations_rejected(compiled,bad):
    s,r,receipt,events=deepcopy(compiled)
    if bad=='unknown':events.insert(1,dict(kind='unaccounted_call'))
    elif bad in ('module','gradient','scalar_vector'):
        key={'module':'predict:output','gradient':'value','scalar_vector':'grad:loss'}[bad]
        e=next(e for e in events if key in e.get('arrays',{}));a=m.signal.unblob(e['arrays'][key]).copy()
        if bad=='scalar_vector':a=np.repeat(a,2)
        else:a.flat[0]+=.1
        e['arrays'][key]=m.signal.blob(a)
    elif bad=='AD_fields':next(e for e in events if e['kind']=='AD_API_attempt')['scalar_outputs']=True
    elif bad=='drop':events.remove(next(e for e in events if e['kind']=='program_packet'))
    elif bad=='Q':r['plan']['new_suffix_Q']+=1
    elif bad=='parent':r['plan']['parent_plan_sha256']='0'*64
    elif bad=='context':r['plan']['observed_context_sha256']='0'*64
    else:r['extra']='unapproved'
    with pytest.raises((ValueError,StopIteration)):m.audit_row(s,r,receipt,events)


@pytest.mark.parametrize('move',['packet','AD','numeric_return','selection','replay'])
def test_complete_chronology_rejects_splice(compiled,move):
    s,r,receipt,events=deepcopy(compiled)
    if move=='packet':
        e=next(e for e in events if e['kind']=='program_packet' and e['phase']=='f000')
        target=next(e for e in events if e['kind']=='forward_started' and e['phase']=='f001')
    elif move=='AD':
        e=next(e for e in events if e['kind']=='AD_API_completed')
        target=next(e for e in events if e['kind']=='forward_started' and e['phase']=='f002')
    elif move=='numeric_return':
        e=next(e for e in events if e['kind']=='numeric_event' and e['event']['kind']=='observed_gradient_return')
        target=next(e for e in events if e['kind']=='AD_API_completed')
    elif move=='selection':
        e=[e for e in events if e['kind']=='numeric_event'][-1]
        target=next(e for e in events if e['kind']=='forward_started' and e['phase']=='f016')
    else:
        e=next(e for e in events if e['kind']=='winner_replay_intent')
        target=next(e for e in events if e['kind']=='forward_started' and e['phase']=='f017')
    events.remove(e);events.insert(events.index(target)+(move in ('AD','replay')),e)
    with pytest.raises(ValueError):m.audit_row(s,r,receipt,events)


def test_original_overcast_incumbent_not_repaired(complete):
    s=spec_for(complete,overcast=True);j=n.parent.Events();l=m.core.Ledger(m.F.caps(18,8),sink=j.event)
    r=m.compile_row(s,RAM(s['request']['task'],j,l),j);m.audit_row(s,r,l.receipt(),j.events)
    q=s['request'];d=q['incumbent']['original_ridge_domain']
    assert d['solver_float64_norm']<=m.F.CAP<d['postcast_float64_norm']
    assert r['numeric']['iterates'][0]['raw_bits']==q['incumbent']['raw_bits']
    assert r['numeric']['iterates'][0]['logits']==q['incumbent']['logits']


@pytest.mark.parametrize('fail',[False,True])
def test_actual_durable_dispatch_all4_and_partial(complete,tmp_path,monkeypatch,fail):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    p=dict(rows=[spec_for(complete,i) for i in range(4)],production_ready=True);rows=m.roster()
    store=m.Store(tmp_path/'results'/(m.NAME+'_RAM'),time.monotonic())
    count=0
    def factory(t,j,l):
        nonlocal count
        count+=1
        if fail and count==2:raise RuntimeError('RAM constructor failure')
        return RAM(t,j,l)
    if fail:
        with pytest.raises(RuntimeError):m.dispatch(p,store,rows,lambda:None,factory)
        assert [x['status'] for x in rows]==['sealed','failed_partial','not_started','not_started']
        assert rows[1]['resources']['completed']==m.F.caps(0,0)
    else:
        m.dispatch(p,store,rows,lambda:None,factory)
        assert sum(x['resources']['completed']['predict_API'] for x in rows)==72
        assert sum(x['resources']['completed']['grad_API'] for x in rows)==32
    assert len(rows)==4
    for spec,row in zip(p['rows'],rows):
        if row['status']=='not_started':continue
        events=m.F.load_journal(store.path,row);m.transport.audit_resources(row['resources'],events)
        if row['status']=='sealed':m.audit_row(spec,row['result'],row['resources'],events)
    # Exercise whole consume/census/partial semantics. The ONLY substituted
    # authority is upstream verify: RAM is explicitly not a real prepare.
    model_hash='d'*64
    p.update(expected_model_hash=model_hash,bindings={},inherited_resources={},midpoint_authority={},original_physical_authority={})
    prepared=tmp_path/'RAM_prepare.json';prepared.write_bytes(m.raw(p));h=m.sha(prepared)
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=h))
    monkeypatch.setattr(m,'verify',lambda path,expected:p if str(path)==str(prepared) and expected==h else m.require(False,'RAM binding'))
    precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    artifacts={str(f.relative_to(store.path)):m.sha(f) for f in store.path.rglob('*') if f.is_file()}
    report=dict(schema=m.NAME+'.report',contract=m.contract(),prepared=str(prepared),prepare_sha256=h,
        rows=rows,fixed_denominator=4,known_roots=2,errors=['RAM failure'] if fail else [],
        status='partial_failclosed' if fail else 'observed_suffix4_recompiled_not_admitted',
        P1_admission=False,scientific_admission=False,freshness=False,physical_release=False,
        artifact_sha256=artifacts,source_verified=True,model_unchanged=True,
        model_before=dict(original_state_dict=model_hash),model_after=dict(original_state_dict=model_hash),
        source_manifest_sha256=m.digest({}),precision_before=precision,precision_after=precision,
        wall_seconds=1.,CUDA_peak_bytes=[0,0])
    store.put('report.json',report,final=True)
    receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    assert receipt['complete'] is (not fail)
    assert len(receipt['plans'])==(0 if fail else 4) and len(receipt['rows'])==4
    assert receipt['physical_metrics'] is None and not receipt['physical_release']
    (store.path/'unaccounted.txt').write_text('unexpected')
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))


def test_actual_frozen_sources_and_precost_contract():
    m.check_pins();c=m.contract()
    assert c['total']['predict']==72 and c['total']['scalar_reverse']==32
    assert c['independent_replay']==4 and c['new_visual']==c['new_physics']==0
    assert not c['production_run_authorized'] and not c['P1_admission']
    with pytest.raises(ValueError):m.run('/nonexistent','0'*64,'/nonexistent','unauthorized')


def test_binding_and_source_tamper_failclosed(tmp_path):
    f=tmp_path/'source.py';f.write_text('sealed fixture\n');bindings={str(f):m.sha(f)}
    m.F.check_bindings(bindings);f.write_text('changed fixture\n')
    with pytest.raises(ValueError):m.F.check_bindings(bindings)
    with pytest.raises(ValueError):m.verify(f,'0'*64)


def test_scalar_transport_only_one_element_and_rawleaf_not_relaxed(compiled):
    _,_,_,es=compiled;e=next(e for e in es if 'grad:loss' in e.get('arrays',{}))
    assert m.signal.unblob(m.signal.blob(np.array(1.,np.float32))).shape==(1,)
    assert m.signal.unblob(e['arrays']['grad:loss']).shape==(1,)
    assert m.signal.unblob(e['arrays']['grad:raw']).shape==(5,2)


@pytest.fixture(scope='module')
def saved_layout():
    """Read existing exact metadata/arrays, NOT repeat strict or claim acceptance.

    Entire authority must still be reconstructed by production build(). This
    regression solely catches literal schema joins against MAIN's actual ABI.
    """
    root=m.ROOT/'results';stem=m.midpoint.NAME
    prepared=root/(stem+'_prepare_MAIN01.json');output=root/(stem+'_cuda_MAIN01')
    receipt=root/(stem+'_consume_MAIN01.json')
    if not prepared.exists():pytest.skip('optional exact MAIN saved layout unavailable')
    assert m.sha(prepared)=='6dd4ef991ebf2947388ae7c139a23776592b200e11032fdc106be281ae9cc5e8'
    assert m.sha(output/'report.json')=='1639745d33424a28e39a36924b4fe161a00aed2f5a6ae035352c898a9c8e88a0'
    assert m.sha(receipt)=='127c650eace2c41e43dc0361aa13009981acfc8e9509dfff7a1f924283f87521'
    p=m.midpoint.read(prepared,64*1024**2);r=m.midpoint.read(output/'report.json',64*1024**2)
    accepted=m.midpoint.read(receipt,64*1024**2);a=p['physical_authority']
    c=m.midpoint.read(a['producer_prepared'],16*1024**2)
    parent=m.midpoint.read(Path(a['producer_output'])/'report.json',64*1024**2)
    return p,r,accepted,c,parent,output


def test_exact_real_saved_midpoint_parent_material_schema(saved_layout):
    p,r,a,c,parent,out=saved_layout
    rows=m.material(p,r,a,c,parent,lambda ref:m.midpoint.load_saved_array(out,ref))
    assert len(rows)==4 and all(s['status']=='ready' for s in rows)
    for i,s in enumerate(rows):
        q=s['request'];assert (q['root_slot'],q['condition'])==m.numeric.ORDER[i]
        m.validate_request(q)
        assert q['incumbent']['prefix']+q['incumbent']['commands']==p['rows'][i]['full10']
        assert not {'state','reward','success','native_evaluation'} & set(q)
        assert s['observed_incumbent_Q']==a['rows'][i]['conditioned_final_Q']


@pytest.mark.parametrize('bad',['parent_raw','goal','normalizer','task_binding','missing','array','receipt_status'])
def test_exact_saved_schema_binding_mutations(saved_layout,bad):
    op,orr,oa,c,oparent,out=saved_layout
    p,r,a,parent=deepcopy(op),deepcopy(orr),deepcopy(oa),deepcopy(oparent)
    if bad=='parent_raw':parent['finals'][0]['plan']['commands'][0][0]+=.1
    elif bad=='goal':p['rows'][0]['goal192'][0]+=.1
    elif bad=='normalizer':p['rows'][0]['action_std'][0]+=.1
    elif bad=='task_binding':p['rows'][0]['model_task_binding']['task_id']='0'*64
    elif bad=='missing':p['rows'][0]['eligible']=False
    elif bad=='receipt_status':a['status']='unverified'
    def load(ref):
        x=m.midpoint.load_saved_array(out,ref)
        if bad=='array':x=x.astype(np.float64)
        return x
    with pytest.raises(ValueError):m.material(p,r,a,c,parent,load)
